import torch
import os
import time
import logging
import random
import glob
import segmentation_models_pytorch as smp
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from losses import *
from augmentation import *
import cv2
import albumentations as A
from dataset import *
from torch.nn import functional as F
from iternetmodel import Iternet


def train(model, data_loader, optimizer, criterion, device, writer, save_path, training=False):
    losses = AverageMeter("Loss", ".16f")
    dices1 = AverageMeter("Dice", ".8f")
    jaccards1 = AverageMeter("Jaccard", ".8f")
    dices2 = AverageMeter("Dice", ".8f")
    jaccards2 = AverageMeter("Jaccard", ".8f")
    dices3 = AverageMeter("Dice", ".8f")
    jaccards3 = AverageMeter("Jaccard", ".8f")
    dices4 = AverageMeter("Dice", ".8f")
    jaccards4 = AverageMeter("Jaccard", ".8f")

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    process = tqdm(data_loader)
    for i, (inputs, targets) in enumerate(process):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad()

        logit = model(inputs)
        raw_logits = logit[0]
        raw_outputs = torch.sigmoid(raw_logits)
        raw_preds = torch.round(raw_outputs)

        refine_logits = logit[1]
        refine_outputs = torch.sigmoid(refine_logits)
        refine_preds = torch.round(refine_outputs)

        last_logits = logit[2]
        last_outputs = torch.sigmoid(last_logits)
        last_preds = torch.round(last_outputs)

        final_logits = logit[3]
        final_outputs = torch.sigmoid(final_logits)
        final_preds = torch.round(final_outputs)

        targets = torch.unsqueeze(targets, 1)

        loss_criterion, dice_criterion, jaccard_criterion = criterion[
            0], criterion[1], criterion[2]
        
        loss1 = loss_criterion(raw_outputs, targets.float())
        loss2 = loss_criterion(refine_outputs, targets.float())
        loss3 = loss_criterion(last_outputs, targets.float())
        loss4 = loss_criterion(final_outputs, targets.float())
        loss = (loss1 + loss2 + loss3 + loss4) / 4

        dice1 = 1 - dice_criterion(raw_preds, targets.to(torch.int64))
        jaccard1 = 1 - jaccard_criterion(raw_preds, targets.to(torch.int64))
        dices1.update(dice1.item(), inputs.size(0))
        jaccards1.update(jaccard1.item(), inputs.size(0))

        dice2 = 1 - dice_criterion(refine_preds, targets.to(torch.int64))
        jaccard2 = 1 - jaccard_criterion(refine_preds, targets.to(torch.int64))
        dices2.update(dice2.item(), inputs.size(0))
        jaccards2.update(jaccard2.item(), inputs.size(0))

        dice3 = 1 - dice_criterion(last_preds, targets.to(torch.int64))
        jaccard3 = 1 - jaccard_criterion(last_preds, targets.to(torch.int64))
        dices3.update(dice3.item(), inputs.size(0))
        jaccards3.update(jaccard3.item(), inputs.size(0))

        dice4 = 1 - dice_criterion(final_preds, targets.to(torch.int64))
        jaccard4 = 1 - jaccard_criterion(final_preds, targets.to(torch.int64))
        dices4.update(dice4.item(), inputs.size(0))
        jaccards4.update(jaccard4.item(), inputs.size(0))

        if training:
            loss.backward()
            optimizer.step()

        process.set_description('Loss: ' + str(round(losses.avg, 4)))

    epoch_dice1 = dices1.avg
    epoch_jaccard1 = jaccards1.avg
    epoch_dice2 = dices2.avg
    epoch_jaccard2 = jaccards2.avg
    epoch_dice3 = dices3.avg
    epoch_jaccard3 = jaccards3.avg
    epoch_dice4 = dices4.avg
    epoch_jaccard4 = jaccards4.avg

    return epoch_dice1, epoch_jaccard1, epoch_dice2, epoch_jaccard2, epoch_dice3, epoch_jaccard3, epoch_dice4, epoch_jaccard4


def main():

    args = create_train_arg_parser().parse_args()
    CUDA_SELECT = "cuda:{}".format(args.cuda_no)

    log_path = os.path.join(args.save_path, "summary/")
    writer = SummaryWriter(log_dir=log_path)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = os.path.join(log_path, str(rq) + '.log')
    logging.basicConfig(
        filename=log_name,
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info(args)
    print(args)

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    model = Iternet(1, 1)

    logging.info(model)
    model = model.to(device)

    if args.set == '6M':
        # 6M
        train_num_list = list(range(10001, 10181))
        val_num_list = list(range(10181, 10201))
    elif args.set == '3M':
        # 3M
        train_num_list = list(range(10301, 10441))
        val_num_list = list(range(10441, 10451))

    m, s = mean_and_std(args.data_path, train_num_list)
    print(m, s)

    train_data = TrainDataSet(data_path=args.data_path, mask_path=args.mask_path, transform=train_aug(m, s), num_list=train_num_list, class_num=args.classnum)
    val_data = TestDataSet(data_path=args.val_data_path, mask_path=args.val_mask_path, transform=val_aug(m, s), num_list=val_num_list, class_num=args.classnum)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.LR_seg}
    ])

    if args.loss_type == 'dice':
        criterion = [
            define_loss(args.loss_type, args.classnum),
            train_DiceLoss(mode='binary'),
            train_JaccardLoss(mode='binary')
        ]
    elif args.loss_type == 'ce' or args.loss_type == 'bce':
        criterion = [
            define_loss(args.loss_type, args.classnum),
            train_DiceLoss_logit(mode='binary'),
            train_JaccardLoss_logit(mode='binary')
        ]
    else :
        criterion = [
            define_loss(args.loss_type, args.classnum),
            train_DiceLoss(mode='binary'),
            train_JaccardLoss(mode='binary')
        ]

    max_dice = 0.86
    epoch_start = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

        print('\nEpoch: {}'.format(epoch))

        train_dice1, train_jaccard1, train_dice2, train_jaccard2, train_dice3, train_jaccard3, train_dice4, train_jaccard4 = train(
            model, train_loader, optimizer, criterion, device, writer, args.save_path, training=True)
        val_dice1, val_jaccard1, val_dice2, val_jaccard2, val_dice3, val_jaccard3, val_dice4, val_jaccard4 = train(
            model, val_loader, optimizer, criterion, device, writer, args.save_path, training=False)

        epoch_info = "Epoch: {}".format(epoch)
        train_info = "Dice1:{:.4f}, Jaccard1:{:.4f} Dice2:{:.4f}, Jaccard2:{:.4f} Dice3:{:.4f}, Jaccard3:{:.4f} Dice4:{:.4f}, Jaccard4:{:.4f}".format(
            train_dice1, train_jaccard1, train_dice2, train_jaccard2, train_dice3, train_jaccard3, train_dice4, train_jaccard4)
        val_info = "Dice1:{:.4f}, Jaccard1:{:.4f}, Dice2:{:.4f}, Jaccard2:{:.4f}, Dice3:{:.4f}, Jaccard3:{:.4f} Dice4:{:.4f}, Jaccard4:{:.4f}".format(
            val_dice1, val_jaccard1, val_dice2, val_jaccard2, val_dice3, val_jaccard3, val_dice4, val_jaccard4)
        print(train_info)
        print(val_info)
        logging.info(epoch_info)
        logging.info(train_info)
        logging.info(val_info)
        writer.add_scalar("train_dice1", train_dice1, epoch)
        writer.add_scalar("train_jaccard1", train_jaccard1, epoch)
        writer.add_scalar("train_dice2", train_dice2, epoch)
        writer.add_scalar("train_jaccard2", train_jaccard2, epoch)
        writer.add_scalar("train_dice3", train_dice3, epoch)
        writer.add_scalar("train_jaccard3", train_jaccard3, epoch)
        writer.add_scalar("train_dice4", train_dice4, epoch)
        writer.add_scalar("train_jaccard4", train_jaccard4, epoch)
        writer.add_scalar("val_dice1", val_dice1, epoch)
        writer.add_scalar("val_jaccard1", val_jaccard1, epoch)
        writer.add_scalar("val_dice2", val_dice1, epoch)
        writer.add_scalar("val_jaccard2", val_jaccard1, epoch)
        writer.add_scalar("val_dice3", val_dice3, epoch)
        writer.add_scalar("val_jaccard3", val_jaccard3, epoch)
        writer.add_scalar("val_dice4", val_dice4, epoch)
        writer.add_scalar("val_jaccard4", val_jaccard4, epoch)

        val_dice = max(val_dice1, max(val_dice2, max(val_dice3, val_dice4)))
        val_jaccard = max(val_jaccard1, max(val_jaccard2, max(val_jaccard3, val_jaccard4)))

        best_name = os.path.join(args.save_path, "best_dice_" + str(
            round(val_dice, 4)) + "_jaccard_" + str(round(val_jaccard, 4)) + ".pt")
        save_name = os.path.join(args.save_path, str(epoch) + "_dice_" + str(
            round(val_dice, 4)) + "_jaccard_" + str(round(val_jaccard, 4)) + ".pt")

        if max_dice < val_dice:
            max_dice = val_dice
            if max_dice > 0.86:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best model saved!')
                logging.warning('Best model saved!')
        if epoch % 200 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))
            else:
                torch.save(model.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))


if __name__ == "__main__":
    main()

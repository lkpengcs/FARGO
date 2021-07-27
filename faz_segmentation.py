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


def train(model, data_loader, optimizer, criterion, device, writer, save_path, training=False):
    losses = AverageMeter("Loss", ".16f")
    dices = AverageMeter("Dice", ".8f")
    jaccards = AverageMeter("Jaccard", ".8f")
    faz_dices = AverageMeter("Dice", ".8f")
    faz_jaccards = AverageMeter("Jaccard", ".8f")

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

        outputs = model(inputs)
        preds = torch.round(outputs)

        loss_criterion, dice_criterion, jaccard_criterion = criterion[
            0], criterion[1], criterion[2]

        fazloss, loss = loss_criterion(outputs, targets.to(torch.int64))

        faz_dice, dice = dice_criterion(preds, targets.to(torch.int64))
        faz_dice = 1 - faz_dice
        dice = 1 - dice
        faz_jaccard, jaccard = jaccard_criterion(preds, targets.to(torch.int64))
        faz_jaccard = 1 - faz_jaccard
        jaccard = 1 - jaccard
        dices.update(dice.item(), inputs.size(0))
        jaccards.update(jaccard.item(), inputs.size(0))
        faz_dices.update(faz_dice.item(), inputs.size(0))
        faz_jaccards.update(faz_jaccard.item(), inputs.size(0))

        if training:
            loss.backward()
            optimizer.step()

        process.set_description('Loss: ' + str(round(losses.avg, 4)))

    epoch_dice = dices.avg
    epoch_jaccard = jaccards.avg
    epoch_faz_dice = faz_dices.avg
    epoch_faz_jaccard = faz_jaccards.avg

    return epoch_dice, epoch_jaccard, epoch_faz_dice, epoch_faz_jaccard


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
    print(args.encoder)

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    pretrain = args.pretrain
    model = build_model(args.model_type, args.encoder, pretrain, args.classnum, args.encoder_depth, args.activation)
   
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

    if args.loss_type == 'dice':
        train_data = TrainDataSet(data_path=args.data_path, mask_path=args.mask_path,
                                transform=train_crop_aug(m, s), num_list=train_num_list, class_num=args.classnum)
        val_data = TestDataSet(data_path=args.val_data_path, mask_path=args.val_mask_path,
                                transform=val_crop_aug(m, s), num_list=val_num_list, class_num=args.classnum)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.LR_seg)
    ])

    if args.loss_type == 'dice':
        criterion = [
            weighted_define_loss(args.loss_type, args.classnum),
            train_DiceLoss_weighted(mode='multiclass'),
            train_JaccardLoss_weighted(mode='multiclass')
        ]
    elif args.loss_type == 'ce':
        criterion = [
            weighted_define_loss(args.loss_type, args.classnum),
            train_DiceLoss_logit(mode='multiclass'),
            train_JaccardLoss_logit(mode='multiclass')
        ]
    else :
        criterion = [
            weighted_define_loss(args.loss_type, args.classnum),
            train_DiceLoss_logit(mode='multiclass'),
            train_JaccardLoss_logit(mode='multiclass')
        ]

    max_dice = 0.9
    epoch_start = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

        print('\nEpoch: {}'.format(epoch))

        train_dice, train_jaccard, train_faz_dice, train_faz_jaccard = train(
            model, train_loader, optimizer, criterion, device, writer, args.save_path, training=True)
        val_dice, val_jaccard, val_faz_dice, val_faz_jaccard = train(
            model, val_loader, optimizer, criterion, device, writer, args.save_path, training=False)

        epoch_info = "Epoch: {}".format(epoch)
        train_info = "Training Dice:{:.4f}, Training Jaccard:{:.4f}, Faz Dice:{:.4f}, Faz Jaccard:{:.4f}".format(
            train_dice, train_jaccard, train_faz_dice, train_faz_jaccard)
        val_info = "Validation Dice: {:.4f}, Validation Jaccard: {:.4f}, Faz Dice: {:.4f}, Faz Jaccard: {:.4f}".format(
            val_dice, val_jaccard, val_faz_dice, val_faz_jaccard)
        print(train_info)
        print(val_info)
        logging.info(epoch_info)
        logging.info(train_info)
        logging.info(val_info)
        writer.add_scalar("train_dice", train_dice, epoch)
        writer.add_scalar("train_jaccard", train_jaccard, epoch)
        writer.add_scalar("val_dice", val_dice, epoch)
        writer.add_scalar("val_jaccard", val_jaccard, epoch)
        writer.add_scalar("train_faz_dice", train_faz_dice, epoch)
        writer.add_scalar("train_faz_jaccard", train_faz_jaccard, epoch)
        writer.add_scalar("val_faz_dice", val_faz_dice, epoch)
        writer.add_scalar("val_faz_jaccard", val_faz_jaccard, epoch)

        best_name = os.path.join(args.save_path, "best_faz_dice_" + str(
            round(val_faz_dice, 4)) + "faz_jaccard_" + str(round(val_faz_jaccard, 4)) + ".pt")
        save_name = os.path.join(args.save_path, str(epoch) + "faz_dice_" + str(
            round(val_faz_dice, 4)) + "faz_jaccard_" + str(round(val_faz_jaccard, 4)) + ".pt")

        if max_dice < val_faz_dice:
            max_dice = val_faz_dice
            if max_dice > 0.9:
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

import torch
import os
from tqdm import tqdm
import numpy as np
from utils import create_validation_arg_parser, build_model
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd
import surface_distance
import scipy.spatial
from numpy import mean, std
from utils import *
from losses import *
from sklearn.preprocessing import label_binarize
from dataset import TestDataSet
from augmentation import *


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def getJaccard(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.jaccard(testArray, resultArray)


def getPrecisionAndRecall(testImage, resultImage):

    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    TP = np.sum(testArray*resultArray)
    FP = np.sum((1-testArray)*resultArray)
    FN = np.sum(testArray*(1-resultArray))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    return precision, recall


def getHD_ASSD(seg_preds, seg_labels):
    label_seg = np.array(seg_labels, dtype=bool)
    predict = np.array(seg_preds, dtype=bool)

    surface_distances = surface_distance.compute_surface_distances(
        label_seg, predict, spacing_mm=(1, 1))

    HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) + np.sum(distances_gt_to_pred * surfel_areas_gt))/(np.sum(surfel_areas_gt)+np.sum(surfel_areas_pred))

    return HD, ASSD


def evaluate(model, valLoader, device, save_path):

    model.eval()

    name = []
    faz_dice1 = []
    faz_dice2 = []
    faz_jaccard1 = []
    faz_jaccard2 = []
    rv_dice1 = []
    rv_dice2 = []
    rv_jaccard1 = []
    rv_jaccard2 = []
    dice = []
    jaccard = []
    faz_HD = []
    faz_ASSD = []
    rv_HD = []
    rv_ASSD = []
    dicelosses1 = []
    jaccardlosses1 = []
    dicelosses2 = []
    jaccardlosses2 = []

    for i, (inputs, targets) in enumerate(tqdm(valLoader)):

        inputs = inputs.to(device)
        seg_labels = targets.numpy().squeeze()
        targets = targets.to(device)

        outputs = model(inputs)

        seg_outputs = outputs.detach().cpu().numpy().squeeze()
        c, h, w = seg_outputs.shape
        seg_prs = label_binarize(seg_outputs.argmax(
            0).flatten(), classes=[0, 1, 2])
        seg_prs = seg_prs.reshape(h, w, c)
        seg_prs = seg_prs.transpose(2, 0, 1)
        faz_labels = seg_labels[1, :, :]
        faz_prs = seg_prs[1, :, :]
        rv_labels = seg_labels[2, :, :]
        rv_prs = seg_prs[2, :, :]

        faz_dice_1 = f1_score(faz_labels.flatten(),
                              faz_prs.flatten(), average=None)
        faz_dice_2 = getDSC(faz_labels, faz_prs)
        rv_dice_1 = f1_score(rv_labels.flatten(),
                             rv_prs.flatten(), average=None)
        rv_dice_2 = getDSC(rv_labels, rv_prs)
        faz_jaccard_1 = jaccard_score(
            faz_labels.flatten(), faz_prs.flatten(), average=None)
        faz_jaccard_2 = getJaccard(faz_labels, faz_prs)
        rv_jaccard_1 = jaccard_score(
            rv_labels.flatten(), rv_prs.flatten(), average=None)
        rv_jaccard_2 = getJaccard(rv_labels, rv_prs)

        dices, _dicelosses = eval_DiceLoss(mode='multiclass')(torch.from_numpy(
            np.expand_dims(seg_prs, axis=0)), torch.from_numpy(np.expand_dims(seg_labels, axis=0)))
        jaccards, _jaccardlosses = eval_JaccardLoss(mode='multiclass')(torch.from_numpy(np.expand_dims(
            seg_prs, axis=0)), torch.from_numpy(np.expand_dims(seg_labels, axis=0)))

        dices = 1-dices
        jaccards = 1-jaccards

        faz_HDs, faz_ASSDs = getHD_ASSD(faz_prs, faz_labels)
        rv_HDs, rv_ASSDs = getHD_ASSD(rv_prs, rv_labels)

        name.append(str(i))
        faz_dice1.append(faz_dice_1)
        faz_dice2.append(faz_dice_2)
        faz_jaccard1.append(faz_jaccard_1)
        faz_jaccard2.append(faz_jaccard_2)
        rv_dice1.append(rv_dice_1)
        rv_dice2.append(rv_dice_2)
        rv_jaccard1.append(rv_jaccard_1)
        rv_jaccard2.append(rv_jaccard_2)
        dice.append(dices.to(torch.float64))
        jaccard.append(jaccards.to(torch.float64))
        faz_HD.append(faz_HDs)
        faz_ASSD.append(faz_ASSDs)
        rv_HD.append(rv_HDs)
        rv_ASSD.append(rv_ASSDs)
        dicelosses1.append(1-_dicelosses[1])
        jaccardlosses1.append(1-_jaccardlosses[1])
        dicelosses2.append(1-_dicelosses[2])
        jaccardlosses2.append(1-_jaccardlosses[2])

    return name, faz_dice1, faz_dice2, faz_jaccard1, faz_jaccard2, rv_dice1, rv_dice2, rv_jaccard1, rv_jaccard2, dice, jaccard, faz_HD, faz_ASSD, rv_HD, rv_ASSD, dicelosses1, jaccardlosses1, dicelosses2, jaccardlosses2


def main():
    args = create_validation_arg_parser().parse_args()
    model_file = args.model_file
    save_path = args.save_path

    names = []
    faz_dice1 = []
    faz_dice2 = []
    faz_jaccard1 = []
    faz_jaccard2 = []
    rv_dice1 = []
    rv_dice2 = []
    rv_jaccard1 = []
    rv_jaccard2 = []
    dice = []
    jaccard = []
    faz_HD = []
    faz_ASSD = []
    rv_HD = []
    rv_ASSD = []
    dicelosses1 = []
    jaccardlosses1 = []
    dicelosses2 = []
    jaccardlosses2 = []

    pretrain = args.pretrain
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.batch_size = 1
    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    if args.set == '6M':
        # 6M
        train_num_list = list(range(10001, 10181))
        test_num_list = list(range(10201, 10301))
    elif args.set == '3M':
        # 3M
        train_num_list = list(range(10301, 10441))
        test_num_list = list(range(10451, 10501))

    m, s = mean_and_std(args.data_path, train_num_list)

    test_data = TestDataSet(data_path=args.data_path, mask_path=args.mask_path,
                             transform=val_crop_aug(m, s), num_list=test_num_list, class_num=args.classnum)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model_type, args.encoder,
                        pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    model.load_state_dict(torch.load(model_file))

    name, _faz_dice1, _faz_dice2, _faz_jaccard1, _faz_jaccard2, _rv_dice1, _rv_dice2, _rv_jaccard1, _rv_jaccard2, _dice, _jaccard, _faz_HD, _faz_ASSD, _rv_HD, _rv_ASSD, _dicelosses1, _jaccardlosses1, _dicelosses2, _jaccardlosses2 = evaluate(
        model, test_loader, device, save_path)

    names.extend(name)
    faz_dice1.extend(_faz_dice1)
    faz_dice2.extend(_faz_dice2)
    faz_jaccard1.extend(_faz_jaccard1)
    faz_jaccard2.extend(_faz_jaccard2)
    rv_dice1.extend(_rv_dice1)
    rv_dice2.extend(_rv_dice2)
    rv_jaccard1.extend(_rv_jaccard1)
    rv_jaccard2.extend(_rv_jaccard2)
    dice.extend(_dice)
    jaccard.extend(_jaccard)
    faz_HD.extend(_faz_HD)
    faz_ASSD.extend(_faz_ASSD)
    rv_HD.extend(_rv_HD)
    rv_ASSD.extend(_rv_ASSD)
    dicelosses1.extend(_dicelosses1)
    jaccardlosses1.extend(_jaccardlosses1)
    dicelosses2.extend(_dicelosses2)
    jaccardlosses2.extend(_jaccardlosses2)

    dataframe = pd.DataFrame({'case': names,
                              'faz_dice1': faz_dice1, 'faz_dice2': faz_dice2,
                             'faz_jaccard1': faz_jaccard1, 'faz_jaccard2': faz_jaccard2,
                              'rv_dice1': rv_dice1, 'rv_dice2': rv_dice2,
                              'rv_jaccard1': rv_jaccard1, 'rv_jaccard2': rv_jaccard2,
                              'dice': dice, 'jaccard': jaccard,
                              'faz_HD': faz_HD, 'faz_ASSD': faz_ASSD,
                              'rv_HD': rv_HD, 'rv_ASSD': rv_ASSD,
                              'dicelosses1': dicelosses1, 'jaccardlosses1': jaccardlosses1,
                              'dicelosses2': dicelosses2, 'jaccardlosses2': jaccardlosses2,
                              })
    dataframe.to_csv(save_path + "/detail_metrics.csv",
                     index=False, sep=',')
    print('Counting CSV generated!')
    mean_resultframe = pd.DataFrame({
        'faz_dice': mean(faz_dice2), 'faz_jaccard': mean(faz_jaccard2),
                                     'rv_dice': mean(rv_dice2), 'rv_jaccard': mean(rv_jaccard2),
                                     'dice': mean(dice), 'jaccard': mean(jaccard),
                                     'faz_HD': mean(faz_HD), 'faz_ASSD': mean(faz_ASSD),
                                     'rv_HD': mean(rv_HD), 'rv_ASSD': mean(rv_ASSD)}, index=[1])
    mean_resultframe.to_csv(save_path + "/mean_metrics.csv", index=0)
    std_resultframe = pd.DataFrame({
        'faz_dice': std(faz_dice2, ddof=1), 'faz_jaccard': std(faz_jaccard2, ddof=1),
        'rv_dice': std(rv_dice2, ddof=1), 'rv_jaccard': std(rv_jaccard2, ddof=1),
        'dice': std(dice, ddof=1), 'jaccard': std(jaccard, ddof=1),
        'faz_HD': std(faz_HD, ddof=1), 'faz_ASSD': std(faz_ASSD, ddof=1),
        'rv_HD': std(rv_HD, ddof=1), 'rv_ASSD': std(rv_ASSD, ddof=1)}, index=[1])
    std_resultframe.to_csv(save_path + "/std_metrics.csv", index=0)
    print('Calculating CSV generated!')


if __name__ == "__main__":
    main()

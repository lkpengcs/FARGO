import torch
import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser, build_model
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd
from scipy.special import softmax
import surface_distance
import scipy.spatial
from numpy import mean, std
from utils import *
from losses import *
from sklearn.preprocessing import label_binarize
from dataset import TestDataSet
from augmentation import *
from iternetmodel import Iternet


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
    rv_dice1 = []
    rv_dice2 = []
    rv_jaccard1 = []
    rv_jaccard2 = []
    rv_dice3 = []
    rv_dice4 = []
    rv_jaccard3 = []
    rv_jaccard4 = []
    rv_HD1 = []
    rv_ASSD1 = []
    rv_HD2 = []
    rv_ASSD2 = []
    rv_HD3 = []
    rv_ASSD3 = []
    rv_HD4 = []
    rv_ASSD4 = []

    for i, (inputs, targets) in enumerate(tqdm(valLoader)):
        
        inputs = inputs.to(device)
        seg_labels = targets.numpy().squeeze()
        targets = targets.to(device)
        rv_labels = seg_labels
        logits = model(inputs)
        logits1 = logits[0]
        logits2 = logits[1]
        logits3 = logits[2]
        logits4 = logits[3]
        outputs1 = torch.sigmoid(logits1)
        outputs2 = torch.sigmoid(logits2)
        outputs3 = torch.sigmoid(logits3)
        outputs4 = torch.sigmoid(logits4)
        rv_prs1 = torch.round(outputs1).cpu().detach().numpy().squeeze()
        rv_prs2 = torch.round(outputs2).cpu().detach().numpy().squeeze()
        rv_prs3 = torch.round(outputs3).cpu().detach().numpy().squeeze()
        rv_prs4 = torch.round(outputs4).cpu().detach().numpy().squeeze()

        rv_dice_1 = getDSC(rv_labels, rv_prs1)
        rv_dice_2 = getDSC(rv_labels, rv_prs2)
        rv_dice_3 = getDSC(rv_labels, rv_prs3)
        rv_dice_4 = getDSC(rv_labels, rv_prs4)

        rv_jaccard_1 = getJaccard(rv_labels, rv_prs1)
        rv_jaccard_2 = getJaccard(rv_labels, rv_prs2)
        rv_jaccard_3 = getJaccard(rv_labels, rv_prs3)
        rv_jaccard_4 = getJaccard(rv_labels, rv_prs4)

        rv_HDs1, rv_ASSDs1 = getHD_ASSD(rv_prs1, rv_labels)
        rv_HDs2, rv_ASSDs2 = getHD_ASSD(rv_prs2, rv_labels)
        rv_HDs3, rv_ASSDs3 = getHD_ASSD(rv_prs3, rv_labels)
        rv_HDs4, rv_ASSDs4 = getHD_ASSD(rv_prs4, rv_labels)
        
        outputs1 = outputs1.cpu().detach().numpy().squeeze()
        outputs2 = outputs2.cpu().detach().numpy().squeeze()
        outputs3 = outputs3.cpu().detach().numpy().squeeze()
        outputs4 = outputs4.cpu().detach().numpy().squeeze()

        name.append(str(i))
        rv_dice1.append(rv_dice_1)
        rv_dice2.append(rv_dice_2)
        rv_jaccard1.append(rv_jaccard_1)
        rv_jaccard2.append(rv_jaccard_2)
        rv_dice3.append(rv_dice_3)
        rv_dice4.append(rv_dice_4)
        rv_jaccard3.append(rv_jaccard_3)
        rv_jaccard4.append(rv_jaccard_4)
        rv_HD1.append(rv_HDs1)
        rv_ASSD1.append(rv_ASSDs1)
        rv_HD2.append(rv_HDs2)
        rv_ASSD2.append(rv_ASSDs2)
        rv_HD3.append(rv_HDs3)
        rv_ASSD3.append(rv_ASSDs3)
        rv_HD4.append(rv_HDs4)
        rv_ASSD4.append(rv_ASSDs4)

    return name, rv_dice1, rv_dice2, rv_dice3, rv_dice4, rv_jaccard1, rv_jaccard2, rv_jaccard3, rv_jaccard4, rv_HD1, rv_ASSD1, rv_HD2, rv_ASSD2, rv_HD3, rv_ASSD3, rv_HD4, rv_ASSD4, 


def main():
    args = create_validation_arg_parser().parse_args()
    model_file = args.model_file
    save_path = args.save_path

    names = []
    rv_dice1 = []
    rv_dice2 = []
    rv_jaccard1 = []
    rv_jaccard2 = []
    rv_dice3 = []
    rv_dice4 = []
    rv_jaccard3 = []
    rv_jaccard4 = []
    rv_HD1 = []
    rv_ASSD1 = []
    rv_HD2 = []
    rv_ASSD2 = []
    rv_HD3 = []
    rv_ASSD3 = []
    rv_HD4 = []
    rv_ASSD4 = []

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

    model = Iternet(1, 1).to(device)
    model.load_state_dict(torch.load(model_file))
    name, _rv_dice1, _rv_dice2, _rv_dice3, _rv_dice4, _rv_jaccard1, _rv_jaccard2, _rv_jaccard3, _rv_jaccard4, _rv_HD1, _rv_ASSD1, _rv_HD2, _rv_ASSD2,_rv_HD3, _rv_ASSD3,_rv_HD4, _rv_ASSD4, = evaluate(
        model, test_loader, device, save_path)

    names.extend(name)
    rv_dice1.extend(_rv_dice1)
    rv_dice2.extend(_rv_dice2)
    rv_jaccard1.extend(_rv_jaccard1)
    rv_jaccard2.extend(_rv_jaccard2)
    rv_dice3.extend(_rv_dice3)
    rv_dice4.extend(_rv_dice4)
    rv_jaccard3.extend(_rv_jaccard3)
    rv_jaccard4.extend(_rv_jaccard4)
    rv_HD1.extend(_rv_HD1)
    rv_ASSD1.extend(_rv_ASSD1)
    rv_HD2.extend(_rv_HD2)
    rv_ASSD2.extend(_rv_ASSD2)
    rv_HD3.extend(_rv_HD3)
    rv_ASSD3.extend(_rv_ASSD3)
    rv_HD4.extend(_rv_HD4)
    rv_ASSD4.extend(_rv_ASSD4)

    dataframe = pd.DataFrame({'case': names,
                              'rv_dice1': rv_dice1, 'rv_dice2': rv_dice2, 'rv_dice3': rv_dice3, 'rv_dice4': rv_dice4,
                              'rv_jaccard1': rv_jaccard1, 'rv_jaccard2': rv_jaccard2, 'rv_jaccard3': rv_jaccard3, 'rv_jaccard4': rv_jaccard4,
                              'rv_HD1': rv_HD1, 'rv_ASSD1': rv_ASSD1,'rv_HD2': rv_HD2, 'rv_ASSD2': rv_ASSD2,'rv_HD3': rv_HD3, 'rv_ASSD3': rv_ASSD3,'rv_HD4': rv_HD4, 'rv_ASSD4': rv_ASSD4,
                              })
    dataframe.to_csv(save_path + "/detail_metrics.csv",
                     index=False, sep=',')
    print('Counting CSV generated!')
    mean_resultframe = pd.DataFrame({
        'rv_dice1': mean(rv_dice1), 'rv_jaccard1': mean(rv_jaccard1),
        'rv_dice2': mean(rv_dice2), 'rv_jaccard2': mean(rv_jaccard2),
        'rv_dice3': mean(rv_dice3), 'rv_jaccard3': mean(rv_jaccard3),
        'rv_dice4': mean(rv_dice4), 'rv_jaccard4': mean(rv_jaccard4),
        'rv_HD1': mean(rv_HD1), 'rv_ASSD1': mean(rv_ASSD1),
        'rv_HD2': mean(rv_HD2), 'rv_ASSD2': mean(rv_ASSD2),
        'rv_HD3': mean(rv_HD3), 'rv_ASSD3': mean(rv_ASSD3),
        'rv_HD4': mean(rv_HD4), 'rv_ASSD4': mean(rv_ASSD4)
        }, index=[1])
    mean_resultframe.to_csv(save_path + "/mean_metrics.csv", index=0)
    std_resultframe = pd.DataFrame({
        'rv_dice1': std(rv_dice1, ddof=1), 'rv_jaccard1': std(rv_jaccard1, ddof=1),
        'rv_dice2': std(rv_dice2, ddof=1), 'rv_jaccard2': std(rv_jaccard2, ddof=1),
        'rv_dice3': std(rv_dice3, ddof=1), 'rv_jaccard3': std(rv_jaccard3, ddof=1),
        'rv_dice4': std(rv_dice4, ddof=1), 'rv_jaccard4': std(rv_jaccard4, ddof=1),
        'rv_HD1': std(rv_HD1, ddof=1), 'rv_ASSD1': std(rv_ASSD1, ddof=1),
        'rv_HD2': std(rv_HD2, ddof=1), 'rv_ASSD2': std(rv_ASSD2, ddof=1),
        'rv_HD3': std(rv_HD3, ddof=1), 'rv_ASSD3': std(rv_ASSD3, ddof=1),
        'rv_HD4': std(rv_HD4, ddof=1), 'rv_ASSD4': std(rv_ASSD4, ddof=1)
        }, index=[1])
    std_resultframe.to_csv(save_path + "/std_metrics.csv", index=0)
    print('Calculating CSV generated!')


if __name__ == "__main__":
    main()

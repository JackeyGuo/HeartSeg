import os
import sys
import json
import re
import argparse
import tensorlayer as tl
import numpy as np
import sklearn.metrics as metrics
import SimpleITK as sitk

model_name = 'xception_aspp'

output_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/score" % (model_name)
result_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/score.txt" % (model_name)

def compute_dice(output, target, loss_type='jaccard'):

    inse = np.sum(output * target)
    if loss_type == 'jaccard':
        l = np.sum(output * output)
        r = np.sum(target * target)
    elif loss_type == 'sorensen':
        l = np.sum(output)
        r = np.sum(target)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse) / (l + r)
    dice = np.mean(dice)

    return dice
def compute_indice(truth_path,predict_path):
    predict_list = os.listdir(predict_path)
    for file in predict_list:
        file_name = file.split('.')[0]
        truth = os.path.join(truth_path, str(file_name) + seg_tail)
        predict = os.path.join(predict_path, str(file_name) + '.mha')
        json = os.path.join(output_path, str(file_name) + ".json")

        truth = sitk.GetArrayFromImage(sitk.ReadImage(truth))
        predict = sitk.GetArrayFromImage(sitk.ReadImage(predict))
        #dice = tl.cost.dice_coe(truth,predict,axis=[1,2])
        dice2 = tl.cost.dice_hard_coe(truth,predict)
        dice = compute_dice(predict,truth)
        #f1_score = metrics.f1_score(truth,predict)
        print(file_name,dice)
        #t_p_o_list.append((t, p, o))


    return

if __name__=='__main__':
    truth_path = '/home/data_new/dcm/training_T1'
    seg_tail = "_seg.mha"

    predict_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/predict" % (model_name)
    compute_indice(truth_path,predict_path)
import SimpleITK as sitk
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import config.config as cfg

'''dirname = '/home/data_new/dcm/training_T1'
for maindir, subdir, file_name_list in os.walk(dirname):
    for filename in file_name_list:
        dicom_file_path = os.path.join(maindir, filename)
        # get patient name
        #print(dicom_file_path)
        patient_img = sitk.ReadImage(dicom_file_path)
        print(patient_img.GetDirection())'''

# import json
#
# def resolveJson(path):
#     file = open(path, "rb")
#     fileJson = json.load(file)[0]
#     print(fileJson)
#     field = fileJson["DICE"]
#     futures = fileJson["PRECISION"]
#     type = fileJson["RECALL"]
#
#     return (field, futures, type)
#
# def output():
#     result = resolveJson(path)
#     print(result)
#     for x in result:
#         for y in x:
#             print(y)
#
# model_name = 'xception_aspp'
#
# path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/score/20150204_wang_T1_Series0014.json" % (model_name)
# #path = r"C:\Users\dell\Desktop\kt\test.json"
# output()
data_npy_path = os.path.join(cfg.output_path, "data")
data = np.load(data_npy_path + '/data.npy')
#print(data[133:-1])

model_name = 'denseDilatedASPP'# denseDilatedASPP,xception_aspp,deform_xception_aspp,deeplab
image_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/predict" % (model_name)

pat_list = []
patient = os.listdir(image_path)
for da in data:
    if da+'.mha' in patient:
        #print(da,patient)
        pat_list.append(da)
for da in data:
    if da+'.mha' not in patient:
        #print(da,patient)
        pat_list.append(da)
np.save('./data.npy',pat_list)
print(len(pat_list))
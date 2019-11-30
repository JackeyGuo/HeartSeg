import numpy as np
import os
import config.config as cfg

path = '/home/data_new/dcm/normal_control_T1_prep' #patient_T1_prep

# with open('./normal_control_T1_prep.txt','w') as file:
#     for pat_name in os.listdir(path):
#         if pat_name.split('.')[0].split('_')[-1] != 'seg':
#             file.write(pat_name+'\n')
# pat_txt = open('./txts/patient_T1_prep.txt', 'r')
# pat_data = pat_txt.read().split('\n')
# nor_txt = open('./txts/normal_control_T1_prep.txt', 'r')
# nor_data = nor_txt.read().split('\n')
#
# print(nor_data)
# data = np.load('./npys/%s_pred.npy'%cfg.name)
# print(np.shape(data))
# data_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/xception_aspp/data"
# npy_data = np.load(os.path.join(data_path,'data.npy'))
# print(npy_data)
#
# cls_truth = []
# pat_txt = open('./txts/patient_T1_prep.txt', 'r')
# pat_data = pat_txt.read().split('\n')
# nor_txt = open('./txts/normal_control_T1_prep.txt', 'r')
# nor_data = nor_txt.read().split('\n')
#
# for pat in npy_data:
#     #print(pat)
#     if pat + '.mha' in pat_data:
#         cls_truth.append(1)
#     if pat + '.mha' in nor_data:
#         cls_truth.append(0)
# print(cls_truth)
size = '91'
patient_path = '/home/data_new/dcm/DCM_2rd_preprocess/patient_%sx%sx1/'%(size,size)
nor_path = '/home/data_new/dcm/DCM_2rd_preprocess/normal_%sx%sx1/'%(size,size)

pat_txt = open('./txts/patient_%s.txt'%size, 'w')
nor_txt = open('./txts/normal_%s.txt'%size, 'w')
for name_list in os.listdir(patient_path):
    #print(name_list.split('.')[0].split('_'))
    if name_list.split('.')[0].split('_')[-1] != 'seg':
        pat_txt.write(name_list+'\n')
for name_list in os.listdir(nor_path):
    if name_list.split('.')[0].split('_')[-1] != 'seg':
        nor_txt.write(name_list+'\n')
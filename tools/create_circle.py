import SimpleITK as sitk
import numpy as np
import os
import shutil

model_name = 'deeplab'# denseDilatedASPP,xception_aspp,deform_xception_aspp,deeplab
#truth_path = r"/home/data_new/dcm/training_T1"
image_size = 91
#data_path = '/home/data_new/guofeng/data/heart/training_images_%d/'%image_size
#data_path = '/home/data_new/guofeng/data/heart/training_T1_circle_%d/'%image_size
data_path = '/home/data_new/guofeng/data/heart/modify/'
output_path = '/home/data_new/guofeng/data/heart/test/'
score_dir = r"/home/gf/Downloads/EvaluateSegmentation"
seg_tail = r"_seg.mha"

#output_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/predict_circle" % (model_name)
#seg_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/predict" % (model_name)
def create_circle(data_path):
    data_list = os.listdir(data_path)
    for patient in data_list:
        patient_path = os.path.join(data_path, patient)
        if patient.split('.')[0].split('_')[-1] == 'seg':
            sitk_array = sitk.ReadImage(patient_path)
            pat_array = sitk.GetArrayFromImage(sitk_array)
            zeor_array = np.zeros(np.shape(pat_array), dtype=pat_array.dtype)

            new_array = np.where(pat_array == 1, pat_array, zeor_array)
            # mean_value = np.sum(new_array) / total_one

            # if need to write the predict circle
            sitk_img = sitk.GetImageFromArray(new_array, isVector=False)
            sitk_img.SetOrigin(sitk_array.GetOrigin())
            sitk_img.SetSpacing(sitk_array.GetSpacing())
            sitk.WriteImage(sitk_img, os.path.join(output_path, patient))
def circle(seg_path,out_path,write_circle=False):

    seg_list = os.listdir(seg_path)
    for seg in seg_list:
        if model_name == 'unet':
            patient_seg = os.path.join(seg_path, seg)
            #print(patient_seg)
        else:
            patient_seg = os.path.join(seg_path, seg)
            # if seg.split('.')[0].split('_')[-1] == 'seg':
            #     patient_seg = os.path.join(seg_path,seg)
            # else:
            #     shutil.copyfile(os.path.join(seg_path,seg),os.path.join(output_path,seg))
            #     continue
        #if seg.split('.')[0].split('_')[-1] != 'seg':

        sitk_array = sitk.ReadImage(patient_seg)
        seg_array = sitk.GetArrayFromImage(sitk_array)
        zeor_array = np.zeros(np.shape(seg_array),dtype=seg_array.dtype)

        new_array = np.where(seg_array == 1,seg_array,zeor_array)
        #mean_value = np.sum(new_array) / total_one

        # if need to write the predict circle
        if write_circle:
            sitk_img = sitk.GetImageFromArray(new_array, isVector=False)
            sitk_img.SetOrigin(sitk_array.GetOrigin())
            sitk_img.SetSpacing(sitk_array.GetSpacing())
            sitk.WriteImage(sitk_img, os.path.join(out_path,seg))

if __name__=='__main__':
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #circle(data_path,output_path,write_circle=True)
    create_circle(data_path)
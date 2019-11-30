import SimpleITK as sitk
import numpy as np
import os
import cv2
import config.config as cfg
from skimage import io
import scipy.misc
from sklearn import preprocessing

model_name = 'deeplab'# denseDilatedASPP,xception_aspp,deform_xception_aspp,deeplab merge_model_sig_cmsf
image_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result_0410/%s/images" % (model_name)
#image_path = '/home/data_new/guofeng/data/heart/training_T1_images_seg'
predict_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result_0410/%s/predict" % (model_name)
#predit_path = '/home/data_new/guofeng/data/heart/training_T1_images'
data_path = cfg.data_path
raw_seg_path = '/home/data_new/guofeng/data/heart/training_T1_images_with_seg_488'#


def normalize(image, ranges=(0, 255)):
    """
    do image normalize, image to [min, max], default is [-1., 1.]
    :param image: ndarray
    :param ranges: tuple, (min, max)
    :return:
    """
    _min = ranges[0]
    _max = ranges[1]
    #print(image.min(),image.max())
    return (_max - _min) * (image - image.min()) / (image.max() - image.min()) + _min

def save_png(pred_path, truth_path):

    pred_path_list = os.listdir(pred_path)

    for pred_list in pred_path_list:
        if model_name == 'unet':
            patient_name = '_'.join(pred_list.split('.')[0].split('_')[:-2])
            pred_list = patient_name + '.mha'
        if model_name == 'raw':
            patient_name = '_'.join(pred_list.split('.')[0].split('_')[:-1])
            truth_list = patient_name + '.mha'
        else:
            patient_name = pred_list.split('.')[0]
        print(patient_name,pred_list)

        # predict circle array
        pred_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_path, pred_list)))
        #pred_array = np.squeeze(pred_array)
        # truth mha array
        sitk_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(truth_path, pred_list)))
        #sitk_array = np.squeeze(sitk_array)
        #print(np.shape(pred_array),np.shape(sitk_array))
        # saved image
        red_array = np.zeros((128, 128, 3), dtype=sitk_array.dtype)
        red_array[:, :, 0] = np.where(pred_array == 1, 0, sitk_array)
        red_array[:, :, 1] = np.where(pred_array == 1, 0, sitk_array)
        red_array[:, :, 2] = np.where(pred_array == 1, 255, sitk_array)
        #print(patient_name)
        cv2.imwrite(image_path+'/%s.png'%patient_name, red_array)
def save_raw_png(truth_path):

    truth_path_list = os.listdir(truth_path)

    for truth_list in truth_path_list:

        patient_name = truth_list.split('.')[0]

        if patient_name.split('_')[-1] != 'seg':
            #print(patient_name)
            # truth mha array
            sitk_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(truth_path, truth_list)))
            sitk_array = np.squeeze(sitk_array)
            # saved image
            cv2.imwrite(image_path+'/%s.png'%patient_name, sitk_array)

def save_seg_png(truth_path):
    if not os.path.exists(raw_seg_path):
        os.makedirs(raw_seg_path)

    path_list = os.listdir(truth_path)[:10]

    for lst in path_list:

        patient_name = lst.split('.')[0]

        #print(patient_name)
        if patient_name.split('_')[-1] != 'seg':
            #print(patient_name,lst,os.path.join(truth_path, patient_name+'_seg.mha'))
            pred_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(truth_path, patient_name+'_seg.mha')))
            pred_array = np.squeeze(pred_array)
            # truth mha array
            sitk_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(truth_path, lst)))
            sitk_array = np.squeeze(sitk_array)
            #sitk_array = normalize(sitk_array)
            sitk_array = preprocessing.MinMaxScaler((0,255)).fit_transform(sitk_array)
            sitk_array = sitk_array.astype(np.float32)

            print(sitk_array.dtype)
            if cfg.use_488_data == False:
                # saved image
                red_array = np.zeros((128, 128, 3), dtype=np.int32)
                red_array[:, :, 0] = np.where(pred_array == 1, 255, sitk_array)
                red_array[:, :, 1] = np.where(pred_array == 1, 0, sitk_array)
                red_array[:, :, 2] = np.where(pred_array == 1, 0, sitk_array)
            else:
                #sitk_array = np.transpose(sitk_array, (1, 2, 0))
                #pred_array = np.transpose(pred_array, (1, 2, 0))

                #red_array = np.concatenate([sitk_array,sitk_array,sitk_array],axis=-1)
                #print(np.shape(sitk_array),np.shape(red_array),np.shape(pred_array))

                # sitk_array = np.squeeze(sitk_array)
                # pred_array = np.squeeze(pred_array)

                red_array = np.zeros((128, 128, 3), dtype=sitk_array.dtype)
                red_array[:, :, 0] = np.where(pred_array == 1, 0, sitk_array)
                red_array[:, :, 1] = np.where(pred_array == 1, 0, sitk_array)
                red_array[:, :, 2] = np.where(pred_array == 1, 255, sitk_array)
            from skimage import io, data, color
            #sitk_array = color.gray2rgb(sitk_array)
            #cv2.imwrite(raw_seg_path+'/%s.png'%patient_name, red_array)
            #scipy.misc.imsave(raw_seg_path + '/%s.jpeg' % patient_name, red_array)
            #scipy.misc.toimage(red_array, cmin=0, cmax=255).save(raw_seg_path + '/%s.png' % patient_name)
            #io.imsave(raw_seg_path + '/%s.png' % patient_name, red_array)
            # saved image
            #sitk_array = sitk_array.astype(np.float32)
            #sitk_array = np.squeeze(sitk_array)
            from PIL import Image
            im = Image.fromarray(sitk_array)
            print(im.mode)
            im = im.convert('L')
            im.save(raw_seg_path+'/%s.jpeg'%patient_name)


if __name__=='__main__':
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    #save_png(predict_path,data_path)
    save_seg_png(data_path)
    #save_raw_png(data_path)
    #save_png('/home/data_new/guofeng/data/heart/training_T1_bound',data_path)
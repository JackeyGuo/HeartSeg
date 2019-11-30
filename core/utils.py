import os
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import random
import cv2
from config import config as cfg

class DataProvider(object):
    def __init__(self):
        #self.cfg = config_holder.get_instance()
        self.train_list = []
        self.val_list = []
        self.test_list = []
        self.test_iter = 0
        self.data_path = cfg.data_path
        self.patient_tail = cfg.patient_file_tail
        self.seg_tail = cfg.patient_seg_tail
        self.output_path = cfg.output_path
        #self.output_path = ''
        self.predict_tail = cfg.predict_tail
        self.patient_file_tail = cfg.patient_file_tail
        self.weight_distance_path = cfg.weight_distance_path
        self.patient_weight_tail = cfg.patient_weight_tail
        self.random_mirror = cfg.random_mirror
        self.image_size = cfg.image_size
        self.use_488_data = cfg.use_488_data

    #@staticmethod
    def getPatientName(self):

        files = os.listdir(self.data_path)
        patient_list = []
        for f in files:
            file_tail = f.split('_')[-1].split('.')[0]
            if file_tail != 'seg':
                patient_list.append(f.split('.')[0])

        return patient_list

    def setTrainValTest(self,train_list,val_list,test_list):
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        #print(len(train_list),len(val_list),len(test_list))

    def setTrainVal(self, train_list, val_list):
        self.train_list = train_list
        self.test_list = val_list
    def load_classes(self):

        pat_txt = open('./txts/patient_T1_prep.txt','r')
        pat_data = pat_txt.read().split('\n')
        nor_txt = open('./txts/normal_control_T1_prep.txt', 'r')
        nor_data = nor_txt.read().split('\n')

        return pat_data,nor_data
    def load_label(self,size):

        pat_txt = open('./txts/patient_%s.txt'%size,'r')
        pat_data = pat_txt.read().strip().split('\n')
        nor_txt = open('./txts/normal_%s.txt'%size, 'r')
        nor_data = nor_txt.read().strip().split('\n')

        return pat_data,nor_data
    def get_train_holder(self,batch_size=1):
        """
        choice one patient and create a placeholder with it`s shape.
        :return:
        """

        if len(self.train_list) == 0:
            return None
        patient = self.train_list[0]

        full_path = os.path.join(self.data_path,patient+self.patient_tail)

        npa = sitk.GetArrayFromImage(sitk.ReadImage(full_path))
        npa = np.squeeze(npa)
        #npa = npa[1:103, :, :]
        #print(np.shape(npa))
        # z, y, x
        shape = npa.shape
        img_shape = [batch_size,]
        img_shape.extend(shape)
        #img_shape = np.squeeze(img_shape,axis=-1)
        img_shape.append(1)#add one for the channel.

        seg_shape = [batch_size,]
        seg_shape.extend(shape)
        seg_shape = np.squeeze(seg_shape)
        #seg_shape.append(1)

        if cfg.joint_train:
            class_shape = [batch_size,]
            return tf.placeholder(tf.float32, img_shape), tf.placeholder(tf.int32, seg_shape), tf.placeholder(tf.int32, class_shape)
        else:
            if cfg.use_dst_loss == 1:
                dst_shape = seg_shape
                return tf.placeholder(tf.float32, img_shape), tf.placeholder(tf.int32, seg_shape), tf.placeholder(
                    tf.float32, dst_shape)
            if cfg.use_dst_loss == 0:
                return tf.placeholder(tf.float32, img_shape), tf.placeholder(tf.int32, seg_shape)

    def normalize(self, image, ranges=(0, 255)):
        """
        do image normalize, image to [min, max], default is [-1., 1.]
        :param image: ndarray
        :param ranges: tuple, (min, max)
        :return:
        """
        _min = ranges[0]
        _max = ranges[1]
        return (_max - _min) * (image - image.min()) / (image.max() - image.min()) + _min
    def _get_patient_value_random(self,batch_size=1,target_set="train",with_weight=False):
        """
        old version.See the _get_patient_value
        :param batch_size:
        :param target_set:
        :param with_weight:
        :return:
        """
        t_set = []
        if target_set == "train":
            t_set = self.train_list
        elif target_set == "val":
            t_set = self.test_list
        imgs = []
        labels = []
        cls = []
        weights = []
        wp = self.weight_distance_path
        w_tail = self.patient_weight_tail
        for i in range(batch_size):
            rnd = random.randint(0, len(t_set) - 1)
            pat = t_set[rnd]
            #print(i,len(t_set),pat)
            train_path = os.path.join(self.data_path, pat + self.patient_tail)
            seg_path = os.path.join(self.data_path, pat + self.seg_tail)
            pat_arr = sitk.GetArrayFromImage(sitk.ReadImage(train_path))
            # do normalize
            #pat_arr = self.normalize(pat_arr)

            #pat_arr = pat_arr[:,:,:,np.newaxis]
            # transpose [1,128,128] to [128,128,1]
            pat_arr = np.transpose(pat_arr, (1, 2, 0))

            seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
            seg_arr = np.transpose(seg_arr, (1, 2, 0))
            if with_weight:
               weight_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(wp,pat + w_tail)))
               weights.append(weight_arr)
            if self.random_mirror == 1:
                if random.randint(0,99) < 50:
                    # do this will squeeze [128,128,1] to [128,128]
                    pat_arr = cv2.flip(pat_arr, 0)
                    seg_arr = cv2.flip(seg_arr, 0)
                    pat_arr = pat_arr[:, :, np.newaxis]
                    seg_arr = seg_arr[:, :, np.newaxis]
                if random.randint(0,99) >= 50:
                    pat_arr = cv2.flip(pat_arr, 1)
                    seg_arr = cv2.flip(seg_arr, 1)
                    pat_arr = pat_arr[:, :, np.newaxis]
                    seg_arr = seg_arr[:, :, np.newaxis]
            imgs.append(pat_arr)
            labels.append(seg_arr)

            if self.use_488_data == True:
                pat_data, nor_data = self.load_label(self.image_size)
            else:
                pat_data, nor_data = self.load_classes()

            if pat + self.patient_tail in pat_data:
                cls.append(1)
            if pat + self.patient_tail in nor_data:
                cls.append(0)
        #print(np.shape(imgs))
        images = np.stack(imgs)
        labels = np.stack(labels)
        clss = np.stack(cls)
        #print(np.shape(imgs),np.shape(clss),np.shape(labels))
        if with_weight:
            weights = np.stack(weights)
        if with_weight:
            if self.random_mirror == 1:
                if random.randint(0,99) < 50:
                    images = images[:,:,::-1,:,:]
                    labels = labels[:,:,::-1,:]
                    weights = weights[:,:,::-1,:]
            images = np.squeeze(images, axis=1)
            labels = np.squeeze(labels, axis=1)
            weights = np.squeeze(weights, axis=1)
            return images,labels,weights

        # if self.random_mirror == 1:
        #     if random.randint(0,99) < 50:
        #         images = images[:,:,::-1,:,:]
        #         labels = labels[:,:,::-1,:]
        #images = np.squeeze(images,axis=1)
        # squeeze [batchsize, 128, 128, 1] to [batchsize, 128, 128]
        labels = np.squeeze(labels,axis=-1)
        #print(np.shape(images),np.shape(labels))
        if cfg.joint_train:
            return images, labels, clss
        else:
            return images, labels

    def get_train_value(self,batch_size=1,with_weight=False):
        return self._get_patient_value_random(batch_size,with_weight=with_weight)

    def get_val_holder(self):
        """
        maybe not use.
        :return:
        """
        pass

    def get_val_value(self,batch_size=1,with_weight=False):
        return self._get_patient_value_random(batch_size=batch_size,target_set="val",with_weight=with_weight)

    def init_test(self):
        self.test_iter = 0

    def get_test_holder(self,batch_size = 1):
        if len(self.test_list) == 0:
            return None
        npa = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path,self.test_list[0] + self.patient_tail)))
        npa = np.squeeze(npa)
        shape = [batch_size, ]
        shape.extend(npa.shape)
        shape.append(1)
        #print('shape',shape)
        if cfg.joint_train:
            class_shape = [batch_size,]
            return tf.placeholder(tf.float32,shape),tf.placeholder(tf.int32,class_shape)
        else:
            return tf.placeholder(tf.float32,shape)

    def get_test_value(self,batch_size=1):
        """
        use a list as return for the patch_base.
        one time one patient.
        :return:
        """
        ret_list = []
        if self.test_iter == len(self.test_list):
            return None

        for i in range(batch_size):
            npa = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path,self.test_list[self.test_iter] + self.patient_tail)))
            # do normalize
            #npa = self.normalize(npa)
            npa = np.transpose(npa, (1, 2, 0))
            ret_list.append(npa[np.newaxis,:,:,:])
        ret_list = np.stack(ret_list)
        #print('ret_list', np.shape(ret_list))
        return ret_list

    def write_test(self,value):
        output_dir = os.path.join(self.output_path, "predict")
        #pro_dir = os.path.join(self.output_path, "probability")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        '''if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)'''
        patient_name = self.test_list[self.test_iter]
        print('patient_name: ',patient_name)
        patient_img = sitk.ReadImage(os.path.join(self.data_path, patient_name + self.patient_tail))
        # patient_img.
        predicted_label = value[0]
        #print('predicted_label', np.shape(predicted_label))
        #predicted_label = np.squeeze(predicted_label)
        if cfg.name == 'denseDilatedASPP':
            predicted_label = np.array(predicted_label + 0.5, dtype=np.uint8)
        else:
            predicted_label = np.array(predicted_label, dtype=np.uint8)
        '''outprob = sitk.GetImageFromArray(np.squeeze(value[0][:,:103,:198,:]))
        outprob.SetSpacing(patient_img.GetSpacing())
        outprob.SetDirection(patient_img.GetDirection())
        outprob.SetOrigin(patient_img.GetOrigin())
        sitk.WriteImage(outprob,os.path.join(pro_dir,patient_name + self.predict_tail))'''
        #print(patient_img.GetDirection(),patient_img.GetOrigin(),patient_img.GetSpacing())
        # for i in range(int(np.shape(predicted_label)[0])):
        #     for j in range(int(np.shape(predicted_label)[1])):
        #         for k in range(int(np.shape(predicted_label)[2])):
        #             if predicted_label[i][j][k] == 1:
        #                 print(1)
        outseg = sitk.GetImageFromArray(predicted_label)
        outseg.SetOrigin(patient_img.GetOrigin())
        outseg.SetDirection(patient_img.GetDirection())
        outseg.SetSpacing(patient_img.GetSpacing())
        #print('out_seg',np.shape(outseg))
        sitk.WriteImage(outseg, os.path.join(output_dir, patient_name + self.predict_tail))
        self.test_iter += 1
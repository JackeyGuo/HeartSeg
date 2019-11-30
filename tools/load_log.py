import tensorflow as tf
import os
import numpy as np

model_name = 'merge_model_sig' # denseDilatedASPP,xception_aspp,deform_xception_aspp,deeplab,merge_model_11 model_1113_80 unet_2d_entropy_
log_path = r'/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/log/val'%model_name

def load_log_data(log_path):
    total_loss = np.zeros((2000,))
    csv_path = open('../loss/%s.csv'%model_name, mode='w', encoding='utf-8')

    # e，即event，代表某一个batch的日志记录
    for logs in os.listdir(log_path):
        #csv_path = open('../loss/%s.csv' % logs, mode='w', encoding='utf-8')
        loss_data = []
        for e in tf.train.summary_iterator(os.path.join(log_path,logs)):
            # v，即value，代表这个batch的某个已记录的观测值，loss或者accuracy
            #print(e)
            for v in e.summary.value:
                #print(v)
                if v.tag == 'seg_loss':
                    #print(v.simple_value)
                    loss_data.append(v.simple_value)
                    #csv_path.write('{}\n'.format(v.simple_value))
        total_loss = np.array(total_loss) + np.array(loss_data)
        #print(total_loss[0])
    for loss in total_loss:
        csv_path.write('{}\n'.format(loss/5))
load_log_data(log_path=log_path)

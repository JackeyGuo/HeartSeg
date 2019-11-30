# model parameter
learn_rate = 0.001
lr_decay_step = 200
lr_decay_rate = 0.9
iter_step = 2000
# softmax or sigmoid
predict_op = 'softmax'
save_interval = 1000
classes = 2
# if use random_mirror
random_mirror = 0

fold_num = 5
batch_size = 64
one_by_one = False
begin_fold_iter = 1
end_fold_iter = 5
# model name [cnn_v1,cnn_v2, denseDilatedASPP, deeplab, mynet,
# resnet_aspp, xception_aspp, deform_xception_aspp, resnet_aspp_5fold,vnet,vnet_new,merge_model,
# vnet_2d] model_1113 unet_2d
name = 'unet_2d'
use_weight = False
use_dst_weight = False
only_jacc_loss = True
jacc_entropy_loss = False
use_jacc_loss = False

joint_train = False
use_param = False

if name == 'denseDilatedASPP':
    block_size=4
    block_count=4
    use_bc=1
    weight_loss=0
    use_dst_loss=1
    use_focal_dst_loss=0
    focal_loss=1
    focal_loss_r=0.5
    focal_loss_a=2

use_488_data = True
num_iter = 4
# data information
#data_path = '/home/data_new/dcm/training_T1'
if use_488_data:
    image_size = 128
    # data path 488
    data_path = '/home/data_new/guofeng/data/heart/training_T1_circle_%d/' % image_size
else:
    # data path 165
    data_path = '/home/data_new/guofeng/data/heart/training_T1_circle'

weight_distance_path= '/home/data_new/guofeng/data/heart/training_T1_dstmap'
# original patient data
patient_file_tail = '.mha'
# segment data
patient_seg_tail = '_seg.mha'
patient_weight_tail = '_dst.mha'

predict_tail = '.mha'
output_path = '/home/data_new/guofeng/projects/Segmentation/HeartSeg/result_0410/%s/' % (name)

# GPUs
gpu = '0'

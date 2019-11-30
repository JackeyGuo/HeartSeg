import os
import numpy as np
import SimpleITK as sitk

model_name = 'denseDilatedASPP_0903'# denseDilatedASPP,xception_aspp,deform_xception_aspp,deeplab
predict_output_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/predict" % (model_name)
bound_output_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/bound" % (model_name)

seg_tail = r"_seg.mha"

def _get_file_name(full_path):
    return os.path.basename(full_path)
seg = []
def create_bound(inputs,output_path):
    for f in inputs:
        #print(f)
        img = sitk.ReadImage(f)
        np_float = sitk.GetArrayFromImage(img)

        np_int = np.array(np_float,dtype=np.int32)
        img_int = sitk.GetImageFromArray(np_int)
        distmap = sitk.SignedMaurerDistanceMap(img_int)
        np_dist_map = sitk.GetArrayFromImage(distmap)
        zero_dist = np.logical_and(np_dist_map < 0.01,np_dist_map > -0.01)
        zero_dist = np.array(zero_dist,dtype=np.float32)
        #seg.append(zero_dist)
        #print(np.shape(seg),seg[0])
        # define the range of label 1,2,
        #result = seg[0] + seg[1]*2
        #result = seg[0]
        #print(np.shape(result[0]))
        opt_image = sitk.GetImageFromArray(zero_dist)
        print(f)
        sitk.WriteImage(opt_image,os.path.join(output_path,_get_file_name(f)))

def drow_gt_bound(gt_path, out_path):
    files = os.listdir(gt_path)
    file_list = []
    for f in files:
        if not -1 == f.find(seg_tail):
            file_list.append(os.path.join(gt_path,f))

    create_bound(file_list,out_path)

def drow_bound(target_path,out_path):
    files = os.listdir(target_path)
    file_list = []
    for f in files:
        file_list.append(os.path.join(target_path,f))

    create_bound(file_list,out_path)

if __name__ == "__main__":
    if not os.path.exists(bound_output_path):
        os.makedirs(bound_output_path)
    #files = os.listdir(predict_output_path)
    #lst = [r"D:\chromeDownload\CT MRI\CT+MRI\test_opt.mha",]
    # lst = []
    # for f in files:
    #     lst.append(os.path.join(predict_output_path,f))
    # create_bound(lst,bound_output_path)
    #open it while use.
    #drow_gt_bound('/home/data_new/guofeng/data/heart/training_T1_circle','/home/data_new/guofeng/data/heart/training_T1_bound')
    drow_bound(predict_output_path, bound_output_path)
    #drow_bound(r"D:\cuitjobs\newBegin\densedilatedaspp\provider\predict", r"D:\cuitjobs\newBegin\densedilatedaspp\bound")
    #drow_bound(r"D:\cuitjobs\newBegin\deeplab_like\provider\predict",r"D:\cuitjobs\newBegin\deeplab_like\bound")
    pass



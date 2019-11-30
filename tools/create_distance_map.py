import SimpleITK as sitk

import os
import numpy as np

seg_tail = r"_seg.mha"
output_tail = r"_dst.mha"

def _create_distance_map(target_path,output_path):
    files = os.listdir(target_path)
    for f in files:
        if f.find(seg_tail) == -1:
            continue
        print(f)
        name = f[:f.find(seg_tail)]
        file_path = os.path.join(target_path,f)
        simg = sitk.ReadImage(file_path)
        npa_simg = sitk.GetArrayFromImage(simg)
        rate = (1 - npa_simg.mean())/npa_simg.mean()
        print("with rate:",rate)
        dist = sitk.SignedDanielssonDistanceMap(simg)
        npa_dist = sitk.GetArrayFromImage(dist)
        max_v = npa_dist.max()
        npa_dist = np.where(npa_dist >= 0,npa_dist,0)
        npa_dist /= max_v
        npa_dist *= rate
        npa_dist += 1
        dist_opt = sitk.GetImageFromArray(npa_dist)
        dist_opt.SetSpacing(dist.GetSpacing())
        dist_opt.SetDirection(dist.GetDirection())
        dist_opt.SetOrigin(dist.GetOrigin())
        sitk.WriteImage(dist_opt,os.path.join(output_path,name + output_tail))

if __name__ == "__main__":
    _create_distance_map(r"/home/data_new/guofeng/data/heart/training_T1_circle", r"/home/data_new/guofeng/data/heart/training_T1_dstmap")
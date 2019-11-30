import os
#通过minidom解析xml文件
import xml.dom.minidom as xmldom
import numpy as np
import config.config as cfg
import sklearn.metrics as eval_metrics
# merge_model_multi_cmsf deeplab
model_name = 'unet_2d' # denseDilatedASPP,xception_aspp,deform_xception_aspp,deeplab,merge_model_11 model_1113_80 unet_2d_entropy merge_model_noaug
folder_name = 'result_0410'
#truth_path = r"/home/data_new/dcm/training_T1"
if cfg.use_488_data:
    truth_path = '/home/data_new/guofeng/data/heart/training_T1_circle_%d/' % cfg.image_size
else:
    truth_path = r"/home/data_new/guofeng/data/heart/truth_circle"

score_dir = r"/home/gf/Downloads/EvaluateSegmentation"
seg_tail = r"_seg.mha"

# where to output your .xml files
output_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/%s/%s/score" % (folder_name, model_name)
predict_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/%s/%s/predict" % (folder_name, model_name)
data_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/%s/%s/data" % (folder_name, model_name)

def compute_metrics(truth_path, predict_path, output_path):
    os.chdir(score_dir)
    predict_list = os.listdir(predict_path)
    #from_who = 'xcy'
    for f in predict_list:
        if model_name == 'unet':
            name = '_'.join(f.split('.')[0].split('_')[:-2])
            #print(name)
        else:
            name = f.split('.')[0]

        truth = os.path.join(truth_path, str(name) + seg_tail)
        predict = os.path.join(predict_path, f)
        #print(truth,predict)
        output = os.path.join(output_path, str(name) + ".xml")

        os.system("./EvaluateSegmentation " + truth + " " + predict + " –use all " + " -xml " + output)


def load_truth_class():
    npy_data = np.load(os.path.join(data_path,'data.npy'))
    cls_truth = []
    if cfg.use_488_data:
        pat_txt = open('/home/data_new/guofeng/projects/Segmentation/HeartSeg/txts/patient_%s.txt'%cfg.image_size, 'r')
        nor_txt = open('/home/data_new/guofeng/projects/Segmentation/HeartSeg/txts/normal_%s.txt'%cfg.image_size, 'r')
    else:
        pat_txt = open('/home/data_new/guofeng/projects/Segmentation/HeartSeg/txts/patient_T1_prep.txt', 'r')
        nor_txt = open('/home/data_new/guofeng/projects/Segmentation/HeartSeg/txts/normal_control_T1_prep.txt', 'r')

    pat_data = pat_txt.read().split('\n')
    nor_data = nor_txt.read().split('\n')

    for pat in npy_data:
        #print(pat)
        if pat + '.mha' in pat_data:
            cls_truth.append(1)
        if pat + '.mha' in nor_data:
            cls_truth.append(0)
    return cls_truth

def parse_xml(output_path):
    file_txt = open('/home/data_new/guofeng/projects/Segmentation/HeartSeg/%s/%s/result.txt'%(folder_name, model_name),'w')
    dice_txt = open('/home/data_new/guofeng/projects/Segmentation/HeartSeg/%s/%s/dice.txt'%(folder_name, model_name),'w')
    xml_list = os.listdir(output_path)
    #metrics = ['DICE','JACRD','AUC','SNSVTY','PRCISON','FMEASR']
    metrics = ['DICE', 'JACRD', 'AUC', 'SNSVTY', 'PRCISON']
    recall_value = []
    precision_value = []

    for metric in metrics:
        avg_value = 0
        xml_num = 488
        for xml in xml_list:
            try:
                # 得到文档对象
                domobj = xmldom.parse(os.path.join(output_path,xml))
                # 得到元素对象
                elementobj = domobj.documentElement
                # 获得子标签
                subElementObj = elementobj.getElementsByTagName("%s"%metric)
                # 获得标签属性值
                name = subElementObj[0].getAttribute("name")
                value = subElementObj[0].getAttribute("value")
                avg_value += float(value)
                #xml_num += 1
            except:
                print('wrong xml file:',xml)
                dice_txt.write(xml + '\n')
                continue
            if metric == 'DICE':
                dice_txt.write(xml+' '+str(value)+'\n')
                if float(value) > 0.89:
                    print(xml,value)
        #print(str(name)+' '+str(avg_value/len(xml_list)))
        #print(xml_num,len(xml_list))
        if metric == 'SNSVTY':
            recall_value = avg_value/xml_num
        if metric == 'PRCISON':
            precision_value = avg_value/xml_num

        file_txt.write(str(name)+' '+str(round(avg_value/xml_num,4))+'\n')
    seg_f1_score = 2 * recall_value * precision_value / (recall_value + precision_value)
    seg_f1_score = round(seg_f1_score,4)

    file_txt.write('SEGENTATION F1_SCORE' + ' ' + str(seg_f1_score) + '\n')
    print('SEGENTATION F1_SCORE' + ' ' + str(seg_f1_score))

    if cfg.joint_train:
        pred_class = list(np.load('/home/data_new/guofeng/projects/Segmentation/HeartSeg/%s/%s/result/result.npy'%(folder_name, model_name)))
        truth_class = load_truth_class()
        #print(np.shape(truth_class),np.shape(pred_class))
        cls_accuracy = round(eval_metrics.accuracy_score(truth_class,pred_class),4)
        print(truth_class)
        print(pred_class)
        auc = round(eval_metrics.roc_auc_score(truth_class,pred_class),4)
        cls_f1_score = round(eval_metrics.f1_score(truth_class,pred_class),4)

        print('CLASS ACCURACY' + ' ' + str(cls_accuracy))
        print('AUC' + ' ' + str(auc))
        print('CLASS F1_SCORE' + ' ' + str(cls_f1_score))

        file_txt.write('CLASS ACCURACY'+' '+str(cls_accuracy)+'\n')
        file_txt.write('AUC' + ' ' + str(auc) + '\n')
        file_txt.write('CLASS F1_SCORE' + ' ' + str(cls_f1_score) + '\n')
    return

if __name__ == "__main__":
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    compute_metrics(truth_path,predict_path,output_path)
    parse_xml(output_path)
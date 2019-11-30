import os
import sys
import json
import re
import argparse
model_name = 'xception_aspp'

output_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/score" % (model_name)
result_path = r"/home/data_new/guofeng/projects/Segmentation/HeartSeg/result/%s/score.txt" % (model_name)

def count(files, threshold=0.80):
    counts = {}
    error_file = 0
    result_file = open(result_path,'w')
    json_file = open('./json.txt','w')
    for file in files:
        with open(file) as f:
            try:
                jsinfo = json.load(f)
                filename = file.split('/')[-1]
                json_file.write(filename + ': ' + str(jsinfo[0]['DICE']) + '\n')
                for k in jsinfo[0]:
                    print(k)
                    if k == 'DICE' or k == 'PRECISION' or k == 'RECALL':
                        if not counts.get(k):
                            counts[k]=0
                        counts[k] += jsinfo[0][k]
                #print(jsinfo[0]['DICE'])
                # print file name
                if jsinfo[0]['DICE'] <= threshold:
                    error_file += 1
                    filename = file.split('/')[-1]
                    result_file.write(filename + ': ' + str(jsinfo[0]['DICE']) + '\n')
                    print(filename)
            except:
                print('wrong file :', file)
    print('error file number : ', error_file)

    precision = counts['PRECISION'] / len(files)
    recall = counts['RECALL'] / len(files)
    f1_score = 2 * precision * recall / (precision + recall)
    print('f1_score ', f1_score)
    for k in counts:
        counts[k] = counts[k] / len(files)
        print (k,counts[k])
        result_file.write(str(k) + ' '+ str(counts[k]) + '\n')
    result_file.write('F1_SCORE' + ' ' + str(f1_score) + '\n')


def runforfile(pat,dir,func):
    repa = re.compile(pat)
    countfiles = []
    for path,dirs,files in os.walk(dir):
        for file in files:
            if (repa.match(file)):
                countfiles.append(os.path.join(path, file))

    if len(countfiles) > 0:
        func(countfiles)


if __name__ == "__main__":
    AP = argparse.ArgumentParser(description='this script count the ASSD,DICE,and some thing others.')
    AP.add_argument('-target',
                    default=output_path)
    AP.add_argument('-pat',default='.*\.json')
    pa = AP.parse_args()
    runforfile(pa.pat,pa.target,count)

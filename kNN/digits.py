#!/usr/bin/env python

import numpy as np
import operator
from os import listdir

IMAGE_ROW = 32
IMAGE_COL = 32
IMAGE_SIZE = IMAGE_ROW * IMAGE_COL

def knn_classify(record, trainset, labels, k):
    n = trainset.shape[0]
    distances = np.tile(record, (n,1))
    distances -= trainset
    distances *= distances
    distances = np.sum(distances, axis=1)
    distances = distances ** 0.5
    sorted_indices = np.argsort(distances)
    class_cnt = {}
    for i in range(k):
        class_vote = labels[sorted_indices[i]]
        class_cnt[class_vote] = class_cnt.get(class_vote,0) + 1
    sorted_class = sorted(class_cnt.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class[0][0]

def imagefile2vector(filename):
    vec = np.zeros((1,IMAGE_SIZE))
    fp = open(filename)
    text = fp.readlines()
    for i in range(IMAGE_ROW):
        for j in range(IMAGE_COL):
            vec[0,i*IMAGE_COL+j] = int(text[i][j])
    return vec


def load_trainset(train_dir):
    train_files = listdir(train_dir)
    n = len(train_files)
    trainset = np.zeros((n,IMAGE_SIZE))
    labels = [ 0 for x in range(n) ]
    for i in range(n):
        filename = train_files[i]
        trainset[i,:] = imagefile2vector(train_dir+filename)
        labels[i] = int(filename.split('_')[0])
    return trainset, labels

def run_test(train_dir, test_dir):
    trainset, labels = load_trainset(train_dir)
    test_files = listdir(test_dir)
    cnt_all = len(test_files)
    cnt_correct = 0
    for filename in test_files:
        test_vec = imagefile2vector(test_dir+filename)
        answer = knn_classify(test_vec,trainset,labels,10)
        target = int(filename.split('_')[0])
        if answer == target: 
            cnt_correct += 1
        else: 
            print filename + ': ' + str(target) + ' classified as ' + str(answer)
    print cnt_correct * 1.0 / cnt_all

if __name__ == '__main__':
    run_test('./digits/trainingDigits/','./digits/testDigits/')


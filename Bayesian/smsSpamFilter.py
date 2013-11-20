#!/usr/bin/env python

import re
from numpy import *

def load_trainset(filename):
    fp = open(filename)
    content = fp.readlines()
    splitter = re.compile(r'\W+')
    labels = []
    trainset = []
    for line in content:
        terms = splitter.split(line)
        labels.append(terms[0])
        trainset.append([ t for t in terms[1:] if len(t)>2 ])
    return trainset, labels

def build_vocabulary(dataset):
    vocabulary = set()
    for vec in dataset:
        vocabulary |= set(vec)
    vocabulary = sorted(vocabulary)
    return vocabulary

def text2vec(vocabulary, text, mode='bag_of_words'):
    vec = [ 0 for word in vocabulary ]
    for term in text:
        if term in vocabulary:
            if mode == 'bag_of_words':
                vec[vocabulary.index(term)] += 1
            elif mode == 'set_of_words':
                vec[vocabulary.index(term)] = 1
        #else:
            #print term + ' is not in vocabulary.'
    return vec

def train_with_naive_bayes(trainset, labels, vocabulary):
    train_mtrx = array([ text2vec(vocabulary,text) for text in trainset ])
    priori_ham = labels.count('ham') * 1.0 / len(labels)
    priori_spam = labels.count('spam') * 1.0 / len(labels)
    cnt_ham = ones(len(vocabulary))
    cnt_spam = ones(len(vocabulary))
    cnt_ham_all = 2.0
    cnt_spam_all = 2.0
    for i in range(len(trainset)):
        if labels[i] == 'ham':
            cnt_ham += train_mtrx[i]
            cnt_ham_all += sum(train_mtrx[i])
        else:
            cnt_spam += train_mtrx[i]
            cnt_spam_all += sum(train_mtrx[i])
    likelihood_ham = log(cnt_ham/cnt_ham_all)
    likelihood_spam = log(cnt_spam/cnt_spam_all)
    return priori_ham, priori_spam, likelihood_ham, likelihood_spam 

def classify(vec, priori_ham, priori_spam, likelihood_ham, likelihood_spam):
    posterior_ham = sum(vec*likelihood_ham) + log(priori_ham)
    posterior_spam = sum(vec*likelihood_spam) + log(priori_spam)
    return 'ham' if posterior_ham > posterior_spam else 'spam'

def run_test(filename, vocabulary, priori_ham, priori_spam, likelihood_ham, likelihood_spam):
    fp = open(filename)
    content = fp.readlines()
    splitter = re.compile(r'\W+')
    cnt_all = 0
    cnt_correct = 0
    for line in content:
        terms = splitter.split(line)
        label = terms[0]
        vec = text2vec(vocabulary,terms[1:],'set_of_words')
        target = classify(vec,priori_ham,priori_spam,likelihood_ham,likelihood_spam)
        if label == target: cnt_correct += 1
        cnt_all += 1
    print cnt_correct * 1.0 / cnt_all

def run(train_filename, test_filename):
    trainset, labels = load_trainset(train_filename)
    vocabulary = build_vocabulary(trainset)
    priori_ham,priori_spam,likelihood_ham,likelihood_spam = train_with_naive_bayes(trainset,labels,vocabulary)
    run_test(test_filename,vocabulary,priori_ham,priori_spam,likelihood_ham,likelihood_spam)

if __name__ == '__main__':
    run('./sms/trainset.txt','./sms/testset.txt')


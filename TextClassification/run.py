#!/usr/bin/env python
#-*- encoding:utf-8 -*-

import sys, os

from preprocess import Preprocessor
from features import FeatureSelector
from bayes import BayesClassifier

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    pr = Preprocessor()
    pr.build_vocabulary_and_categories(train_file)

    fs = FeatureSelector(train_file, ck = 500)
    fs.select_features()

    bc = BayesClassifier(train_file, test_file, model = 'bernoulli')
    bc.train()
    bc.test()

#!/usr/bin/env python

from numpy import *
from logisticRegression import *

def parse_instance(text):
    terms = text.split(',')
    x = [1.0]
    for t in terms[1:-1]:
        if t == '?': x.append(0)
        else: x.append(float(t))
    v = 1 if int(terms[-1]) == 2 else 0
    return x, v

if __name__ == '__main__':
    run('./breastCancer/breast-cancer-train.txt',
        './breastCancer/breast-cancer-test.txt',
        parse_instance)

#!/usr/bin/env python

from numpy import *
from logisticRegression import *

def parse_instance(text):
    terms = text.split()
    x = [1.0]
    for i in range(23):
        if i == 2: continue
        t = terms[i]
        if i == 22: 
            if t == '?': v = 0
            else: v = 1 if int(t) == 1 else 0
        else:
            if t == '?': x.append(0)
            else: x.append(float(t))
    return x, v

if __name__ == '__main__':
    run('./horseColic/horse-colic.data',
        './horseColic/horse-colic.test',
        parse_instance)

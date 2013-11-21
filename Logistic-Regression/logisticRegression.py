#!/usr/bin/env python

from numpy import *

EPSILON = 1e-8

def load_transet(filename, parse_instance):
    fp = open(filename)
    content = fp.readlines()
    X = []
    y = []
    for line in content:
        x, t = parse_instance(line)
        X.append(x)
        y.append(t)
    X = mat(X)
    y = mat(y).transpose()
    return X, y

def sigmoid(x):
    return 1.0 / ( 1.0 + exp(-x) )

def batch_gradient_ascent(X, y, alpha=0.001, max_step_num=0):
    m = X.shape[0]
    n = X.shape[1]
    curr_theta = ones((n,1))
    step = 0
    while True:
        h = sigmoid(X*curr_theta)
        error = y - h
        next_theta = curr_theta + alpha * X.transpose() * error 
        diff = sum(abs(next_theta-curr_theta))
        if diff < EPSILON: break
        curr_theta = next_theta
        step += 1
        if max_step_num > 0 and step >= max_step_num: break
        if step % 1000 == 0: print curr_theta.transpose()
    return curr_theta

def stochastlc_gradient_ascent(X, y, alpha=0.001, max_step_num=0):
    m = X.shape[0]
    n = X.shape[1]
    curr_theta = ones((n,1))
    step = 0
    while True:
        for i in range(m):
            h = sigmoid(sum(X[i]*curr_theta))
            error = y[i] - h
            next_theta = curr_theta + alpha * X[i].transpose() * error
            diff = sum(abs(next_theta-curr_theta))
            if diff < EPSILON: return curr_theta
            curr_theta = next_theta
        step += 1
        if max_step_num > 0 and step >= max_step_num: return curr_theta
        print curr_theta.transpose()

def classify(theta, x):
    x = array(x)
    y = sigmoid(sum(theta*x))
    return 1 if y > 0.5 else 0

def test(filename, theta, parse_instance):
    fp = open(filename)
    content = fp.readlines()
    cnt_all = 0
    cnt_correct = 0
    for line in content:
        x, t = parse_instance(line)
        y = classify(theta,x)
        print t, y
        if t == y: cnt_correct += 1
        cnt_all += 1
    print cnt_correct * 1.0 / cnt_all

def run(train_file, test_file, parse_instance):
    X, y = load_transet(train_file,parse_instance)
    theta = batch_gradient_ascent(X,y,0.01,50000)
    #theta = stochastlc_gradient_ascent(X,y)
    test(test_file,theta, parse_instance)


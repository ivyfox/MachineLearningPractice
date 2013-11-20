#!/bin/bash

per=$1

cnt_ham=`grep '^ham' ./SMSSpamCollection | wc -l`
cnt_spam=`grep '^spam' ./SMSSpamCollection | wc -l`
cnt_train_ham=`echo "scale=0; $cnt_ham*$per/1" | bc`
cnt_train_spam=`echo "scale=0; $cnt_spam*$per/1" | bc`
cnt_test_ham=`echo "$cnt_ham-$cnt_train_ham" | bc`
cnt_test_spam=`echo "$cnt_spam-$cnt_train_spam" | bc`

grep '^ham' ./SMSSpamCollection | head -$cnt_train_ham > trainset.txt
grep '^spam' ./SMSSpamCollection | head -$cnt_train_spam >> trainset.txt
grep '^ham' ./SMSSpamCollection | tail -$cnt_test_ham > testset.txt
grep '^spam' ./SMSSpamCollection | tail -$cnt_test_spam >> testset.txt


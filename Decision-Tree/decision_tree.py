#!/usr/bin/env python

def load_trainset(filename):
    fp = open(filename)
    text = fp.readlines()
    fp.close()
    trainset = []
    for line in text:
        terms = line.split(',')
        trainset.append(terms)
    return trainset

def calculate_entropy(dataset):
    from math import log
    cnt_all = len(dataset)
    cnt_class = {}
    for vec in dataset:
        label = vec[-1]
        if label not in cnt_class: cnt_class[label] = 0
        cnt_class[label] += 1
    ent = 0.0
    for label in cnt_class:
        p = cnt_class[label] * 1.0 / cnt_all
        ent += -p*log(p,2)
    return ent

def partition_dataset_by_feature(dataset, idx_feature):
    result = {}
    for vec in dataset:
        feature = vec[idx_feature]
        if feature not in result: result[feature] = []
        result[feature].append(vec)
    return result

def find_best_feature(dataset, mask):
    num_features = len(dataset[0]) - 1
    base_entropy = calculate_entropy(dataset)
    best_info_gain = 0.0
    best_feature_idx = -1
    best_partition_result = {}
    for i in range(num_features):
        if mask[i] == False: continue
        new_dataset = partition_dataset_by_feature(dataset,i)
        curr_entropy = 0.0
        for key in new_dataset:
            prob = len(new_dataset[key]) * 1.0 / len(dataset)
            entropy = calculate_entropy(new_dataset[key])
            curr_entropy += prob * entropy
        if base_entropy - curr_entropy > best_info_gain:
            best_info_gain = base_entropy - curr_entropy
            best_feature_idx = i
            best_partition_result = new_dataset
    return best_feature_idx, best_partition_result

def majority_class(dataset):
    cnt_class = {}
    for vec in dataset:
        label = vec[-1]
        if label not in cnt_class: cnt_class[label] = 0
        cnt_class[label] += 1
    return max(cnt_class,key=cnt_class.get)

def build_decision_tree(dataset, mask):
    if len(set([ x[-1] for x in dataset ])) == 1:
        return dataset[0][-1]
    if len(dataset[0]) == 1: 
        return majority_class(dataset)
    best_feature_idx, best_partition_result = find_best_feature(dataset,mask)
    mask[best_feature_idx] = False
    tree = { best_feature_idx:{} }
    for key in best_partition_result:
        tree[best_feature_idx][key] = build_decision_tree(best_partition_result[key], mask[:])
    return tree

def build_decision_tree_from_file(filename):
    trainset = load_trainset(filename)
    mask = [ True for x in range(len(trainset[0])-1) ]
    return build_decision_tree(trainset,mask)

def store_decision_tree(tree, filename):
    import pickle
    fp = open(filename,'w')
    pickle.dump(tree,fp)
    fp.close()

def load_decision_tree(filename):
    import pickle
    fp = open(filename)
    return pickle.load(fp)

def classify(tree, vec):
    feature_idx = tree.keys()[0]
    sub_tree = tree[feature_idx]
    for key in sub_tree:
        if vec[feature_idx] == key:
            if type(sub_tree[key]).__name__ == 'dict':
                return classify(sub_tree[key],vec)
            else:
                return sub_tree[key]
    return -1

def run_test(tree, filename):
    fp = open(filename)
    cnt_all = 0
    cnt_correct = 0
    text = fp.readlines()
    fp.close()
    for line in text:
        terms = line.split(',')
        #print terms
        target = classify(tree,terms)
        label = terms[-1]
        if target == label: cnt_correct += 1
        cnt_all += 1
    print cnt_correct * 1.0 / cnt_all

def draw_tree(filename):
    import treePlotter
    trainset = load_trainset(filename)
    mask = [ True for x in range(len(trainset)-1) ]
    tree = build_decision_tree(trainset, mask)
    treePlotter.createPlot(tree)

if __name__ == '__main__':
    trainset = load_trainset('./poker-hand-training-true.data')
    mask = [ True for x in range(len(trainset)-1) ]
    tree = build_decision_tree(trainset, mask)
    run_test(tree, './poker-hand-testing.data')


#!/usr/bin/env python

from numpy import *
import sys

def load_graph(node_file, edge_file):
    nodes = load_nodes(node_file)
    graph = load_edges(edge_file)
    return nodes, graph

def load_nodes(filename):
    nodes = {}
    fp = open(filename,'r')
    content = fp.readlines()
    for line in content:
        terms = line.split()
        node_id = int(terms[0])
        url = terms[1]
        nodes[node_id] = url
    return nodes

def load_edges(filename):
    graph = {}
    fp = open(filename,'r')
    content = fp.readlines()
    for line in content:
        terms = line.split()
        from_id = int(terms[0])
        to_id = int(terms[1])
        if from_id not in graph: graph[from_id] = []
        if to_id not in graph: graph[to_id] = []
        graph[from_id].append(to_id)
    fp.close()
    return graph

def build_transmit_matrix(graph, d=0.85):
    n = max(graph.keys())
    mtrx = mat(zeros((n+1,n+1))) # the node-ids start from 1
    for from_id in graph:
        for to_id in graph[from_id]:
            mtrx[to_id,from_id] = 1.0 / len(graph[from_id])
    for node_id in graph:
        if len(graph[node_id]) == 0:
            mtrx[1:,node_id] = ones((n,1)) / n
    mtrx = mtrx[1:,1:]
    mtrx = d * mtrx + (1-d) * mat(ones((n,n))) / n
    return mtrx

def pagerank(P):
    n = P.shape[0]
    r = mat(ones((n,1))) / n
    step = 0
    while True:
        nr = P * r
        if sum(abs(nr-r)) < 1.0 / n / n: return r
        r = nr
        step += 1
        print str(step)+':', r.transpose()

def sort_pages(nodes, ranks):
    sorted_indices = argsort(ranks.transpose())
    sorted_indices = sorted_indices.tolist()[0][::-1]
    for idx in sorted_indices:
        node_id = idx + 1
        print nodes[node_id]

if __name__ == '__main__':
    filename = sys.argv[1]
    nodes, graph = load_graph(filename+'.nodes',filename+'.edges')
    mtrx = build_transmit_matrix(graph)
    ranks = pagerank(mtrx)
    sort_pages(nodes,ranks)


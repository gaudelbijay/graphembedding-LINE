import numpy as np 
import pandas as pd
import random,math

import tensorflow as tf 
from tensorflow.keras.layers import Embedding,Input,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k 

from utils import preprocess_nxgraph
from alias import create_alias_table,alias_sample

def line_loss(y_true,y_pred):
    return tf.reduce_sum(tf.log_sigmoid(y_true*y_pred))

def create_model(numNodes,embedding_size,order='second'):
    v_i = Input(shape=(1,))
    v_j = Input(shape=(1,))

    first_emb = Embedding(numNodes,embedding_size,name='first_emb')
    second_emb = Embedding(numNodes,embedding_size,name='second_emb')
    context_emb = Embedding(numNodes,embedding_size,name='context_emb')

    v_i_emb = first_emb(v_i)
    v_j_emb = first_emb(v_j)

    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)

    first = Lambda(lambda x:tf.reduce_sum(x[0]*x[1],axis=1,keepdims=False),name='first_order')([v_i_emb,v_j_emb])
    second = Lambda(lambda x:tf.reduce_sum(x[0]*x[1],axis=1,keepdims=False),name='second_order')([v_i_emb,v_j_emb])

    if order == 'first':
        output = [first]
    elif order == 'second':
        output = [second]
    else:
        output = [first,second]
    model = Model(input=[v_i,v_j],outputs=output)

    return model,{'first':first,'second':second}

class Line:
    def __init__(self,graph,embedding_size=8,negative_ratio=5,order='second'):
        self.graph = graph 
        self.embedding_size = embedding_size
        self.negative_ratio = negative_ratio

        if order not in ['first','second','third']:
            raise ValueError('order must be first, second or all')

        self.idx2node,self.node2idx = preprocess_nxgraph(graph)
        self.use_alias = True 
        self._embeddings = {}
        self.order = order
        self.nodes_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.samples_per_epoch = self.edge_size*(1+negative_ratio)
        self._gen_sampling_table()
        self.reset_model()

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = ((self.samples_per_epoch - 1) // self.batch_size + 1)*times

    def reset_model(self, opt='adam'):
        self.model, self.embedding_dict = create_model(self.node_size, self.rep_size, self.order)
        self.model.compile(opt, line_loss)
        self.batch_it = self.batch_iter(self.node2idx) 


    def _gen_sampling_table(self):
        #generating sampling table for vertex
        power = 0.75
        numNodes = self.nodes_size
        nodeDegree = np.zeros(numNodes) # To save outdegree 
        node2idx = self.node2idx
        for edge in self.graph.edges():
            nodeDegree[node2idx[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight',1.0)
        total_sum = sum([math.pow(node_degree[i], power) for i in range(numNodes)])
        norm_prob = [float(math.pow(node_degree[j], power))/total_sum for j in range(numNodes)]
        self.node_accept, self.node_alias = create_alias_table(norm_prob)



        #create sampling table for edge
        numEdges = self.graph.number_of_edges()
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0)
                         for edge in self.graph.edges()])
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) *
                     numEdges / total_sum for edge in self.graph.edges()]
        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)
    
    def batch_iter(self, node2idx):

        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]

        data_size = self.graph.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        while True:
            if mod == 0:

                h = []
                t = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                sign = np.ones(len(h))*-1
                t = []
                for i in range(len(h)):

                    t.append(alias_sample(
                        self.node_accept, self.node_alias))

            if self.order == 'all':
                yield ([np.array(h), np.array(t)], [sign, sign])
            else:
                yield ([np.array(h), np.array(t)], [sign])
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)

    def get_embeddings(self,):
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[
                                   0], self.embedding_dict['second'].get_weights()[0]))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding

        return self._embeddings

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        self.reset_training_config(batch_size, times)
        hist = self.model.fit_generator(self.batch_it, epochs=epochs, initial_epoch=initial_epoch, steps_per_epoch=self.steps_per_epoch,
                                        verbose=verbose)

        return hist

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
    return k.mean(k.log(k.sigmoid(y_true*y_pred)))

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





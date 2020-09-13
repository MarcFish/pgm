# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 19:40:06 2020

@author: zs_fi
"""


import tensorflow as tf
import tensorflow.keras as keras
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str,default='../data/train_rating.csv')
parser.add_argument("--test",type=str,default='../data/test_rating.csv')
parser.add_argument("--result",type=str,default='../results/result.txt')
parser.add_argument("--user_size",type=int,default=6040)
parser.add_argument("--item_size",type=int,default=3706)
parser.add_argument("--lr_u",type=float,default=0.01)
parser.add_argument("--lr_i",type=float,default=0.01)
parser.add_argument("--embed_size",type=int,default=128)
parser.add_argument("--lr",type=float,default=0.005)
parser.add_argument("--batch",type=int,default=2048)
parser.add_argument("--epochs",type=int,default=50)

arg = parser.parse_args()

class MF(keras.Model):
    def __init__(self, user_size, item_size, embed_size=128):
        super(MF, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.embed_size = embed_size

        self.user_embeddings = keras.layers.Embedding(input_dim=self.user_size, output_dim=self.embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(arg.lr_u),
                                                      embeddings_initializer='he_uniform')
        self.item_embeddings = keras.layers.Embedding(input_dim=self.item_size, output_dim=self.embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(arg.lr_i),
                                                      embeddings_initializer='he_uniform')

    def call(self, inputs):
        user_batch = inputs[0]
        item_batch = inputs[1]
        user_embeddings = self.user_embeddings(user_batch)
        item_embeddings = self.item_embeddings(item_batch)
        return tf.math.sigmoid(tf.reduce_sum(tf.multiply(user_embeddings, item_embeddings), axis=-1))

def loss_(y_true,y_pred):
    return keras.losses.MSE(y_true,y_pred)/2

rating = pd.read_csv(arg.file)
test = pd.read_csv(arg.test)
user_size = arg.user_size
item_size = arg.item_size
mf = MF(user_size=user_size, item_size=item_size,embed_size=arg.embed_size)
mf.compile(loss=loss_, optimizer=keras.optimizers.SGD(arg.lr),metrics=[keras.metrics.RootMeanSquaredError()])
mf.fit(x=[rating.user.values, rating.movie.values], y=rating.rating.values, epochs=arg.epochs, batch_size=arg.batch)
loss,rmse = mf.evaluate(x=[test.user.values,test.movie.values],y=test.rating.values)
print("RMSE:{:.4f}".format(rmse))
with open(arg.result, 'a',encoding='utf-8',newline='') as f:
    f.write("PMF,{:.4f}".format(rmse))

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:12:45 2019

@author: MarcFish
"""

import logging
import numpy as np
from numpy.linalg import inv
from scipy.stats import wishart
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix as csr
from datetime import datetime

logging.basicConfig(format="%(asctime)s %(name)s %(message)s",level=logging.DEBUG,filename=r'BPMF.log')

class BPMF:
    
    def __init__(self,coor_x,coor_y,val,max_rat,min_rat,user_num,movie_num,U=None,V=None,alpha = 2.0,mu0 = 0.0,D=30,T=100):
        """
        coor和val是表示稀疏的评分矩阵
        coor是一个列表，每一项为一个列表，此列表第一项为用户，第二项为电影
        val是一个列表，每一项为coor中对应的评分
        """
        self.alpha = alpha
        self.mu0 = mu0
        self.D = D
        self.v0 = D
        self.beta0 = 2.0
        
        self.val = val
        self.N = user_num
        self.M = movie_num
        
        self.max_rat = max_rat
        self.min_rat = min_rat
        
        self.T = T
        self.R = csr((val,(coor_x,coor_y)),shape=(self.N,self.M))
        self.U = np.random.normal(size=(self.N,self.D))
        self.V = np.random.normal(size=(self.M,self.D))
        
        self.W0_user = np.eye(self.D)
        self.W0_item = np.eye(self.D)
        
        self.rmses = []

    def train(self):
        for t in range(self.T):
            self.getR_()
            logging.info('iter'+str(t)+'time:'+str(datetime.now())+'rmse:{:.6f}'.format(self.getRMSE()))
            print('iter'+str(t)+'time:'+str(datetime.now())+'rmse:{:.6f}'.format(self.getRMSE()))
            self.update_user_params()
            self.update_item_params()
            self.update_user_features()
            self.update_item_features()
        

    def getRMSE(self):
        R = self.R.todense()
        R_t = R.astype('bool').astype('int')
        #R = (R-self.min_rat)/(self.max_rat-1)
        rmse = np.sqrt(np.sum(np.square(np.multiply(R_t,(R - self.R_))))/(self.R.data.shape[0]))
        #rmse = np.sqrt(np.sum(np.square(R - self.R_))/(self.N*self.M))
        
        self.rmses.append(rmse)
        return rmse
        
    def getR_(self):
        #self.R_ = np.divide(1,1+np.exp(-np.matmul(self.U,self.V.T)))
        self.R_ = np.matmul(self.U,self.V.T)
        #self.R_ = multivariate_normal.rvs(mean=np.matmul(self.U,self.V.T),cov=1/self.alpha)
    
    def update_user_features(self):
        for i in range(self.N):
            Lambda_user_ = np.zeros(shape=[self.D,self.D])
            mu_user = np.zeros(shape=[1,self.D])
            
            for j in range(self.M):
                if self.R[i,j] == 0:
                    Lambda_user_ += 1.0
                    mu_user += 1.0
                else:
                    Lambda_user_ += np.matmul(self.V[j].reshape(1,self.D).T,self.V[j].reshape(1,self.D))
                    mu_user += self.V[j].reshape(1,self.D) * self.R[i,j]
            '''
            Lambda_user_ = np.matmul(self.V.T,self.V)
            mu_user = np.array(np.matmul(self.V.T,self.R[i].todense().T).reshape(1,self.D))
            '''
            Lambda_user_inv = inv(self.alpha * Lambda_user_ +  self.Lambda_user)
            mu_user_ = np.matmul((mu_user * self.alpha + np.matmul(self.mu_user.reshape(1,self.D),self.Lambda_user.T)),Lambda_user_inv.T).reshape(self.D,)
            self.U[i,:] = multivariate_normal.rvs(mean=mu_user_,cov=Lambda_user_inv)
    
    def update_item_features(self):
        for j in range(self.M):
            Lambda_item_ = np.zeros(shape=[self.D,self.D])
            mu_item = np.zeros(shape=[1,self.D])
            
            for i in range(self.N):
                if self.R[i,j] == 0:
                    Lambda_item_ += 1.0
                    mu_item += 1.0
                else:
                    Lambda_item_ += np.matmul(self.U[i].reshape(1,self.D).T,self.U[i].reshape(1,self.D))
                    mu_item += self.U[i].reshape(1,self.D) * self.R[i,j]
            '''
            Lambda_item_ = np.matmul(self.U.T,self.U)
            mu_item = np.array(np.matmul(self.U.T,self.R[:,j].todense()).reshape(1,self.D))
            '''
            Lambda_item_inv = inv(self.alpha * Lambda_item_ + self.Lambda_item)
            mu_item_ = np.matmul((mu_item * self.alpha + np.matmul(self.mu_item.reshape(1,self.D),self.Lambda_item.T)),Lambda_item_inv.T).reshape(self.D,)
            self.V[j,:] = multivariate_normal.rvs(mean=mu_item_,cov = Lambda_item_inv)
        
    def update_item_params(self):
        V_ = np.mean(self.V,0)
        S_ = np.matmul(np.transpose(self.V - V_),self.V - V_)/self.M
        diff_mu0_V_ = self.mu0 - V_
        W0_item_ = inv(inv(self.W0_item) + S_ * self.M + (self.beta0 * self.M / (self.beta0 + self.M)) * np.matmul(np.transpose(diff_mu0_V_),diff_mu0_V_))
        W0_item_ = (W0_item_ + W0_item_.T) / 2.0
        
        beta_ = self.beta0 + self.M
        v0_ = self.v0 + self.M
        mu0_ = (self.beta0 * self.mu0 + self.M * V_)/(self.beta0 + self.M)
        
        self.Lambda_item = wishart.rvs(df=v0_,scale = W0_item_)
        self.mu_item = multivariate_normal.rvs(mean=mu0_,cov=inv(beta_ * self.Lambda_item))

    def update_user_params(self):
        U_ = np.mean(self.U,0)
        S_ = np.matmul(np.transpose(self.U - U_),self.U - U_)/self.N
        diff_mu0_U_ = self.mu0 - U_
        W0_user_ = inv(inv(self.W0_user) + S_ * self.N + (self.beta0 * self.N / (self.beta0 + self.N)) * np.matmul(np.transpose(diff_mu0_U_),diff_mu0_U_))
        W0_user_ = (W0_user_ + W0_user_.T) / 2.0
        
        beta_ = self.beta0 + self.N
        v0_ = self.v0 + self.N
        mu0_ = (self.beta0 * self.mu0 + self.N * U_)/(self.beta0 + self.N)
        
        self.Lambda_user = wishart.rvs(df=v0_,scale = W0_user_)
        self.mu_user = multivariate_normal.rvs(mean=mu0_,cov=inv(beta_ * self.Lambda_user))
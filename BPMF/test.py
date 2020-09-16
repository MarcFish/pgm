# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:59:00 2019

@author: MarcFish
"""

from BPMF_numpy import BPMF

filepath = 'E:/project/data/raw/ml-1m/ratings.dat'

with open(filepath,'r',encoding='utf-8') as f:
    ls = f.readlines()

coor = []
val = []
coor_x = []
coor_y = []
user_dict = {}
movie_dict = {}
rat_dict = {}
u_c = 0
m_c = 0
v_c = 1
for l in ls:
    r = l.split('::')
    u = int(r[0])
    m = int(r[1])
    v = int(r[2])
    if not user_dict.get(u):
        user_dict[u] = u_c
        u_c += 1
    if not movie_dict.get(m):
        movie_dict[m] = m_c
        m_c += 1
    if not rat_dict.get(v):
        rat_dict[v] = v_c
        v_c +=1
        
    coor.append([user_dict.get(u),movie_dict.get(m)])
    coor_x.append(user_dict.get(u))
    coor_y.append(movie_dict.get(m))
    val.append(rat_dict.get(v))
max_rat = v_c - 1
min_rat = 1
user_num = len(user_dict) + 1
movie_num = len(movie_dict) + 1


B = BPMF(coor_x=coor_x,coor_y=coor_y,val=val,max_rat = max_rat,min_rat = min_rat,user_num = user_num,movie_num = movie_num)
#B.train()
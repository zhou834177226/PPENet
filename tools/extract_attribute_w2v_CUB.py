# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:43:05 2019

@author: badat
"""
import os,sys
pwd = os.getcwd()
sys.path.insert(0, pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import pdb
import pandas as pd
import numpy as np
import gensim.downloader as api
import pickle
#%%

# _DEFAULT_BASE_DIR = os.path.expanduser('~/gensim-data')
# BASE_DIR = os.environ.get('GENSIM_DATA_DIR')
# print(BASE_DIR)

print('Loading pretrain w2v modeling')
model_name = 'word2vec-google-news-300'  # best modeling
model = api.load(model_name)
#从网络下载？
dim_w2v = 300
print('Done loading modeling')
#%%
replace_word = [('spatulate', 'broad'), ('upperparts', 'upper parts'), ('grey', 'gray')]
#%%
path = 'datasets/Attribute/attribute/{}/attributes.txt'.format('CUB')
df = pd.read_csv(path, sep=' ', header=None, names=['idx', 'des'])
des = df['des'].values
#%% filter
new_des = [' '.join(i.split('_')) for i in des]
new_des = [' '.join(i.split('-')) for i in new_des]
new_des = [' '.join(i.split('::')) for i in new_des]
new_des = [i.split('(')[0] for i in new_des]
new_des = [i[4:] for i in new_des]  #  ？？？
# 改分隔符？
#%% replace out of dictionary words
for pair in replace_word:
    for idx, s in enumerate(new_des):
        new_des[idx]=s.replace(pair[0], pair[1])
print('Done replace OOD words')
#%%
df['new_des']=new_des
df.to_csv('datasets/Attribute/attribute/CUB/new_des.csv')
# 保存数据
print('Done preprocessing attribute des')
#%%
all_w2v = []
for s in new_des:
    print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
    all_w2v.append(w2v[np.newaxis,:])
# %%
all_w2v=np.concatenate(all_w2v,axis=0)
# 拼接array
pdb.set_trace()
# pdb.set_trace() # 设置追踪断点
# ENTER (重复上次命令)
# c (继续)
# l (查找当前位于哪里)
# s (进入子程序,如果当前有一个函数调用，那么 s 会进入被调用的函数体)
# n(ext) 让程序运行下一行，如果当前语句有一个函数调用，用 n 是不会进入被调用的函数体中的
# r (运行直到子程序结束)
# !<python 命令>
# h (帮助)
# a(rgs) 打印当前函数的参数
# j(ump) 让程序跳转到指定的行数
# l(ist) 可以列出当前将要运行的代码块
# p(rint) 最有用的命令之一，打印某个变量
# q(uit) 退出调试
# r(eturn) 继续执行，直到函数体返回
with open('datasets/Attribute/w2v/CUB_attribute.pkl', 'wb') as f:
    pickle.dump(all_w2v, f)
    # 将all_w2v保存进文件f中

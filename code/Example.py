#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os


# In[ ]:





# In[4]:


import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
 
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
session = tf.Session(config=config)
 
KTF.set_session(session)


# In[5]:


from utils import *
from preprecoess import *
from generator import *
from models import *
from fl_training import *


# In[ ]:





# In[6]:


root_data_path = '/home/lunwang/FedNewsRec/data' # MIND-Dataset Path
embedding_path = '/home/lunwang/FedNewsRec/wordvec' # Word Embedding Path


# In[ ]:





# In[7]:


# Read News
news,news_index,category_dict,subcategory_dict,word_dict = read_news(root_data_path,['train','val'])
news_title,news_vert,news_subvert=get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict)
title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)


# In[ ]:





# In[8]:


#Parse User
train_session, train_uid_click, train_uid_table = read_clickhistory(root_data_path,'train')
test_session, test_uid_click,test_uid_table = read_clickhistory(root_data_path,'val')
train_user = parse_user(train_session,news_index)
test_user = parse_user(test_session,news_index)
train_sess, train_user_id, train_label, train_user_id_sample = get_train_input(train_session,train_uid_click,news_index)
test_impressions, test_userids = get_test_input(test_session,news_index)


# In[ ]:





# In[9]:


get_user_data = GetUserDataFunc(news_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id)


# In[ ]:





# In[ ]:


lr = 0.3
delta = 0.05
lambd = 0
ratio = 0.02
num = 6

model, doc_encoder, user_encoder, news_encoder = get_model(lr,delta,title_word_embedding_matrix)
Res = []
Loss = []
count = 0
while True:
    loss = fed_single_update(model,doc_encoder,user_encoder,num,lr,get_user_data,train_uid_table)
    Loss.append(loss)
    if count % 25 == 0:
        news_scoring = news_encoder.predict(news_title,verbose=0)
        user_generator = get_hir_user_generator(news_scoring,test_user['click'],64)
        user_scoring = user_encoder.predict_generator(user_generator,verbose=0),
        user_scoring = user_scoring[0]
        g = evaluate(user_scoring,news_scoring,test_impressions)
        Res.append(g)
        print(g)
        with open('FedRec-woLDP-1.json','a') as f:
            s = json.dumps(g) + '\n'
            f.write(s)
    count += 1


# In[ ]:





# In[ ]:





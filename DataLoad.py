#import pyodbc
import codecs
import csv
import datetime
import time
import re
import os
import io
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import *
from collections import Counter
import csv
import os
import heapq
import time
import datetime
import xlrd
import xlwt
import json
import string
import gensim
from gensim import models
from gensim.models import Word2Vec
from gensim import corpora
from config import conf
from torch.nn.utils.rnn import pad_sequence
import torch

Dir=os.path.abspath('')
cfg=conf()
'''
class Word2Vector():
    def __init__(self):
        super(Word2Vector, self).__init__()
        
    def TrainSaveWordtoVev():
        emb_dim = cfg.embedding_size2 #title
        conn1 = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;PORT=1433;DATABASE=Tuniu Deep Learning',
                               charset="utf8")
        cur1 = conn1.cursor()
        sql = 'SELECT [routeName] FROM [Tuniu Deep Learning].[dbo].[ItemInformation fenci]'
        cur1.execute(sql)
        resList = cur1.fetchall()
        conn1.commit()
        sens_list=[]
        for data in resList:
            sen=data[0].split('\ ')
            sens_list.append(sen)
        #print (np.array(sens_list).shape)
        model = word2vec.Word2Vec(sens_list, min_count=1, iter=20, size=emb_dim)
        model.save("word2vec.model")
'''
model = Word2Vec.load('word2vec.model')
#model=gensim.models.keyedvectors.KeyedVectors.load_word2vec_format
#print(model['三亚'])
#print (model['海南'])
#print( model['九寨沟'])
# print ("海南和三亚的相似度:")
#print(model.similarity('三亚','海南'))    # 两个词的相似性距离
# print ("海南和九寨沟的相似度:")
# print (model.similarity('海南','九寨沟'))    # 两个词的相似性距离
#
# print ("海南和马尔代夫的相似度:")
# print (model.similarity('海南','马尔代夫'))    # 两个词的相似性距离
#
# print ("普吉岛和马尔代夫的相似度:")
# print (model.similarity('普吉岛','马尔代夫'))    # 两个词的相似性距离

class TPData():
    @property
    def TP(self):  #Travel Package数据
        ItemIDList = []  # ItemID
        Titlefencilist = []  # 旅游标题
        Title_embedding_list = []  # 旅游标题
        Travel_RegionList = []  # TR旅游区域
        Travel_TypeList = []  # 旅游类型
        DestinationList = []  # 目的地
        filename='Travel Products Infor.xls'
        data = xlrd.open_workbook(Dir+'/data/'+filename)
        table = data.sheets()[0]          #通过索引顺序获取TP
        for i in range(0,table.nrows):  #遍历excel文档，获取信息存储在list
            #print (table.row(i))
            ItemID_i=int(table.cell_value(i,0))
            ItemIDList.append(ItemID_i)
            title_i=table.cell_value(i,1)
            #title_i_list=title_i.split(' ')
            Title_fenci=title_i.split(' ')
            #for word in Title_fenci:
                #Title_Words.append(word)
            #print(Title_fenci)
            Titlefencilist.append(Title_fenci)
            #print(Title_embedding.shape)
            #print('Title_embedding:',Title_embedding)
            #print (Title_fenci)
            #Title_embedding_list.append(Title_embedding)
            Destination_i = table.cell_value(i, 2)
            DestinationList.append([Destination_i])
            #print(DestinationList)
            Travel_region_i = table.cell_value(i, 3)
            Travel_RegionList.append([Travel_region_i])
            Travel_type_i = table.cell_value(i, 4)
            Travel_TypeList.append([Travel_type_i])
        #length=max([len(x) for x in Titlefencilist])
        #print ('max len of title:',length)
        #Titlefencilist=np.array(Titlefencilist)
        #Titlefencilist = Titlefencilist.reshape(Titlefencilist.shape[0], 1)
        dictionary_title = corpora.Dictionary(Titlefencilist)  # 字典
        #print ('word:',dictionary_title[269],'270')  #269+1=270,三峡
        #print ('WordID_embedding:',model['三峡'])
        vocab_size_title = len(dictionary_title)
        TitleWordIDlist=[]
        for title in Titlefencilist:
            #print('title:',title)
            TitlefenciIDList=[(dictionary_title.token2id[word]+1) for word in title]   #转换为ID,ID从1开始
            #print (TitlefenciIDList)
            #TitleWordIDlist.append(np.array(TitlefenciIDList))
            TitleWordIDlist.append(torch.Tensor(TitlefenciIDList))
        TitleWordIDlist= pad_sequence(TitleWordIDlist, batch_first=True).int()#所有的title统一补齐
        #print (TitleWordIDlist.size())
        #TitleWordIDlist=np.array(TitleWordIDlist)
        #print (TitleWordIDlist.shape)
        WordID_embedding=[model[dictionary_title[word]] for word in dictionary_title]
        WordID_embedding.insert(0,np.zeros(cfg.embedding_size2))  #抬头补0向量，与Title中的WordID保持一致
        #print('WordID_embedding2:', WordID_embedding[270])
        #print('WordID_embedding:',WordID_embedding)
        np.save('WordID_embedding',WordID_embedding)  #ID对应vocab_size_title-embedding
        #WordID_embedding=np.load('WordID_embedding.npy')
        # print('2晚',dictionary_title.token2id['2晚']+1)
        # print(model['2晚'])
        # print(WordID_embedding[dictionary_title.token2id['2晚']+1])
        DestinationList=np.array(DestinationList)
        DestinationList=DestinationList.reshape(DestinationList.shape[0],1)
        #print(DestinationList.shape)
        dictionary_Destination = corpora.Dictionary(DestinationList)  # 转换为字典
        #print ('dictionary_Destination:',dictionary_Destination )
        vocab_size_Destination = len(dictionary_Destination)  #不重复词总数
        #print (vocab_size_Destination,dictionary_Destination.token2id['上海'])  #key:为目的地，value为ID
        #print(dictionary_Destination.dfs)  # 出现文档次数
        DestinationIDList=[(dictionary_Destination.token2id[x[0]]+1) for x in DestinationList]  #转换为ID对应

        Travel_RegionList=np.array(Travel_RegionList)
        Travel_RegionList=Travel_RegionList.reshape(Travel_RegionList.shape[0],1)
        dictionary_Travel_Region= corpora.Dictionary(Travel_RegionList)  # 转换为字典
        #print('dictionary_Travel_Region:',dictionary_Travel_Region)
        vocab_size_Travel_Region = len(dictionary_Travel_Region)  #不重复词总数
        Travel_RegionIDList=[(dictionary_Travel_Region.token2id[x[0]]+1) for x in Travel_RegionList]  #转换为ID对应

        Travel_TypeList=np.array(Travel_TypeList)
        #print(Travel_TypeList)
        Travel_TypeList=Travel_TypeList.reshape(Travel_TypeList.shape[0],1)
        dictionary_Travel_Type= corpora.Dictionary(Travel_TypeList)  # 转换为字典
        #print('dictionary_Travel_Type:',dictionary_Travel_Type)
        vocab_size_Travel_Type = len(dictionary_Travel_Type)  #不重复词总数
        Travel_TypeIDList=[(dictionary_Travel_Type.token2id[x[0]]+1) for x in Travel_TypeList]  #转换为ID对应

        table = data.sheets()[1]          #通过索引顺序获取UserID
        UserID_list=[]
        for i in range(0,table.nrows):  #遍历excel文档，获取信息存储在list
            #print (table.row(i))
            UserID_i=int(table.cell_value(i,0))
            UserID_list.append(str(UserID_i))
        dictionary_UserID = corpora.Dictionary([UserID_list])  # 字典
        vocab_size_UserID = len(dictionary_UserID)
        #print (UserID_list)
        #print (vocab_size_UserID)
        #return Title_maxlen,vocab_size_title,vocab_size_Destination,vocab_size_Travel_Region,vocab_size_Travel_Type,vocab_size_UserID,UserID_list
        return np.array(ItemIDList), np.array(TitleWordIDlist), np.array(DestinationIDList), np.array(Travel_RegionIDList), np.array(Travel_TypeIDList)

#
# class UserData(): #UserID
#     def UserInput():
#         filename='Travel Products Infor.xls'
#         data = xlrd.open_workbook(Dir+'/data/'+filename)
#         table = data.sheets()[1]          #通过索引顺序获取
#         UserID_list=[]
#         for i in range(0,table.nrows):
#             #print (table.row(i))
#             UserID_i=int(table.cell_value(i,0))
#             #print(UserID_i)
#             UserID_list.append(UserID_i)
#         UserID_list=torch.Tensor(UserID_list)
#         #print(UserID_list)
#         return  UserID_list

class ClickStreamData():  #点击流和Groundtryth购买数据
    def ClickStream(month):
        filename='ClickStream'+month+'.xls'
        data = xlrd.open_workbook(Dir+'/data/'+filename)
        table = data.sheets()[0]          #通过索引顺序获取
        UserIDlist=[]
        Training_Userdata=[]   #用1-29号下单数据训练
        Test_Userdata=[]       #用30和31号下单数据测试预测
        OrderItemIDList=[]
        #long_len=[]
        #short_len=[]
        ItemID_train=[]
        Order_train_ItemIDList=[]
        ItemID_test=[]
        Order_test_ItemIDList=[]
        for i in range(0,table.nrows):
            #print (table.row(i))
            UserID_i=int(table.cell_value(i,0))
            UserIDlist.append(UserID_i)
            #print ("UserID_i:",UserID_i)
            ClickItemID_i=table.cell_value(i, 1).strip('\n') #点击项目id列
            #print("ClickItemID_i",ClickItemID_i)
            if '|' in ClickItemID_i: #拆项目id，如果这行里带| ，就根据|拆完分隔开形成新的项目id列，如果没有|就直接塞
                ClickItemID_i_list=ClickItemID_i.split('|')
            else:
                ClickItemID_i_list=[int(ClickItemID_i)]
            ClickItemID_i_list=[int(x) for x in ClickItemID_i_list] #对list中的所有id都转成int行
            #print (ClickItemID_i_list)
            operate_time_i = table.cell_value(i, 2) #这里是用户的操作时间列
            operate_time_i_list= operate_time_i.split('|') #有|就将时间分隔开来
            #print (operate_time_i_list)
            session_time_i=table.cell_value(i, 3) #用户的会话时间列
            session_time_i_list=session_time_i.split('|') #有|就分隔时间
            #print("session_time_i_list:",session_time_i_list)
            Curren_Sessiontime=session_time_i_list[-1] #用户最近的会话时间，例如[a,b,c,d,e]，Curren_Sessiontime=e
            #print("Curren_Sessiontime:",Curren_Sessiontime)
            ClickItemID_i_list_Short=[] #初始化
            ClickItemID_i_list_Long=[]#初始化
            j=session_time_i_list.index(Curren_Sessiontime) #j是某一用户最近的会话时间在该行中排第几个的序号值，比如如果用户最近的会话时间是X，那在这一行[a,b,x,c,t,x]中，j=2，或者[a,v,x],j=2
            if i==0: #该类用户只有一个会话，冷启动用户
                ClickItemID_i_list_Short =ClickItemID_i_list[j:] #ClickItemID_i_list_Short是某一用户在他最近的会话时间后截出SHORT的list，比如j=2时 [a,b,c,t,x],ClickItemID_i_list_Short=[t,x]
                ClickItemID_i_list_Long = []
            else:
                ClickItemID_i_list_Short =ClickItemID_i_list[j:]
                ClickItemID_i_list_Long = ClickItemID_i_list[:j]

            #print ('UserID:',UserID_i)
            #print ('Short-Term:',ClickItemID_i_list_Short)
            #print ('Long-Term:',ClickItemID_i_list_Long)
            OrderItemID_i=int(table.cell_value(i, 4))
            OrderItemIDList.append(OrderItemID_i)
            #print ('OrderItemID_i:',OrderItemID_i)
            OrderTime_i=table.cell_value(i, 5)
            #print('OrderTime_i:', OrderTime_i)
            OrderTime_i_v1=time.strptime(OrderTime_i, "%Y-%m-%d %H:%M:%S")
            CutTime='2013-'+str(month)+'-30 00:00:00'
            #print ('CutTime:',CutTime)
            CutTime=time.strptime(CutTime, "%Y-%m-%d %H:%M:%S")
            Short_N=cfg.Short_N
            Long_N=cfg.Long_N
            #Ngram=False
            Ngram=True
            if Ngram==True:  #依据N就行截取
                ClickItemID_i_list_Long=np.array(ClickItemID_i_list_Long[-Long_N:])
                ClickItemID_i_list_Short=np.array(ClickItemID_i_list_Short[-Short_N:])

            if OrderTime_i_v1<CutTime:  #切割训练集和测试集
                Training_Userdata.append([UserID_i,ClickItemID_i_list_Long,ClickItemID_i_list_Short,OrderItemID_i])#加入训练集
                Train_Item_i=np.append(ClickItemID_i_list_Long,ClickItemID_i_list_Short)
                Order_train_ItemIDList.append(OrderItemID_i)
                for x in Train_Item_i:
                    ItemID_train.append(x)


            else:
                Test_Userdata.append([UserID_i,ClickItemID_i_list_Long,ClickItemID_i_list_Short,OrderItemID_i])  #加入测试集
                Test_Item_i=np.append(ClickItemID_i_list_Long,ClickItemID_i_list_Short)
                Order_test_ItemIDList.append(OrderItemID_i)
                for x in Test_Item_i:
                    ItemID_test.append(x)
        # #########*****将N-gram后的Long和Short-term的ItemID导入数据库，便于对ItemID的规模统计*****#############
        # ItemID=np.append(ItemID_test,ItemID_train)
        # ItemID=list(set(ItemID))
        # conn1 = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;PORT=1433;DATABASE=Tuniu Deep Learning',
        #                        charset="utf8")
        # cur1 = conn1.cursor()
        # sql = 'delete FROM [Tuniu Deep Learning].[dbo].[ItemInformation '+month+' Final]'
        # cur1.execute(sql)
        # conn1.commit()
        # for ID in ItemID:
        #     if ID!='':
        #         ID=str(ID)
        #         sql = 'insert into [Tuniu Deep Learning].[dbo].[ItemInformation '+month+' Final] Values (' +ID+ ')'
        #         cur1.execute(sql)
        # conn1.commit()
        Itemclick=np.append(ItemID_train,ItemID_test)
        ItemID=np.append(OrderItemIDList,Itemclick)

        FinalItemID=[int(x)for x in list(set(ItemID))]
        ItemID_train=np.append(Order_train_ItemIDList,ItemID_train)
        TrainItemID=[int(x)for x in list(set(ItemID_train))]
        ItemID_test=np.append(Order_test_ItemIDList,ItemID_test)
        TestItemID=[int(x)for x in list(set(ItemID_test))]
        Training_Userdata=np.array(Training_Userdata)
        #print (Training_Userdata.shape)
        '''
        ItemID_long_train = Training_Userdata[:, 1]  # long-termd点击流
        long_len=[len(x) for x in ItemID_long_train]
        print('average long_len:',np.mean(long_len))
        ItemID_short_train = Training_Userdata[:, 2]  # short-term点击流
        short_len = [len(x) for x in ItemID_short_train]
        print('average short_len:', np.mean(short_len))
        UserID_train = Training_Userdata[:, 0]  # UserID
        print(len(set(UserID_train)))
        print(len(set(ItemID_train)))
        print('records:',len(ItemID_train))
        Targe_traing = Training_Userdata[:, 3]  # 下单的ItemID
        print(len(set(Targe_traing)))

        Test_Userdata=np.array(Test_Userdata)
        print (Test_Userdata.shape)
        ItemID_long_train = Test_Userdata[:, 1]  # long-termd点击流
        long_len=[len(x) for x in ItemID_long_train]
        print('average long_len:',np.mean(long_len))
        ItemID_short_train = Test_Userdata[:, 2]  # short-term点击流
        short_len = [len(x) for x in ItemID_short_train]
        print('average short_len:', np.mean(short_len))
        UserID_test = Test_Userdata[:, 0]  # UserID
        print(len(set(UserID_test)))
        print(len(set(ItemID_test)))
        print('records:',len(ItemID_test))
        Targe_test = Test_Userdata[:, 3]  # 下单的ItemID
        print(len(set(Targe_test)))
        '''
        return Training_Userdata,Test_Userdata,FinalItemID,TrainItemID,TestItemID


if __name__ == '__main__':

    ClickStreamData.ClickStream('07')

#ClickStreamData.ClickStream('08')
#TPData.TP()
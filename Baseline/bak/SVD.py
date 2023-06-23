from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
import os
import math
import numpy as np
import pandas as pd
from surprise.model_selection import train_test_split
import time
from surprise.model_selection import PredefinedKFold
# data = Dataset.load_builtin('ml-100k')
# # 类似于sklearn中的写法，将数据分割为75%
# trainset, testset = train_test_split(data, test_size=.25)
# algo = SVD()
# # 不同上一个例子，这里使用fit和test函数
# algo.fit(trainset)
# predictions = algo.test(testset)
# print (predictions)
# # 选用rmse指标
# accuracy.rmse(predictions)

Dir=os.path.abspath('')
'''
def RC(month):

    #reader = Reader(line_format='user item rating', sep=',')
    #data = Dataset.load_from_file(Dir+'\\UserMap-ItemMap-Clicks-Label08.csv', reader=reader)
    #print (data)

    #header = ['user_id', 'item_id', 'rating', 'label']
    #data = pd.read_csv(Dir+'\\UserMap-ItemMap-Clicks-Label'+month+'.csv', header=0,names=header)
    #print (data)
    #data = data.build_full_trainset()

    f = open(Dir+'\\test.txt', 'r')
    lines = f.readlines()
    UserIDlist=[]  #
    Labellist=[]
    for line in lines:
        line=line.strip('\n')
        #print(line)
        list=line.split('\t')
        UserID=list[0]
        Label=list[3]
        UserIDlist.append(UserID)
        Labellist.append(Label)
    reader = Reader(line_format='user item rating', sep='\t')
    # 指定要读入的数据文件，本例中为test.txt
    data = Dataset.load_from_file(Dir+'\\test.txt', reader=reader)
    #print ('data:',data)
    trainset, testset = train_test_split(data, test_size=.2)
    #print ('UserID:',testset[0][0],'Label:',Labellist[UserIDlist.index(testset[0][0])])  #用户ID-Label
    #time.sleep(100)hh
    #data = data.build_full_trainset()
    #print(trainset)
    #print(np.array(trainset).shape)
    # Dis_train_ItemID=set(np.array(trainset)[:, 1].tolist())
    # print ('训练集合中不同Item个数:',len(Dis_train_ItemID)) #=fenmu
    # Dis_train_UserID=set(np.array(trainset)[:, 0].tolist())
    # print ('训练集合中不同User个数:',len(Dis_train_UserID)) #=fenmu
    algo = SVD()
    algo.fit(trainset)
    precision = 0.0
    recall = 0.0
    map = 0.0
    ndcg = 0.0
    topk = 50

    Dis_test_ItemID=set(np.array(testset)[:, 1].tolist()) #待推荐的用户
    print('测试集合中不同ItemID个数:', len(Dis_test_ItemID))
    Dis_test_UserID=set(np.array(testset)[:, 0].tolist()) #待推荐的用户
    print('测试集合中不同User个数:', len(Dis_test_UserID))
    fenmu = pd.DataFrame(np.array(testset)[:, 0]).drop_duplicates().shape[0]  #UserID数量
    #print ('测试集合中不同User个数:',fenmu)

    real = [[] for i in range(fenmu)]  #groundtruth
    sor = [[] for i in range(fenmu)]   #推荐的
    hit = 0
    score = 0.0
    dcg = 0.0
    dic = {}
    m = 0
    kflod=1
    num_item=25913
    #num_item=35913
    for UserID in Dis_test_UserID:
        print('UserID:', UserID)
        print ('Label:',Labellist[UserIDlist.index(UserID)])
        uid = str(UserID)
        #iid = str(Labellist[UserIDlist.index(UserID)])
        for j in range(num_item):
            iid = str(j)
            pred = algo.predict(uid, iid)
            print ('pred:',pred)
        time.sleep(100)
    print('Yes!!')

    for i in range(len(testset)):
        if int(testset[i][0]) not in dic:
            dic[int(testset[i][0])] = m
            m += 1
            ls = []
            real[m - 1].append(int(testset[i][1]))
            for j in range(num_item):
                uid = str(testset[i][0])
                iid = str(j)
                pred = algo.predict(uid, iid)
                print('uid:',uid,'iid:',iid,'pred:',pred)
                ls.append([pred[3], j])
            ls = sorted(ls, key=lambda x: x[0], reverse=True)
            for s in range(topk):
                sor[m - 1].append(int(ls[s][1]))
        else:
            real[dic[int(testset[i][0])]].append(int(testset[i][1]))
    for i in range(fenmu):
        idcg = 0.0
        ap_score = 0.0
        ap = 0.0
        cg = 0.0
        for y in range(topk):
            if sor[i][y] in real[i]:
                ap_score += 1
                ap += ap_score / (y + 1)
                cg += 1 / math.log((y + 2), 2)
        score += ap / min(len(real[i]), topk)
        for z in range(int(ap_score)):
            idcg += 1 / math.log((z + 2), 2)
        if idcg > 0:
            dcg += cg / idcg
        recall += ap_score / (len(real[i]) * fenmu)
        precision += ap_score / (topk * fenmu)
    map += float(score) / fenmu
    ndcg += float(dcg) / fenmu


    print ('precision ' + str(precision /kflod))
    print('recall ' + str(recall / kflod))
    print('map ' + str(map / kflod))
    print ('ndcg ' + str(ndcg /kflod))
    #print (predictions)
    #accuracy.rmse(predictions)
RC('07')
'''



import pyodbc
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
import pandas as pd
import csv
import os
import heapq
import time
import datetime
import xlrd
import xlwt
import jieba
import json
import string
import gensim
from gensim import models
from gensim.models import word2vec
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
start = time.clock()
Dir=os.path.abspath('')
conn1 = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;PORT=1433;DATABASE=Tuniu Deep Learning', charset="utf8")
cur1 = conn1.cursor()

# 其中的test是这张表的名字,cell_overwrite_ok，表示是否可以覆盖单元格，其实是Worksheet实例化的一个参数，默认值是False

def TPandUserData():
    #####*****Travek_package******#######
    sql = 'SELECT [ItemID] ,[routeName],[destination],[destinationLarge],[type] FROM [Tuniu Deep Learning].[dbo].[ItemInformation 0708 fenci]'
    cur1.execute(sql)
    resList = cur1.fetchall()
    conn1.commit()
    Travel_Package_list = []
    book_tp = xlwt.Workbook(encoding="utf-8", style_compression=0)
    # 创建一个sheet对象，一个sheet对象对应Excel文件中的一张表格。
    sheet = book_tp.add_sheet('Travel_Package', cell_overwrite_ok=True)
    i=0
    for data in resList:
        TP_Infor = []
        ItemID=data[0]
        TP_Infor.append(ItemID)
        #print(ItemID)
        Title=str(data[1]).replace('\\','')  #海南\ 三亚\ 5日\ 自助游\ 2晚\ 美高梅加\ 2晚\ 银泰\ 接机
        #print (Title)
        TP_Infor.append(Title)  #所有产品的Title
        #l=len(Title.split('\ '))
        #TP_Infor.append(int(l))
        Destination=data[2]
        #print(Destination)
        TP_Infor.append(Destination)
        #print (Destination.decode('utf8'))
        Travel_Region=data[3]
        #print(Travel_Region)
        TP_Infor.append(Travel_Region)
        Travel_Type=data[4]
        TP_Infor.append(Travel_Type)
        Travel_Package_list.append(TP_Infor)
        #sheet.write(ItemID,Title,Destination,Travel_Region,Travel_Type)
        sheet.write(i,0,ItemID)
        sheet.write(i, 1, Title)
        sheet.write(i, 2, Destination)
        sheet.write(i, 3, Travel_Region)
        sheet.write(i, 4, Travel_Type)
        i=i+1
    #print (np.array(Travel_Package_list)[:,0])
    Travel_Package_list=np.array(Travel_Package_list)
    print (Travel_Package_list)

    ########*******UserUD*******##########
    sheet = book_tp.add_sheet('UserID', cell_overwrite_ok=True)
    cur2 = conn1.cursor()
    sql='SELECT [CookieID]  FROM [Tuniu Deep Learning].[dbo].[Distinct cookie Final]'
    cur2.execute(sql)
    UserID_list = cur2.fetchall()
    User_ID_list=[UserID[0] for UserID in UserID_list]
    for i in range(0,len(User_ID_list)):
        sheet.write(i,0,User_ID_list[i])
    conn1.commit()
    User_ID_list=np.array(User_ID_list)
    print (User_ID_list)
    book_tp.save(Dir+'\\data\\Travel Products Infor.xls')

def ClickandPurchaseData(month):
    cur3 = conn1.cursor()
    book_clickstream = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet =book_clickstream.add_sheet('ClickStream', cell_overwrite_ok=True)
    sql="SELECT [CookieID],[tourID],[time] FROM [Tuniu Deep Learning].[dbo].[Trans"+month+" Final]"
    cur3.execute(sql)
    Behavior_list = cur3.fetchall()
    i=0
    for data in Behavior_list:
        #print (data)
        UserID=data[0]
        OrderItemID=data[1]
        OrderTime=data[2]
        sql1 = "SELECT [CookieID] ,[ItemID],[operate_time],[current_session_time] FROM [Tuniu Deep Learning].[dbo].[Order webflow_today_detail2013"+month+" type=1 Final] where [CookieID]="+str(UserID)+"and [operate_time]<'"+OrderTime+"' order by [operate_time]"
        #print (sql1)
        cur3.execute(sql1)
        Behavior_list_UserID = cur3.fetchall()
        #print (Behavior_list_UserID)
        list1=[]
        list2=[]
        list3=[]
        for data in Behavior_list_UserID:
            ItemID=str(data[1])
            list1.append(ItemID)
            operate_time=data[2]
            list2.append(operate_time)
            session_time=data[3]
            list3.append(session_time)
        #if list1!=[]:
        if len(list1)>=2:
            List1=str("|".join(list1))
            List2 = str("|".join(list2))
            List3 = str("|".join(list3))
            sheet.write(i,0,UserID)#UserID
            sheet.write(i,1,List1)#ClickItemID
            sheet.write(i, 2, List2)#operate_time
            sheet.write(i, 3, List3) #session_time
            sheet.write(i, 4, str(OrderItemID)) #OrderItemID
            sheet.write(i, 5, str(OrderTime)) #OrderTime
            i=i+1
    book_clickstream.save(Dir+'\\data\\ClickStream'+month+'.xls')

ClickandPurchaseData('07')
ClickandPurchaseData('08')

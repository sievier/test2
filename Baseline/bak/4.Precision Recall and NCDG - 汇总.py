from math import log
import linecache
import datetime
import os
starttime = datetime.datetime.now()
def Evaluation(date,K,support):
    print("DATE:"+str(date)+'\t'+"Top-K:"+str(K)+'\t'+"support:"+str(support))
    print("**********Result*************")
   
    Dir=os.path.abspath('')   #获得当前的绝对路径并赋给Dir
    #Dir1=Dir+'/s-15/'
    Dir1=Dir+'/RC Result8月份汇总'
    Dir2=Dir1+'/'+date #日期目录  #******************#
    Dir3=Dir+'/8月份测试集/'+date+'/' #测试集目录  #******************#
    file2 = '/sequence-UserID-RouteIDLabel YongGe-GeneralizationItem2.tab' #当前的sequence  #******************#

    fin1 = open( Dir3+file2,'r')  #当前的sequence
    content1=fin1.readlines()
    file_result3 = open( Dir2+'/RC-user_routeID Top'+str(K)+'-Generalization非实时s='+str(support)+'.txt','r')  #推荐结果
    content3=file_result3.readlines()
    file_result4 = open( Dir2+'/评价指标 Top'+str(K)+' support='+str(support)+'.txt','w')
    file_result5 = open( Dir2+'/RightRCTop'+str(K)+' support='+str(support)+'.txt','w')
    RightRC=0
    #print("S:"+str(S))
    #DCG2=[]
    IDCG=0
    DCG1=0
    RightCookieID=[]
    TransCookieID=[]
    RCCookieID=[]
    AllRcItem=[]
    RCTotal=len(content3) #产生推荐总数
    SessionTotal=len(content1) #当天的会话次数
    for line1 in content1:  #Session
        line1=line1.strip('\n')
        line1=line1.split('->')
        line1=line1[0]
        line1=line1.split(' (')
        line1=line1[1]
        CookieID=line1.strip(')')
        #print("line1:"+str(CookieID))
        TransCookieID.append(CookieID)
    #print("RCTotal:"+str(RCTotal))
    
    R=[] #推荐的items集合
    SR=[] #推荐准确的Items集合
    sumRC=0
    DCG2=[]
    RightRC=0
    for line1 in content3:    #推荐结果
        RCItemList=[]
        line1=line1.strip('\n')
        #print(line1)
        line1new=line1.split(' Label:')
        CookieIDRC=line1new[0]
        CookieIDRC=CookieIDRC.split("->")
        CookieID=CookieIDRC[0]
        RCCookieID.append(CookieID)
        RCItems=CookieIDRC[1]
        Label=line1new[1]
        #print("Label:"+str(Label)) 
        #print("RCItems:"+str(RCItems))
        if RCItems != "[]":   
            RCItems=RCItems.strip('[')
            RCItems=RCItems.strip(']')
            RCItems=RCItems.strip('\'')
            RCItems=RCItems.split(',')
            r=len(RCItems)
            #print("r:"+str(r))
            R.append(r)
            for RCItem in RCItems:
                RCItem=RCItem.strip('\'')
                RCItem=RCItem.strip(' \'')
                #print(RCItem)
                AllRcItem.append(RCItem)  #存储所有的推荐item
                RCItemList.append(RCItem)  #推荐结果序列
            sumRC+=1
            #print("RCItemList:"+str(RCItemList))
            if Label in RCItemList:             
                #Rlength+=1
                RightCookieID.append(CookieID)
                SR.append(Label)
                file_result5.write(str(RCItems)+'\t'+str(Label)+'\n') 
                Position=[i for i,v in enumerate(RCItemList) if v==Label]
                nq=len(Position) #该会话推荐了几个Items
                SR.append(nq) #推荐准确的items
                DCG1=0
                for i in range(0,nq):
                    #print("Position[i]:"+str(Position[i]+1))
                    #DCG_T=1/max(1,log(Position[i]+1,2))
                    DCG_T=1/(log((Position[i]+1)+1,2))
                    DCG1=DCG1+DCG_T
                DCG2.append(DCG1)       
                RightRC+=1
                #print("DCG1:"+str(DCG1))
    #print("len(DCG):"+str(len(DCG)))

    


    DisTransCookieID=list(set(TransCookieID))
    DisRightCookieID=list(set(RightCookieID))
    DisRCCookieID=list(set(RCCookieID))
    DisSR=list(set(SR))

    
    #print("准确推荐的不同的项目："+str(len(DisSR))+str(DisSR))
    file_result4.write("准确推荐的不同的项目："+str(len(DisSR))+str(DisSR)+'\n')
    '''
    #Precision分母为会话次数#
    Precision_new=float(RightRC)/float(sumRC)
    print("Precision_new:"+str(Precision_new))
    '''
    #Precision分母为user#
    print("交易的总数："+str(SessionTotal))
    file_result4.write("交易的总数："+str(SessionTotal)+'\n')
    print("产生的推荐总数："+str(RCTotal))
    file_result4.write("产生的推荐总数："+str(RCTotal)+'\n')
    #print("准确的推荐用户总数："+str(len(DisRightCookieID))) 
    #file_result4.write("准确的推荐用户总数："+str(len(DisRightCookieID))+'\n')
    print("准确的推荐总数："+str(RightRC))
    file_result4.write("准确推荐项目总数:"+str(RightRC)+'\n')   
    Precision_new=float(RightRC)/float(RCTotal)
    Recall=float(RightRC)/float(SessionTotal)
    print("Precision@k:"+str(Precision_new))
    print("Recall@k:"+str(Recall))
    file_result4.write("Precision@k:"+str(Precision_new)+'\n')
    file_result4.write("Recall@k:"+str(Recall)+'\n')
    F_1Score=2*float(Precision_new)*float(Recall)/(float(Precision_new)+float(Recall))
    print("F_1Score:"+str(F_1Score))
    IDCG=1/(log(1+1,2))
    #print("DCG1:"+str(DCG1))
    DCG=sum(DCG2)/len(DisRCCookieID)  #推荐总数
    NDCG=DCG/IDCG
    print("NDCG@K:"+str(NDCG))
    #print("DCG1:"+str(DCG1))
    DCG=sum(DCG2)/len(RightCookieID)  #推荐总数
    NDCG=DCG/IDCG
    print("NDCG@Right:"+str(NDCG))


    file_result4.write("NDCG-RIGHTRC:"+str(NDCG)+'\n')
    DisRCItem=list(set(AllRcItem))
    TotalItem=28038
    print("len(DisRCItem):"+str(len(DisRCItem)))
    Coverage=float(len(DisRCItem))/float(TotalItem)
    print("Coverage:"+str(Coverage))
    file_result4.write("Coverage:"+str(Coverage)+'\n')
    '''
    #NDCG方案2（适用于繁华后推荐）
    for i in range(0,k):
        IDCG_T=1/(log(i+1+1,2))
        IDCG=IDCG_T+IDCG
    NDCG=DCG/IDCG
    '''
    #print("NDCG:"+str(NDCG))
    
    file_result3.close()
    fin1.close()
    file_result4.close()
    file_result5.close()



'''
Evaluation('0811-0820',3,'2')    # (date,top k,support)
print('\n')
Evaluation('0811-0820',5,'2')
print('\n')
Evaluation('0811-0820',10,'2')
print('\n')
Evaluation('0811-0820',20,'2')

Evaluation('0821-0831',3,'2')    # (date,top k,support)
print('\n')
Evaluation('0821-0831',5,'2')
print('\n')
Evaluation('0821-0831',10,'2')
print('\n')
Evaluation('0821-0831',20,'2')
'''
'''
Evaluation('0801-0810',3,'4')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'4')
print('\n')
Evaluation('0801-0810',10,'4')
print('\n')
Evaluation('0801-0810',20,'4')

Evaluation('0811-0820',3,'4')    # (date,top k,support)
print('\n')
Evaluation('0811-0820',5,'4')
print('\n')
Evaluation('0811-0820',10,'4')
print('\n')
Evaluation('0811-0820',20,'4')

Evaluation('0821-0831',3,'4')    # (date,top k,support)
print('\n')
Evaluation('0821-0831',5,'4')
print('\n')
Evaluation('0821-0831',10,'4')
print('\n')
Evaluation('0821-0831',20,'4')

Evaluation('0801-0810',3,'6')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'6')
print('\n')
Evaluation('0801-0810',10,'6')
print('\n')
Evaluation('0801-0810',20,'6')

Evaluation('0811-0820',3,'6')    # (date,top k,support)
print('\n')
Evaluation('0811-0820',5,'6')
print('\n')
Evaluation('0811-0820',10,'6')
print('\n')
Evaluation('0811-0820',20,'6')

Evaluation('0821-0831',3,'6')    # (date,top k,support)
print('\n')
Evaluation('0821-0831',5,'6')
print('\n')
Evaluation('0821-0831',10,'6')
print('\n')
Evaluation('0821-0831',20,'6')

'''

'''
Evaluation('0801-0810',3,'8')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'8')
print('\n')
Evaluation('0801-0810',10,'8')
print('\n')
Evaluation('0801-0810',20,'8')

Evaluation('0811-0820',3,'8')    # (date,top k,support)
print('\n')
Evaluation('0811-0820',5,'8')
print('\n')
Evaluation('0811-0820',10,'8')
print('\n')
Evaluation('0811-0820',20,'8')

Evaluation('0821-0831',3,'8')    # (date,top k,support)
print('\n')
Evaluation('0821-0831',5,'8')
print('\n')
Evaluation('0821-0831',10,'8')
print('\n')
Evaluation('0821-0831',20,'8')

'''
'''
Evaluation('0801-0810',3,'10')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'10')
print('\n')
Evaluation('0801-0810',10,'10')
print('\n')
Evaluation('0801-0810',20,'10')

Evaluation('0811-0820',3,'10')    # (date,top k,support)
print('\n')
Evaluation('0811-0820',5,'10')
print('\n')
Evaluation('0811-0820',10,'10')
print('\n')
Evaluation('0811-0820',20,'10')

Evaluation('0821-0831',3,'10')    # (date,top k,support)
print('\n')
Evaluation('0821-0831',5,'10')
print('\n')
Evaluation('0821-0831',10,'10')
print('\n')
Evaluation('0821-0831',20,'10')
'''
'''
Evaluation('0801-0810',3,'12')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'12')
print('\n')
Evaluation('0801-0810',10,'12')
print('\n')
Evaluation('0801-0810',20,'12')

Evaluation('0811-0820',3,'12')    # (date,top k,support)
print('\n')
Evaluation('0811-0820',5,'12')
print('\n')
Evaluation('0811-0820',10,'12')
print('\n')
Evaluation('0811-0820',20,'12')

Evaluation('0821-0831',3,'12')    # (date,top k,support)
print('\n')
Evaluation('0821-0831',5,'12')
print('\n')
Evaluation('0821-0831',10,'12')
print('\n')
Evaluation('0821-0831',20,'12')

Evaluation('0801-0810',3,'50')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'50')
print('\n')
Evaluation('0801-0810',10,'50')
print('\n')
Evaluation('0801-0810',20,'50')
'''
'''
Evaluation('0801',3,'2')    # (date,top k,support)
print('\n')
Evaluation('0801',5,'2')
print('\n')
Evaluation('0801',10,'2')
print('\n')
Evaluation('0801',20,'2')

Evaluation('0801-0810',3,'30')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'30')
print('\n')
Evaluation('0801-0810',10,'30')
print('\n')
Evaluation('0801-0810',20,'30')
print('\n')
print('\n')
print('\n')
Evaluation('0801-0810',3,'40')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'40')
print('\n')
Evaluation('0801-0810',10,'40')
print('\n')
Evaluation('0801-0810',20,'40')
print('\n')
print('\n')
print('\n')

Evaluation('0801-0810',3,'20')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'20')
print('\n')
Evaluation('0801-0810',10,'20')
print('\n')
Evaluation('0801-0810',20,'20')

Evaluation('0801-0810',3,'15')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'15')
print('\n')
Evaluation('0801-0810',10,'15')
print('\n')
Evaluation('0801-0810',20,'15')
Evaluation('0801-0810',3,'25')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'25')
print('\n')
Evaluation('0801-0810',10,'25')
print('\n')
Evaluation('0801-0810',20,'25')
Evaluation('0801-0810',3,'35')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'35')
print('\n')
Evaluation('0801-0810',10,'35')
print('\n')
Evaluation('0801-0810',20,'35')
'''
'''
Evaluation('0801-0810',3,'2')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'2')
print('\n')
Evaluation('0801-0810',10,'2')
print('\n')
Evaluation('0801-0810',20,'2')

Evaluation('0801-0810',3,'5')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'5')
print('\n')
Evaluation('0801-0810',10,'5')
print('\n')
Evaluation('0801-0810',20,'5')

Evaluation('0801-0810',3,'10')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'10')
print('\n')
Evaluation('0801-0810',10,'10')
print('\n')
Evaluation('0801-0810',20,'10')

Evaluation('0801-0810',3,'15')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'15')
print('\n')
Evaluation('0801-0810',10,'15')
print('\n')
Evaluation('0801-0810',20,'15')

Evaluation('0801-0810',3,'20')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'20')
print('\n')
Evaluation('0801-0810',10,'20')
print('\n')
Evaluation('0801-0810',20,'20')

Evaluation('0801-0810',3,'25')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'25')
print('\n')
Evaluation('0801-0810',10,'25')
print('\n')
Evaluation('0801-0810',20,'25')

Evaluation('0801-0810',3,'30')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'30')
print('\n')
Evaluation('0801-0810',10,'30')
print('\n')
Evaluation('0801-0810',20,'30')

Evaluation('0801-0810',3,'35')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'35')
print('\n')
Evaluation('0801-0810',10,'35')
print('\n')
Evaluation('0801-0810',20,'35')

Evaluation('0801-0810',3,'40')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'40')
print('\n')
Evaluation('0801-0810',10,'40')
print('\n')
Evaluation('0801-0810',20,'40')
'''

Evaluation('0801-0810',3,'20')    # (date,top k,support)
print('\n')
Evaluation('0801-0810',5,'20')
print('\n')
Evaluation('0801-0810',10,'20')
print('\n')
Evaluation('0801-0810',20,'20')

##Evaluation('0811-0820',3,'20')    # (date,top k,support)
##print('\n')
##Evaluation('0811-0820',5,'20')
##print('\n')
##Evaluation('0811-0820',10,'20')
##print('\n')
##Evaluation('0811-0820',20,'20')

Evaluation('0821-0831',3,'20')    # (date,top k,support)
print('\n')
Evaluation('0821-0831',5,'20')
print('\n')
Evaluation('0821-0831',10,'20')
print('\n')
Evaluation('0821-0831',20,'20')

endtime = datetime.datetime.now()
print ('运行时间'+str(((endtime - starttime).seconds)/60)+'分')

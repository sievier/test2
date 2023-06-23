#from layer import encodertile
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Data
import DataLoad
#from DataLoad import TPData,UserData,ClickStreamData,Word2Vector
#TPData.TP()
result=DataLoad.TPData.TP
ItemIDList=result[0]
Title_embedding_list=result[1]  #长度一致补齐
DestinationList=result[2]
Travel_RegionList=result[3]
Travel_TypeList=result[4]
Title_embedding_size=result[5]
Title_maxlen = result[6]
Title_embedding_list_pad = pad_sequence([torch.from_numpy(np.array(x)) for x in Title_embedding_list], batch_first=True).float()  #长度补齐到maxlen
#print (Title_embedding_list_pad.shape)
vocab_size_title=result[7]
vocab_size_Destination=result[8]
vocab_size_Travel_Region=result[9]
vocab_size_Travel_Type=result[10]
UserID_list=result[11]
vocab_size_UserID=result[12]
# def Iput():
#print('word2vector的词量:',len(model.wv.vocab))

#训练集合batchsize
ClickStream=DataLoad.ClickStreamData.ClickStream('07')
Training_Clickdata=ClickStream[0]  #用户Encoder中的训练集,[UserID_i,ClickItemID_i_list_Long,ClickItemID_i_list_Short,OrderItemID_i]
#print (Training_Clickdata.shape)


UserID_train=Training_Clickdata[:,0]
ItemID_long_train=Training_Clickdata[:,1]
ItemID_short_train=Training_Clickdata[:,2]
OrderItemID_train=Training_Clickdata[:,3]
#print(OrderItemID_train)
#print(Training_Clickdata[:,0:3])

BATCH_SIZE = 5      # 批训练的数据个数
#print(Training_Clickdata[1,0:3])
#print(OrderItemID_train[1])
#x=torch.from_numpy(Training_Clickdata[:,0:0])
#y=torch.from_numpy(OrderItemID_train)
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())


Long_maxlen=ClickStream[2] #Long-Term最长距离
Short_maxlen=ClickStream[3]#Short-Term最长距离
UserIDlist=ClickStream[4]

# Len_long_train=[len(x)for x in ItemID_long_train]
# Len_short_train=[len(x)for x in ItemID_short_train]
#print (np.mean(Len_long_train),np.mean(Len_short_train))
#print (len(list(set(UserID_Ttrain))),Training_Clickdata.shape)#用户规模

Test_Clickdata=ClickStream[1]#用户Encoder中的测试集
UserID_Test=Test_Clickdata[:,0]
ItemID_long_test=Test_Clickdata[:,1]
ItemID_short_test=Test_Clickdata[:,2]

#print (len(list(set(UserID_Test))),Test_Clickdata.shape) #用户规模
# Len_long_test=[len(x)for x in ItemID_long_test]
# Len_short_test=[len(x)for x in ItemID_short_test]
#print (np.mean(Len_long_test),np.mean(Len_short_test))

#print (UserIDlist)
OrderItemIDList=ClickStream[5]
#print(OrderItemIDList)
#####*****TP Encoder# ***####

#long-term的序列补齐

#Short-term的序列补齐


class conf ():
    def __init__(self):

        '''
        这里的参数都是要自己调的
        '''
        self.vocab1 = 10 #user
        self.vocab2 = 3  #Title
        self.vocab3 = 30 #Destination
        self.vocab4 =20  #Travel Region (TR)
        self.vocab5 = 5   #Travel Type (TT)
        self.embedding_size1 = 300
        self.embedding_size2 = 300
        self.embedding_size3 = 300
        self.embedding_size4 = 300
        self.embedding_size5 = 300

        self.hidden1 = 128
        self.hidden2 = 128
        self.hidden3=128
        self.num_steps =10

cfg=conf()
#print(cfg.vocab1)
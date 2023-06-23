from DataLoad import ClickStreamData
from model import User_Candidate_Encoder
from layer import TPEncoder
from config import conf
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
from DataLoad import TPData
import time
import torch.optim as optim
import math
import os
import shutil
from tensorboardX import SummaryWriter
from random import sample
from constrain import negtive_log
import random
from torchsummary import summary
import xlwt
cfg=conf()
losslog=negtive_log()
#print (torch.cuda.is_available())#GPU兼容问题
#print(torch.cuda.device_count())
device ='cpu'    #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
Dir=os.path.abspath('')
CUR_PATH = Dir+'/runs/'
shutil.rmtree(CUR_PATH)
writer=SummaryWriter()
model = User_Candidate_Encoder(cfg)
yu1=torch.ones(1).detach()
yu2=torch.ones(size=(1,40,23)).detach()
yu3=torch.ones(size=(1,40)).detach()
yu4=torch.ones(size=(1,40)).detach()
yu5=torch.ones(size=(1,40)).detach()
ye1=yu1
ye2=torch.ones(size=(1,40,23)).detach()
ye2=torch.ones(size=(1,40)).detach()
ye3=torch.ones(size=(1,40)).detach()
ye4=torch.ones(size=(1,40)).detach()
ye5=torch.ones(size=(1,40)).detach()

zy1=torch.ones(size=(1,1)).detach()
zy2=torch.ones(size=(1,23)).detach()
zy3=torch.ones(size=(1,1)).detach()
zy4=torch.ones(size=(1,1)).detach()
zy5=torch.ones(size=(1,1)).detach()
#with writer:
#writer.add_graph(model,input_to_model=(yu1,yu2,yu3,yu4,yu5,ye1,ye2,ye3,ye4,ye5,zy1,zy2,zy3,zy4,zy5))
#print(summary(model),input_size=(channels, H, W))
optimizer=optim.Adam(model.parameters(),lr=1e-4)  #lr=1e-4,学习率

def negativesamplingfunc(batchsize,batch_userbehaviorencoder,batch_Positive_Encoder,batch_Negative_Encoder):  #targetlist：16*1，16*L-1 predictlist

    #print('batch_Positive_Encoder_Shape:', batch_Positive_Encoder.size())
    #print('batch_Negative_Encoder shape:', batch_Negative_Encoder.size())
    batch_userbehaviorencoder=torch.unsqueeze(batch_userbehaviorencoder,dim=1)
    #print('batch_userbehaviorencoder_shape:', batch_userbehaviorencoder.size())
    batch_Positive_score=torch.sigmoid(torch.sum(torch.mul(batch_userbehaviorencoder,batch_Positive_Encoder),dim=-1))#torch.Size([2, 1])
    #print('batch_Positive_score:',batch_Positive_score)
    #print('batch_Positive_score_shape:',batch_Positive_score.size())

    batch_Negative_Score=torch.sigmoid(torch.sum(torch.mul(batch_userbehaviorencoder,batch_Negative_Encoder),dim=-1))  #torch.Size([2, 4])
    #print('batch_Negative_Score:',batch_Negative_Score)
    #print('batch_Negative_Score_shape:',batch_Negative_Score.size())
    batch_z=torch.cat((batch_Positive_score,batch_Negative_Score),dim=1)#推荐分值 z.size: torch.Size([2, 5])
    #print('z.size:',batch_z.size())
    batch_y_hat=torch.softmax(batch_z,dim=1)
    loss_value=torch.mean(-torch.log(batch_y_hat[:,0]),dim=0)
    #writer.add_scalar(tag='loss',scalar_value=loss_value.,global_step=)
    print ('loss_value:',loss_value)
    return loss_value.requires_grad_()

#####**TP信息**###
TPInfor = TPData.TP
ItemIDList = TPInfor[0]  # ItemID索引
# print ('ItemIDList:',ItemIDList)
TitleWordIDlist = TPInfor[1]
# print('TitleWordIDlist:',TitleWordIDlist.shape,TitleWordIDlist[0])
DestinationIDList = TPInfor[2]
# print('DestinationIDList:',DestinationIDList)
Travel_RegionIDList = TPInfor[3]
# print(np.max(Travel_RegionIDList))
# print('Travel_RegionIDList:',Travel_RegionIDList)
Travel_TypeIDList = TPInfor[4]

def train(month,model,optimizer):
    #####**User信息**###
    #UserID_tensor=UserData.UserInput()
    #print(UserID_tensor.size())
    losstype = 'negativesampling'
    #losstype = 'lognegatives'
    K=20 #num of neagtive sampling
    #print (np.max(Travel_TypeIDList))
    #print('Travel_TypeIDList:',Travel_TypeIDList)
    BATCH_SIZE = cfg.batch_size  # 批训练的数据个数
    Max_SessionLength=cfg.Long_N
    result=ClickStreamData.ClickStream(month)
    Training_Clickdata=np.array(result[0]) #训练集
    #print (Training_Clickdata)
    ID_traing = np.arange(Training_Clickdata.shape[0])
    UserID_train = Training_Clickdata[:, 0]    #UserID
    ItemID_long_train = Training_Clickdata[:, 1]  #long-termd点击流
    ItemID_short_train = Training_Clickdata[:, 2]  #short-term点击流
    Target_traing=Training_Clickdata[:,3]  #下单的ItemID
    #print (Targe_traing)
    x=torch.from_numpy(ID_traing)
    y=torch.Tensor([int(x) for x in Target_traing]) # 转换为list
    tensor_Train = Data.TensorDataset(x,y)   # 将数据封装成Dataset
    # 可使用索引调用数据
    # print ('tensor_data[0]: ', tensor_dataset[0])
    tensor_dataloader = Data.DataLoader(tensor_Train ,  # 封装的对象
                                        batch_size=BATCH_SIZE,  # 输出的batchsize
                                        shuffle=True,  # 随机输出
                                        num_workers=0)  # 只有1个进程
    candidate_title = []
    candidate_destination = []
    candidate_Travel_Type = []
    candidate_Travel_RegionID = []
    CandicateItemIDList=result[3]  #训练集合中的候选项集
    for i in range(0, len(CandicateItemIDList)):  #计算一遍候选项集(所有的产品)
        ItemID = CandicateItemIDList[i]
        # print('推荐候选ItemID：',ItemID)
        #print ('TitleWordIDlist[i]:',TitleWordIDlist[i])
        candidate_title.append(TitleWordIDlist[i])
        candidate_destination.append([int(DestinationIDList[i])])
        candidate_Travel_Type.append([int(Travel_TypeIDList[i])])
        candidate_Travel_RegionID.append([int(Travel_RegionIDList[i])])
    starttime = time.time()
    for epoch_i in range(cfg.epoch):
        train_loss_list = []
        model=model.train()
        model = model.to(device)
        for step, (batch_data, batch_target) in enumerate(tensor_dataloader):#一个batch（64个会话）
            #batch_data=batch_data.to(device)
            print('******One Batch****')
            #print(data, target)
            #TPEnocderone = TPEncoder(cfg)
            print('Epoch:', epoch_i, '|Step:', step, '|batch_data_user:',[UserID_train[int(ID)] for ID in batch_data], '|batch_target', batch_target)
            batch_userID=[]
            batch_long_term_TitleID=[]  #batch_long_term_embedding的输入
            batch_long_term_DestinationID=[]
            batch_long_term_TravelRegionID=[]
            batch_long_term_TravelTypeID=[]
            batch_long_termID=[]
            batch_short_term_TitleID=[]  #batch_short_term_embedding的输入
            batch_short_term_DestinationID = []
            batch_short_term_TravelRegionID = []
            batch_short_term_TravelTypeID = []
            batch_short_termID=[]
            for ID in batch_data:  #该batch中的一个用户（会话）
                #print('ID:',int(ID))
                #print('*****UserID******:',UserID_train[int(ID)])
                batch_userID.append(UserID_train[int(ID)])  #batch——UserID的输入
                print('Long-term Behaviors:', ItemID_long_train[int(ID)]) #Long-term点击流
                Long_term_TitleID=[]
                Long_term_DestinationID=[]
                Long_term_TravelRegionID=[]
                Long_term_TravelTypeID=[]
                Long_term_ItemID=[]
                if ItemID_long_train[int(ID)]!=[]:
                    for ItemID in ItemID_long_train[int(ID)]:  #一个产品
                        Long_term_ItemID.append(ItemID)
                        #print('ItemID:',ItemID)
                        Index_ID=np.where(ItemIDList == ItemID)
                        WordIDList=TitleWordIDlist[Index_ID]
                        #print(WordIDList)
                        #print ('Title_WordID:',TitleWordIDlist[Index_ID])
                        Long_term_TitleID.append(torch.Tensor(WordIDList[0]))
                        #print(DestinationIDList[Index_ID],Travel_RegionIDList[Index_ID],Travel_TypeIDList[Index_ID])
                        Long_term_DestinationID.append(torch.Tensor(DestinationIDList[Index_ID]))
                        Long_term_TravelRegionID.append(torch.Tensor(Travel_RegionIDList[Index_ID]))
                        Long_term_TravelTypeID.append(torch.Tensor(Travel_TypeIDList[Index_ID]))
                    #print('Long_term_TitleID:', Long_term_TitleID)

                #print ('Long_term_TitleID:',Long_term_TitleID)
                else:  #不存在长期行为，全部补0
                    Long_term_TitleID.append(torch.Tensor(np.zeros(23)))
                    Long_term_DestinationID.append(torch.Tensor(np.zeros(1)))
                    Long_term_TravelRegionID.append(torch.Tensor(np.zeros(1)))
                    Long_term_TravelTypeID.append(torch.Tensor(np.zeros(1)))
                Long_term_TitleID = pad_sequence(Long_term_TitleID, batch_first=True)#补齐
                print('Long_term_TitleID.size：', Long_term_TitleID.size())
                print('Short-term Behaviors:', ItemID_short_train[int(ID)])  #short-term点击流
                Short_term_TitleID = []
                Short_term_DestinationID=[]
                Short_term_TravelRegionID=[]
                Short_term_TravelTypeID=[]
                Short_term_ItemID=[]
                for ItemID in ItemID_short_train[int(ID)]:
                    Short_term_ItemID.append(ItemID)
                    Index_ID = np.where(ItemIDList == ItemID)
                    #print ('ItemID',ItemID,'Index_ID',Index_ID)
                    WordIDList = TitleWordIDlist[Index_ID]
                    #print(WordIDList)
                    Short_term_TitleID.append(torch.Tensor(WordIDList[0]))
                    Short_term_DestinationID.append(torch.Tensor(DestinationIDList[Index_ID]))
                    Short_term_TravelRegionID.append(torch.Tensor(Travel_RegionIDList[Index_ID]))
                    Short_term_TravelTypeID.append(torch.Tensor(Travel_TypeIDList[Index_ID]))
                Short_term_TitleID = pad_sequence(Short_term_TitleID, batch_first=True)
                print('Short_term_TitleID Size:', Short_term_TitleID.size())
                #long_term的4个输入

                batch_long_term_TitleID.append(Long_term_TitleID)
                #print('Long_term_DestinationID:', Long_term_DestinationID)
                batch_long_term_DestinationID.append(torch.Tensor(Long_term_DestinationID))
                #print('Long_term_TravelRegionID:',Long_term_TravelRegionID)
                batch_long_term_TravelRegionID.append(torch.Tensor(Long_term_TravelRegionID))
                #print('Long_term_TravelTypeID:',Long_term_TravelTypeID)
                batch_long_term_TravelTypeID.append(torch.Tensor(Long_term_TravelTypeID))
                batch_long_termID.append(Long_term_ItemID)
                #short_term的4个输入
                batch_short_term_TitleID.append(Short_term_TitleID)
                batch_short_term_DestinationID.append(torch.Tensor(Short_term_DestinationID))
                batch_short_term_TravelRegionID.append(torch.Tensor(Short_term_TravelRegionID))
                batch_short_term_TravelTypeID.append(torch.Tensor(Short_term_TravelTypeID))
                batch_short_termID.append(Short_term_ItemID)

            #print('得出一个batch的输出：')
            batch_userID=np.array(batch_userID)
            x1=torch.from_numpy(batch_userID)

            #print('batch_userID:',x1.size())
            #print('batch_userID:',batch_userID.shape,batch_userID)  #UserID
            #print('batch_long_term_TitleID:',batch_long_term_TitleID)
            x2=pad_sequence(batch_long_term_TitleID, batch_first=True)    #x2,y2:Title;
            x2= torch._cast_Long(x2)
            print('batch_long_term_Title_embedding：',x2.size())
            x3=pad_sequence(batch_long_term_DestinationID, batch_first=True)  #x3,y3:Destination;
            x3 = torch._cast_Long(x3)
            #print('batch_long_term_DestinationID:',x3.size())
            x4=pad_sequence(batch_long_term_TravelTypeID, batch_first=True)  #x4,y4:TravelType;
            x4 = torch._cast_Long(x4)
            #print('batch_long_term_TravelRegionID:', x4.size())
            x5=pad_sequence(batch_long_term_TravelRegionID,batch_first=True) #x5,y5:TravelRegionID;
            x5 = torch._cast_Long(x5)
            #print('batch_long_term_TravelTypeID:', x5.size())

            y1=x1
            #print('batch_userID:', y1.size)
            y2=pad_sequence(batch_short_term_TitleID, batch_first=True)
            y2 = torch._cast_Long(y2)
            print('batch_short_term_Title_embedding：',y2.size())
            y3=pad_sequence(batch_short_term_DestinationID, batch_first=True)
            y3 = torch._cast_Long(y3)
            #print('batch_short_term_DestinationID:',y3.size())
            y4=pad_sequence(batch_short_term_TravelTypeID, batch_first=True)
            y4 = torch._cast_Long(y4)
            #print('batch_short_term_TravelRegionID:', y4.size())
            y5 = pad_sequence(batch_short_term_TravelRegionID, batch_first=True)
            y5 = torch._cast_Long(y5)
            #print('batch_short_term_TravelTypeID:', y5.size())
            #print('batch_target:',batch_target.shape,batch_target)  #Odered ItemOD
            print('END Data Loading：')
            ########******一个batch,Start to Training******####
            print("Start to UserEncoder:")
            #print(model)

            x1 = torch.unsqueeze(x1, dim=1)
            x1 = torch.unsqueeze(x1, dim=1)
            x1 = x1.expand(-1, x2.size(1), -1)
            x3 = torch.unsqueeze(x3, dim=-1)
            x4 = torch.unsqueeze(x4, dim=-1)
            x5 = torch.unsqueeze(x5, dim=-1)

            y1 = torch.unsqueeze(y1, dim=1)
            y1 = torch.unsqueeze(y1, dim=1)
            y1 = y1.expand(-1, y2.size(1), -1)
            y3 = torch.unsqueeze(y3, dim=-1)
            y4 = torch.unsqueeze(y4, dim=-1)
            y5 = torch.unsqueeze(y5, dim=-1)

            #print('Start negativesampling for candidateItem!')  # 消极样本采样
            p1=batch_target #user下单的ItemID
            #print('batch_positive_label',p1)
            p2=[] #positive_sample_title
            p3=[]#positive_destination
            p4=[]#positive_Travel_Type
            p5=[]#positive_Travel_RegionID
            n2=[]  #negative_sample_title
            n3=[]
            n4=[]
            n5=[]
            K=cfg.K_neagtive_sampling
            for itemID in p1:  #一个batch下的下单ItemID
                #print('itemID:',int(itemID))
                #对于Batch_user positive samples的旅游包的特征
                index_p=CandicateItemIDList.index(int(itemID))
                p2.append([candidate_title[index_p]])
                p3.append([candidate_destination[index_p]])
                p4.append([candidate_Travel_Type[index_p]])
                p5.append([candidate_Travel_RegionID[index_p]])
                # 对于Batch_user negative samples的旅游包的特征
                negative_CandicateItemIDList=list(set(CandicateItemIDList)-{int(itemID)})
                random.shuffle(negative_CandicateItemIDList)
                negative_CandicateItemIDList_sample=negative_CandicateItemIDList[:K]
                #print('negative_CandicateItemIDList_sample:',negative_CandicateItemIDList_sample)
                #print (len(negative_CandicateItemIDList_sample),len(CandicateItemIDList))
                candidate_title_negative=[]
                candidate_destination_negative=[]
                candidate_Travel_Type_negative=[]
                candidate_Travel_RegionID_negative=[]
                for negative_CandicateItemID in negative_CandicateItemIDList_sample:
                    index_n= CandicateItemIDList.index(int(negative_CandicateItemID))
                    candidate_title_negative.append(candidate_title[index_n])
                    candidate_destination_negative.append(candidate_destination[index_n])
                    candidate_Travel_Type_negative.append(candidate_Travel_Type[index_n])
                    candidate_Travel_RegionID_negative.append(candidate_Travel_RegionID[index_n])
                n2.append(candidate_title_negative)
                n3.append(candidate_destination_negative)
                n4.append(candidate_Travel_Type_negative)
                n5.append(candidate_Travel_RegionID_negative)
            p2=torch.tensor(p2)
            p3 = torch.tensor(p3)
            p4 = torch.tensor(p4)
            p5 = torch.tensor(p5)
            n2=torch.tensor(n2)
            n3 = torch.tensor(n3)
            n4 = torch.tensor(n4)
            n5 = torch.tensor(n5)
            s_title=torch.cat([p2,n2],dim=1)
            s_destination=torch.cat([p3,n3],dim=1)
            s_travel_type=torch.cat([p4,n4],dim=1)
            s_travel_region=torch.cat([p5,n5],dim=1)
            #print('p2,p3,p4,p5:',p2.size(),p3.size(),p4.size(),p5.size())
            #print('n2,n3,n4,n5:',n2.size(),n3.size(),n4.size(),n5.size())
            encoder=model(x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,s_title,s_destination,s_travel_type,s_travel_region)
            batch_userbehaviorencoder=torch.tensor(encoder[0]) #batch_user Encoder
            batch_sample_Encoder=encoder[1]  #
            #print('batch_sample_Encoder.size():',batch_sample_Encoder.size())
            batch_Positive_Encoder= batch_sample_Encoder[:,0,:]#Positive_Encoder 1个Encoder_value
            #print(batch_Positive_Encoder.size())
            batch_Negative_Encoder=batch_sample_Encoder[:,1:,:]   #K_neagtive_sampling=63 个Encoder_value
            #print(batch_Negative_Encoder.size())
            loss=negativesamplingfunc(cfg.batch_size,batch_userbehaviorencoder,batch_Positive_Encoder,batch_Negative_Encoder)  #计算loss
            #print ('*********************')
            train_loss_list.append(loss.item())
            optimizer.zero_grad()  ##将以前的方向传播
            loss.backward()
            optimizer.step()
            writer.add_scalar(tag='loss', scalar_value=loss ,global_step=(epoch_i)*step+step)
            for name in model.state_dict():
                writer.add_histogram(tag=name,values=model.state_dict()[name],global_step=(epoch_i)*step+step)
        #print ('/model/model-{}'.format(epoch_i) + str(month) + '.pkl')
        torch.save(model, Dir+'/model/model-{}'.format(epoch_i) + str(month)+'.pkl')
        print ('average loss_value:',np.mean(train_loss_list))
        print ('******************************************')
    endtime = time.time()
    running_time = endtime - starttime
    print('Running Time:', running_time / 60.0, '分')
train('07',model=model,optimizer=optimizer)
train('08',model=model,optimizer=optimizer)
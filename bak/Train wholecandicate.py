from DataLoad import ClickStreamData
from model import UserEncoder
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

from torchsummary import summary
cfg=conf()
#print (torch.cuda.is_available())#GPU兼容问题
#print(torch.cuda.device_count())
device ='cpu'    #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
Dir=os.path.abspath('')
CUR_PATH = Dir+'/runs/'
shutil.rmtree(CUR_PATH)

writer=SummaryWriter()
model = UserEncoder(cfg)

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
optimizer=optim.Adam(model.parameters(),lr=3e-1)
def longlikelossfunc(batchsize,batch_positive_value,batch_negative_value):  #targetlist：16*1，16*L-1 predictlist
    batch_log_positive_value=np.log(batch_positive_value)
    #print('batch_log_positive_value:',batch_log_positive_value)
    batch_log_negative_value=[]
    for negative_value in batch_negative_value:
        negative_value=np.mean([math.log(1-value) for value in negative_value])
        batch_log_negative_value.append(negative_value)
    #print('batch_log_negative_value:',batch_log_negative_value)
    L_positive=sum(batch_log_positive_value)
    L_negative=sum(batch_log_negative_value)
    #print('L_positive:',L_positive)
    #print('L_negative:',L_negative)
    loss_value=float(-1/(batchsize)*(L_positive+L_negative))
    #writer.add_scalar(tag='loss',scalar_value=loss_value.,global_step=)
    #print ('loss_value:',loss_value)
    return torch.Tensor([loss_value]).requires_grad_(),loss_value



def negativesamplingfunc(batchsize,batch_positive_value,batch_negative_value):  #targetlist：16*1，16*L-1 predictlist
    batch_exp_positive_value=np.exp(batch_positive_value)
    #print('batch_log_positive_value:',batch_log_positive_value)
    batch_exp_negative_value_sum=[]
    for negative_value in batch_negative_value:
        negative_value=np.sum([np.exp(value) for value in negative_value])
        batch_exp_negative_value_sum.append(negative_value)
    #print('batch_log_negative_value:',batch_log_negative_value)
    batch_p=batch_exp_positive_value/(batch_exp_positive_value+batch_exp_negative_value_sum)
    loss_value=np.mean(-np.log(batch_p))
    #writer.add_scalar(tag='loss',scalar_value=loss_value.,global_step=)
    #print ('loss_value:',loss_value)
    return torch.Tensor([loss_value]).requires_grad_(),loss_value


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
    K=20 #num of neagtive sampling
    #print (np.max(Travel_TypeIDList))
    #print('Travel_TypeIDList:',Travel_TypeIDList)
    BATCH_SIZE = cfg.batch_size  # 批训练的数据个数
    result=ClickStreamData.ClickStream(month)
    Training_Clickdata=np.array(result[0]) #训练集
    #print (Training_Clickdata)

    ID_traing = np.arange(Training_Clickdata.shape[0])
    UserID_train = Training_Clickdata[:, 0]    #UserID
    ItemID_long_train = Training_Clickdata[:, 1]  #long-termd点击流
    ItemID_short_train = Training_Clickdata[:, 2]  #short-term点击流
    Targe_traing=Training_Clickdata[:,3]  #下单的ItemID
    #print (Targe_traing)
    x=torch.from_numpy(ID_traing)
    y=torch.Tensor([int(x) for x in Targe_traing]) # 转换为list
    tensor_Train = Data.TensorDataset(x,y)   # 将数据封装成Dataset
    # 可使用索引调用数据
    # print ('tensor_data[0]: ', tensor_dataset[0])
    tensor_dataloader = Data.DataLoader(tensor_Train ,  # 封装的对象
                                        batch_size=BATCH_SIZE,  # 输出的batchsize
                                        shuffle=True,  # 随机输出
                                        num_workers=0)  # 只有1个进程
    z2 = []
    z3 = []
    z4 = []
    z5 = []
    CandicateItemIDList=result[2]  #候选项集
    for i in range(0, len(CandicateItemIDList)):  #计算一遍候选项集(所有的产品)
        ItemID = CandicateItemIDList[i]
        # print('推荐候选ItemID：',ItemID)
        #print ('TitleWordIDlist[i]:',TitleWordIDlist[i])
        z2.append(TitleWordIDlist[i])
        z3.append([DestinationIDList[i]])
        z4.append([Travel_TypeIDList[i]])
        z5.append([Travel_RegionIDList[i]])
    z2=torch.Tensor(z2)
    z3=torch.Tensor(z3)
    z4 = torch.Tensor(z4)
    z5=torch.Tensor(z5)
    starttime = time.time()
    for epoch_i in range(cfg.epoch):
        train_loss_list = []
        train_hit_list=[]
        model=model.train()
        model = model.to(device)
        for step, (batch_data, batch_target) in enumerate(tensor_dataloader):#一个batch（16个会话）
            #batch_data=batch_data.to(device)
            print('******One Batch****')
            #print(data, target)
            #TPEnocderone = TPEncoder(cfg)
            print('Epoch:', epoch_i, '|Step:', step, '|batch_data_user:',[UserID_train[int(ID)] for ID in batch_data], '|batch_target', batch_target)
            batch_userID=[]
            #batch_target=[]
            batch_long_term_TitleID=[]  #batch_long_term_embedding的输入
            batch_long_term_DestinationID=[]
            batch_long_term_TravelRegionID=[]
            batch_long_term_TravelTypeID=[]
            batch_short_term_TitleID=[]  #batch_short_term_embedding的输入
            batch_short_term_DestinationID = []
            batch_short_term_TravelRegionID = []
            batch_short_term_TravelTypeID = []

            for ID in batch_data:  #该batch中的一个用户（会话）
                #print('ID:',int(ID))
                #print('*****UserID******:',UserID_train[int(ID)])
                batch_userID.append(UserID_train[int(ID)])  #batch——UserID的输入
                #print('Long-term:', ItemID_long_train[int(ID)]) #Long-term点击流
                Long_term_TitleID=[]
                Long_term_DestinationID=[]
                Long_term_TravelRegionID=[]
                Long_term_TravelTypeID=[]
                if ItemID_long_train[int(ID)]!=[]:
                    for ItemID in ItemID_long_train[int(ID)]:  #一个产品
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
                else:
                    Long_term_TitleID.append(torch.Tensor(np.zeros(23)))
                    Long_term_DestinationID.append(torch.Tensor(np.zeros(1)))
                    Long_term_TravelRegionID.append(torch.Tensor(np.zeros(1)))
                    Long_term_TravelTypeID.append(torch.Tensor(np.zeros(1)))

                Long_term_TitleID = pad_sequence(Long_term_TitleID, batch_first=True)#补齐
                #print('Short-term:', ItemID_short_train[int(ID)])  #short-term点击流
                Short_term_TitleID = []
                Short_term_DestinationID=[]
                Short_term_TravelRegionID=[]
                Short_term_TravelTypeID=[]
                for ItemID in ItemID_short_train[int(ID)]:
                    Index_ID = np.where(ItemIDList == ItemID)
                    #print ('ItemID',ItemID,'Index_ID',Index_ID)
                    WordIDList = TitleWordIDlist[Index_ID]
                    #print(WordIDList)
                    Short_term_TitleID.append(torch.Tensor(WordIDList[0]))
                    Short_term_DestinationID.append(torch.Tensor(DestinationIDList[Index_ID]))
                    Short_term_TravelRegionID.append(torch.Tensor(Travel_RegionIDList[Index_ID]))
                    Short_term_TravelTypeID.append(torch.Tensor(Travel_TypeIDList[Index_ID]))
                Short_term_TitleID = pad_sequence(Short_term_TitleID, batch_first=True)
                #print('Short_term_TitleID:', Short_term_TitleID)
                #long_term的4个输入
                batch_long_term_TitleID.append(Long_term_TitleID)
                #print('Long_term_DestinationID:', Long_term_DestinationID)
                batch_long_term_DestinationID.append(torch.Tensor(Long_term_DestinationID))
                #print('Long_term_TravelRegionID:',Long_term_TravelRegionID)
                batch_long_term_TravelRegionID.append(torch.Tensor(Long_term_TravelRegionID))
                #print('Long_term_TravelTypeID:',Long_term_TravelTypeID)
                batch_long_term_TravelTypeID.append(torch.Tensor(Long_term_TravelTypeID))
                #short_term的4个输入
                batch_short_term_TitleID.append(Short_term_TitleID)
                batch_short_term_DestinationID.append(torch.Tensor(Short_term_DestinationID))
                batch_short_term_TravelRegionID.append(torch.Tensor(Short_term_TravelRegionID))
                batch_short_term_TravelTypeID.append(torch.Tensor(Short_term_TravelTypeID))
            #print('得出一个batch的输出：')
            batch_userID=np.array(batch_userID)
            x1=torch.from_numpy(batch_userID)

            #print('batch_userID:',x1.size())
            #print('batch_userID:',batch_userID.shape,batch_userID)  #UserID
            #print('batch_long_term_TitleID:',batch_long_term_TitleID)
            x2=pad_sequence(batch_long_term_TitleID, batch_first=True)
            x2= torch._cast_Long(x2)
            #print('x2:',x2.size())
            #print('batch_long_term_Title_embedding：',x2.size())

            x3=pad_sequence(batch_long_term_DestinationID, batch_first=True)
            x3 = torch._cast_Long(x3)
            #print('batch_long_term_DestinationID:',x3.size())
            x4=pad_sequence(batch_long_term_TravelTypeID, batch_first=True)
            x4 = torch._cast_Long(x4)
            #print('batch_long_term_TravelRegionID:', x4.size())
            x5=pad_sequence(batch_long_term_TravelRegionID,batch_first=True)
            x5 = torch._cast_Long(x5)
            #print('batch_long_term_TravelTypeID:', x5.size())

            y1=x1
            #print('batch_userID:', y1.size)
            y2=pad_sequence(batch_short_term_TitleID, batch_first=True)
            y2 = torch._cast_Long(y2)
            #print('batch_short_term_Title_embedding：',y2.size())
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
            z1 =torch.Tensor(np.zeros(z2.size(0))) #用
            z1=torch.unsqueeze(z1,dim=-1)
            # print ('x1:',x1.size())
            # print('x2:', x2.size())
            # print('x3:', x3.size())
            # print('x4:', x4.size())
            # print('x5:', x5.size())
            # print ('z1:',z1.size())
            # print('z2:', z2.size())
            # print('z3:', z3.size())
            # print('z4:', z4.size())
            # print('z5:', z5.size())
            x1 = torch.unsqueeze(x1, dim=1)
            x1 = torch.unsqueeze(x1, dim=1)
            x1 = x1.expand(-1, x2.size(1), -1)
            x3 = torch.unsqueeze(x3, dim=-1)
            x4 = torch.unsqueeze(x4, dim=-1)
            x5 = torch.unsqueeze(x4, dim=-1)

            y1 = torch.unsqueeze(y1, dim=1)
            y1 = torch.unsqueeze(y1, dim=1)
            y1 = y1.expand(-1, y2.size(1), -1)
            y3 = torch.unsqueeze(y3, dim=-1)
            y4 = torch.unsqueeze(y4, dim=-1)
            y5 = torch.unsqueeze(y5, dim=-1)
            RC_Score=model(x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,z1,z2,z3,z4,z5)
            #print(model)
            #print (summary(model,[x1.size(),x2.size(),x3.size(),x4.size(),x5.size(),y1.size(),y2.size(),y3.size(),y4.size(),y5.size(),z1.size(),z2.size(),z3.size(),z4.size(),z5.size()]))
            #print('RC_Score:',RC_Score.size())  #bitchsize*num(TP)
            top_k=20
            batch_positive_value=[] #target的softmax预测值
            batch_negative_value=[]#其他的候选项目的softmax预测值
            batch_hit=[]#hit
            for i in range(0,len(batch_userID)):
                print ('userID:',batch_userID[i])
                score=RC_Score[i].cpu().detach().numpy()
                #print ('RC_Socre:',score)
                top_k_idx = score.argsort()[::-1][0:top_k]
                #print ('top_k_idx:',top_k_idx)
                RC_ItemID=[CandicateItemIDList[idx] for idx in top_k_idx]
                print ('RC_ItemID:',RC_ItemID)
                target=int(batch_target[i].item())
                print ('target:',target)
                if target in RC_ItemID:
                    batch_hit.append(1)
                    train_hit_list.append(1)
                else:
                    batch_hit.append(0)
                    train_hit_list.append(0)
                index=CandicateItemIDList.index(target)
                positive_value=score[index]   #Target_value
                print('Target_positive_value:', positive_value)
                batch_positive_value.append(positive_value)
                negative_value=np.delete(score,index)
                if losstype == 'negativesampling':
                    negative_value=np.random.choice(negative_value,size=K)
                print ('negative_value:',negative_value)
                batch_negative_value.append(negative_value)
            if losstype=='negativesampling':
                loss = negativesamplingfunc(BATCH_SIZE, batch_positive_value, batch_negative_value)  # 计算loss function  #K个的negative samples
            else:
                loss=longlikelossfunc(BATCH_SIZE,batch_positive_value,batch_negative_value) # 计算loss function  #全局的negative samples
            loss_value=loss[0]
            loss_value_np=loss[1]
            print ('loss_value:',loss_value)
            hit_score=float(sum(batch_hit)) / BATCH_SIZE
            print('hit_score:',hit_score)
            print ('*********************')
            train_loss_list.append(loss_value_np)
            optimizer.zero_grad()  ##将以前的方向传播
            loss_value.backward()   ##直接自
            optimizer.step()
            writer.add_scalar(tag='loss', scalar_value=loss_value_np, global_step=(epoch_i)*step+step)
            writer.add_scalar(tag='accuracy',scalar_value=hit_score,global_step=(epoch_i)*step+step)
            for name in model.state_dict():
                writer.add_histogram(tag=name,values=model.state_dict()[name],global_step=(epoch_i)*step+step)


        #print ('/model/model-{}'.format(epoch_i) + str(month) + '.pkl')
        torch.save(model, Dir+'/model/model-{}'.format(epoch_i) + str(month)+'.pkl')
        print ('average loss_value:',np.mean(train_loss_list))
        print ('average hit ratio:',float(sum(train_hit_list))/len(train_hit_list))
        print ('******************************************')
    endtime = time.time()
    running_time = endtime - starttime
    print('Running Time:', running_time / 60.0, '分')
train('07',model=model,optimizer=optimizer)

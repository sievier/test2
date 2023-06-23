import torch
import torch.nn as nn
import numpy as np
from layer import TPEncoder,Biattention_one,Biattention_two, Gate_fusion,encoderuser
from torchsummary import summary

def binaryMatrix(x,Embeddingsize):  #Mask掩膜  x:输入样本B&L   Embeddingsize
    # 将targets里非pad部分标记为1，pad部分标记为0
    m = []
    PAD_token=0
    Ones=1
    Zeros=0
    #print('Ones,Zeros:',Ones,Zeros)
    for i, seq in enumerate(x):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(Zeros)
                break
            else:
                m[i].append(Ones)
                break
    m=torch.tensor(m)
    m=m.float()
    return m

class User_Candidate_Encoder(nn.Module):   #UserEncoder
    def __init__(self,cfg):
        super(User_Candidate_Encoder, self).__init__()
        """
        可以用这种把要的参数放到config中，我只是做了一部分，剩下的你可以去自己定义去写，这种方式，便于写起来比较方便
        """
        # self.vocab1 = cfg.vocab1
        # self.vocab2 = cfg.vocab2
        # self.vocab3 = cfg.vocab3
        # self.vocab4 = cfg.vocab4
        # self.vocab5 = cfg.vocab5
        # self.embedding_size1 =cfg.emebedding_size1
        # self.embedding_size2 = cfg.emebedding_size2
        # self.embedding_size3 = cfg.emebedding_size3
        # self.embedding_size4 = cfg.emebedding_size4
        # self.embedding_size5 = cfg.emebedding_size5
        # self.hidden1 = cfg.hidden1
        # self.hidden2 = cfg.hidden2
        #self.num_steps =cfg.num_steps
        self.TPEnocderone=TPEncoder(cfg)
        self.Biattention_one=Biattention_one(input_size=cfg.hidden4,hidden=cfg.hidden3,num_steps =cfg.num_steps)
        self.Biattention_two=Biattention_two(input_size=cfg.hidden4,hidden=cfg.hidden3,num_steps =cfg.num_steps)
        self.encoderuser=encoderuser(vocab_size=cfg.vocab1,embedding_size=cfg.embedding_size1,hidden1=cfg.hidden1)
       # self.c2p_attention=c2p_attention(input_size=cfg.hidden5)
        self.Gate_fusion=Gate_fusion(hidden1=cfg.hidden5)
    def forward(self,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,s2,s3,s4,s5):
        #x1,y1,z1为UserID
        ##x2,x3,x4,x5:long-term;
        #y1,y2,y3,y4,y5:short-term;
        ## 所有的传入格式必须是[B,L,D]
        ##candiate tp 只经过 TP encoder  s2,s3,s4,s5分别对应s_title,s_destination,s_travel_type,s_travel_region
        long_tem_list=[]
        short_term_list=[]
        #print('x1[:,1,:]',x1[:,1,:])

        q_u=self.encoderuser(x1[:,0,:]) #所有的UserID进行embedding学习
        #print('userIDEmbedding q_u.shape:',q_u.size())
        print('Encoder long_term behaviors 中的产品')
        weight_view_list = []
        for i in range(x2.size(1)):
            Weight_long_term=self.TPEnocderone(q_u,x2[:,i,:],x3[:,i,:],x4[:,i,:],x5[:,i,:])  #long_term中产品编码
            #Weight=Weight_long_term[0]
            #Weight=Weight[0,0,:]
            #print('View level-attention of title,Destination,TravelType,TravelRegionID:', Weight)
            #print('View level-attention of title,Destination,TravelType,TravelRegionID:',Weight.size())   #[B, l, 1]
            #weight_view_list.append(Weight)
            #print(Weight.size())
            long_term=Weight_long_term[1] #Tp表达向量
            # print('TPEncoderShape:',long_term.size())
            long_tem_list.append(long_term[:,None,:])
            sample_Weight=Weight_long_term[0]
            weight_view_list.append(sample_Weight.detach().numpy())
        #print('long_termTP_i:', x2[:,i,:],x3[:,i,:],x4[:,i,:],x5[:,i,:])
        print('Encoder short_term behaviors 中的产品：')
        for j in range(y2.size(1)):
            Weight_short_term=self.TPEnocderone(q_u,y2[:,j,:],y3[:,j,:],y4[:,j,:],y5[:,j,:])  #short_term中产品编码
            #Weight=Weight_short_term[0]
            #print('View level-attention of title,Destination,TravelType,TravelRegionID:',Weight)
            #print('View level-attention of title,Destination,TravelType,TravelRegionID:',Weight.size())   #[B, l, 1]
            #weight_view_list.append(Weight)
            #print(Weight.size())
            short_term = Weight_short_term[1]
            # print('TPEncoderShape:',long_term.size())
            short_term_list.append(short_term[:,None,:])
            sample_Weight=Weight_short_term[0]
            weight_view_list.append(sample_Weight.detach().numpy())
        long_term_one=torch.cat(long_tem_list,dim=1)  #拼接
        short_term_one=torch.cat(short_term_list,dim=1) #拼接
        #print('weight_view_one:',weight_view_one)
        #print('weight_view_one size:',weight_view_one.size())  #[Num*4*1]
        #print("tp dou zhi xing wang")
        #print('long term one size',long_term_one.size())  #[B*L*256]
        #print('short term one',short_term_one.size())
        #print('x3:',torch.squeeze(x3,dim=2))
        long_term_encoder=self.Biattention_one(q_u,long_term_one)  #需要进行Mask操作
        #print("long_term_behavior_encoder:", long_term_encoder,long_term_encoder.size())# ([B, 256])
        short_term_encoder=self.Biattention_two(q_u,short_term_one)
        #print ("short_term_behavior_encoder:", short_term_encoder.size(),short_term_encoder)
        #print("short_term_encoder shape:",short_term_encoder.size()) #([B, 256])
        #print('x4:',torch.squeeze(x4,dim=2),torch.squeeze(x4,dim=2).size())
        mask=binaryMatrix(torch.squeeze(x4,dim=2),1)
        #print('mask of long_term_beahvior:',mask)
        long_term_encoder=torch.mul(long_term_encoder,mask)
        #print("long_term_behavior_encoder_after mask:", long_term_encoder.size(),long_term_encoder)
        userbehaviorencoder_value,F_u_weight = self.Gate_fusion(q_u,long_term_encoder,short_term_encoder)
        #print('userbeahviorencoder_value_Shape:', userbehaviorencoder_value.size()) #([B, 256])
        print('UserEncoder END!')

        ######*******candidate_Item_Encoder*****#######
        #print('Positive:',p2,p3,p4,p5)
        #print('Negative:',n2,n3,n4,n5)
        batch_sample_EncoderList = []
        #weight_view_list=[]
        for i in range(s2.size(1)):
            sample_Weight_Encoder=self.TPEnocderone(q_u,s2[:,i,:],s3[:,i,:],s4[:,i,:],s5[:,i,:])  #Positive样本中的编码
            #print('Positive_Encoder Size:',Positive_Encoder.size()) #torch.Size([2, 256])
            #sample_Weight=sample_Weight_Encoder[0]
            sample_Encoder=sample_Weight_Encoder[1]
            batch_sample_EncoderList.append(sample_Encoder[:,None,:])
            #weight_view_list.append(sample_Weight.detach().numpy())
        #print('weight_view_list:',weight_view_list)
        batch_sample_Encoder = torch.cat(batch_sample_EncoderList, dim=1)  # 拼接
        print('Candidate_Item_Encoder END!')
        #print ('batch_sample_Encoder:',batch_sample_Encoder)
        #print('batch_sample_Encoder shape',batch_sample_Encoder.size())
        return userbehaviorencoder_value,batch_sample_Encoder,weight_view_list,F_u_weight   #用户行为编码，候选项集的Positive和Negative样本的编码,view-level weight,长期的权重
#########################


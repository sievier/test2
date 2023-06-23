import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import conf
cfg=conf()

device ='cpu'   #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def binaryMatrix(x,Embeddingsize):  #Mask掩膜  x:输入样本B&L   Embeddingsize
    # 将targets里非pad部分标记为1，pad部分标记为0
    m = []
    PAD_token=0
    Ones=np.ones(Embeddingsize)#  填充1
    Zeros=np.zeros(Embeddingsize)# 填充0
    #print('Ones,Zeros:',Ones,Zeros)
    for i, seq in enumerate(x):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(Zeros)
            else:
                m[i].append(Ones)
    m=torch.tensor(m)
    m=m.float()
    return m

def WeightbinaryMatrix(x,Embeddingsize):  #Mask掩膜  x:输入样本B&L   Embeddingsize
    # 将targets里非pad部分标记为1，pad部分标记为0
    m = []
    PAD_token=0
    Ones=np.ones(Embeddingsize)#
    Zeros=np.zeros(Embeddingsize)# 填充接近于0的 -np.inf
    #print('Ones,Zeros:',Ones,Zeros)
    for i, seq in enumerate(x):
        m.append([])
        for token in seq:
            if token == PAD_token:
                Pad_new=Ones*(-1e9) # 填充无穷小的 -np.inf
                # print('Zeros:',Zeros_new)
                m[i].append(Pad_new)
            else:
                m[i].append(Zeros)
    m=torch.tensor(m)
    m=m.float()
    return m

class encodertitle(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden1, file='WordID_embedding.npy'):  # WordID_embedding
        super(encodertitle, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden1 = hidden1
        self.file = file
        # self.embeddingtitle=nn.Embedding(vocab_size,embedding_size)
        self._build_weight()
        self.embeddingtitle = nn.Embedding.from_pretrained(self.weight, padding_idx=0)
        self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden1, num_layers=1, bidirectional=True,
                              batch_first=True)
        self.dopout = nn.Dropout(0.8)

    def _build_weight(self):
        if self.file is not None:
            self.weight = np.load(file=self.file)
            self.weight = torch.Tensor(self.weight)
            # self.weight = np.load('WordID_embedding.npy')
            # WordID_embedding = np.load('WordID_embedding.npy')
            # print(self.weight)
            print("执行传入预训练的词向量%s" % self.file)
        else:
            self.weight = torch.rand(size=(self.vocab_size, self.embedding_size))
            print("执行的随机生成的向量")
    def forward(self,x):
        h0=torch.zeros(size=(1*2,x.size(0),self.hidden1)).to(device)
        c0=torch.zeros(size=(1*2,x.size(0),self.hidden1)).to(device)
        x=torch._cast_Long(x).to(device)
        #print('title_x:',x.size()) #B*23
        #print('title_word:',x)
        output=self.embeddingtitle(x)
        #print('title_embedding_output:', output,output.size())# b*23*300
        output=self.dopout(output)
        output,_=self.bilstm(output,(h0,c0))
        #print('title_embedding_BiLSTM_output.size:', output,output.size())  #b*23*256
        return output

class encoderuser(nn.Module):
    ####hidden1
    def __init__(self,vocab_size,embedding_size,hidden1):
        super(encoderuser, self).__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.hidden1=hidden1
        self.embeddinguser=nn.Embedding(vocab_size,embedding_size)
        self.linear1=nn.Linear(embedding_size,2*hidden1)
    def forward(self,x):
        #x【1,2,89,67】
        ##[4,embedding_size]
        #print('user',x.dtype)
        x=torch._cast_Long(x).to(device)                                   ##chuan GPU
        output=self.embeddinguser(x)
        output=self.linear1(output)
        output=torch.relu(output)
        return output

class ecoderDestination(nn.Module):
    def __init__(self,vocab_size,embedding_size,output_size):
        super(ecoderDestination, self).__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.output_size=output_size
        self.embeddingDes=nn.Embedding(vocab_size,embedding_size,padding_idx=0)
        self.linear1=nn.Linear(embedding_size,output_size,bias=False)

    def forward(self,x):
        #print(x.size())
        #print(x)
        x=torch._cast_Long(x).to(device)
        output=self.embeddingDes(x)
        output=self.linear1(output)
        output=torch.relu(output)

        return output

class Categories(nn.Module):
    def __init__(self,vocab_size1,embedding_size1,vocab_size2,embedding_size2,output_size):
        super(Categories, self).__init__()

        ####embedding1
        self.vocab_size1=vocab_size1
        self.embedding_size1=embedding_size1
        self.embedding1=nn.Embedding(vocab_size1,embedding_size1,padding_idx=0)
        self.linear1=nn.Linear(embedding_size1,output_size,bias=False)
        ##embedding2
        self.vocab_size2=vocab_size2
        self.embedding_size2=embedding_size2
        self.embedding2=nn.Embedding(vocab_size2,embedding_size2,padding_idx=0)
        self.linear2=nn.Linear(embedding_size2,output_size,bias=False)

    def forward(self,x,y):
        ###x,y分别代表不同的输入,x到embeding1,y到embeding2中
        ##输入x
        x=torch._cast_Long(x).to(device)
        output1=self.embedding1(x)
        output1=self.linear1(output1)
        output1=torch.relu(output1)
        ##输入y
        #print(y)
        y=torch._cast_Long(y).to(device)
        output2=self.embedding2(y)
        output2=self.linear2(output2)
        output2=torch.relu(output2)
        return output1,output2

class attention_TP_Title(nn.Module):  #'Word level-attention，Title中word的attention计算
    def __init__(self,input_size,num_steps):
        super(attention_TP_Title, self).__init__()
        self.num_steps=num_steps
        self.w=nn.Parameter(torch.rand(size=(1,num_steps,input_size)))
        #self.w = nn.Parameter(0.1 * torch.rand(size=(1, num_steps, input_size)))
        self.b=nn.Parameter(torch.zeros(size=(1,1,input_size)))
    def forward(self,x,y): #x=userIDEmbedding,y=title
        ##x[B,D]->[B,1,D]->[B,L,D]
        ##[B,L,D]
        #print("attentiion",y)
        #print('attention',x.size())
        #print('attention',y.size())
        x=torch.squeeze(x,dim=1)
        x=torch.unsqueeze(x,dim=1)
        x=x.expand(-1,y.size(1),-1)
        print('userIDEmbedding:',x.size())
        #print('title embedding:',y,y.size())
        #print('title embedding size:', y.size())  #([1, 23, 256])
        qi=torch.tanh(torch.add(torch.mul(x,self.w),self.b))
        ##[B,L,1]
        ai=torch.sum(torch.mul(y,qi),dim=-1,keepdim=True)
        #print('Word level-attention ai and size:',ai, ai.size())#([B, L# , 1])
        mask = WeightbinaryMatrix(ai, 1)  # ([B, L# , 1])
        #print('mask:',mask)
        ai_mask=torch.add(ai,mask)
        #print('3. ai_mask and size:', ai_mask.size(), ai_mask)
        '''
        exp_ai=torch.exp(ai*ai_mask)  #掩膜，0的位置统一补-np.inf
        print('exp_ai and after Mask:',exp_ai, exp_ai.size())
        ai=exp_ai/torch.sum(exp_ai,dim=1,keepdim=True)  #nan
        print('after mask Word-level attention and size:', ai, ai.size())#([B, L# , 1])
        '''
        ai_softmax=torch.softmax(ai_mask,dim=1)
        #print('4. after mask Word-level attention and size:', ai_softmax, ai_softmax.size())#([B, L# , 1])
        ri=torch.sum(torch.mul(ai_softmax,y),dim=1)
        #print('Title_embedding and size:',ri.size(),ri)#  [B*256]
        return ri

class all_attention_TP(nn.Module):   # Weight(Attention) of title,Destination,TravelType,TravelRegionID
    def __init__(self,):
        super(all_attention_TP, self).__init__()

    def build_shape(self,input_shape):
        ##input_shape [B,4,D]
        #self.w=nn.Parameter(0.1*torch.rand(size=(1,input_shape[1],input_shape[2])))
        self.w=nn.Parameter(torch.rand(size=(1,input_shape[1],input_shape[2])))
        self.b=nn.Parameter(torch.zeros(size=(1,1,input_shape[2])))

    def forward(self,x,y1,y2,y3,y4):
        y1=torch.unsqueeze(y1,dim=1)
        #y4=torch.squeeze(y4,dim=2)
        #print("alltp",x.size()) #[1, 1, 256])
        #print("alltp y1", y1.size()) #[1, 1, 256])
        # print("alltp", y2.size())#[1, 1, 256])
        # print("alltp", y3.size())#[1, 1, 256])
        #print("alltp y4-2", y4) #[1, 1, 256])
        ## x[B,D]
        ###y1 tile ,y2 Destination, y3,y4 Categoties
        ###y1[B,D] y2[B,D] y3[B,D] y4[B,D]
        ##[B,4*D],b[B,4,D]
        #print("y1",y1.size())
        y=torch.cat([y1,y2,y3,y4],dim=1)  #y已经mask处理过
        input_shape=y.size()
        self.build_shape(input_shape)
        # ##[B,4,D]
        # print("tpqll x",x.size())
        # print("tqall y",y.size())
        qi=torch.tanh(torch.mul(self.w,x)+self.b)
        ai=torch.softmax(torch.sum(torch.mul(qi,y),dim=-1,keepdim=True),dim=1)
        #print('View-level attention of title,Destination,TravelType,TravelRegionID:',ai)
        '''
        mask = WeightbinaryMatrix(ai, 1)  # ([B, L# , 1])
        #print('mask:',mask)
        ai_mask=torch.add(ai,mask)
        #print(ai.size())
        ri=torch.sum(torch.mul(ai_mask,y),dim=1)
        '''
        ri = torch.sum(torch.mul(ai, y), dim=1)
        #print('ri.size',ri.size())
        return ai,ri

class TPEncoder(nn.Module):
    def __init__(self,cfg):
        super(TPEncoder, self).__init__()
        self.vocab1=cfg.vocab1
        self.vocab2=cfg.vocab2
        self.vocab3=cfg.vocab3
        self.vocab4=cfg.vocab4
        self.vocab5=cfg.vocab5
        self.embedding_size1=cfg.embedding_size1
        self.embedding_size2=cfg.embedding_size2
        self.embedding_size3=cfg.embedding_size3
        self.embedding_size4=cfg.embedding_size4
        self.embedding_size5=cfg.embedding_size5
        self.hidden1=cfg.hidden1
        self.hidden2=cfg.hidden2
        self.num_steps=cfg.num_steps
        self.encodertitle=encodertitle(vocab_size=cfg.vocab2,embedding_size=cfg.embedding_size2,hidden1=cfg.hidden1)
        self.ecoderDestination=ecoderDestination(vocab_size=cfg.vocab3,embedding_size=cfg.embedding_size3,output_size=cfg.hidden2)
        self.Categories=Categories(vocab_size1=cfg.vocab4,embedding_size1=cfg.embedding_size4,vocab_size2=cfg.vocab5,embedding_size2=cfg.embedding_size5,output_size=cfg.hidden2)
        self.attention_TP_Title=attention_TP_Title(input_size=2*cfg.hidden1,num_steps=cfg.num_steps)
        self.all_attention_TP=all_attention_TP()

    def forward(self,x1,x2,x3,x4,x5):
        #print("x1.shape",x1.size())
        user=x1 #UserIDEmbedding
        #print('Title_word:',x2    )
        title=self.encodertitle(x2)  #title tensor
        #print('1. title_word bilstm output:',title,title.size())#[B, 23, 256]
        mask_title=binaryMatrix(x2,self.hidden1*2)  #title tensor对应的mask,掩码为0
        #print('mask_title:',mask_title)  #b*23*256
        title=torch.mul(title,mask_title)  #title经过bi-lstm处理后的Mask操作
        #print('2. title_word after mask:', title,title.size()) #[B, 23, 256]
        user_title=self.attention_TP_Title(x=user,y=title)  #Word_level Attention  [B*256]
        #print('5. user_title', user_title,user_title.size())#[1,256])
        #print('x3,x4,x4:',x3,x4,x4)
        destination=self.ecoderDestination(x3)
        mask_destination=binaryMatrix(x3,cfg.hidden2)
        destination=torch.mul(destination,mask_destination)
        cattion1,cattion2=self.Categories(x4,x5)
        mask_cattion1 = binaryMatrix(x4, cfg.hidden2)
        mask_cattion2 = binaryMatrix(x5, cfg.hidden2)
        cattion1=torch.mul(cattion1,mask_cattion1)
        cattion2=torch.mul(cattion2,mask_cattion2)
        # print('destination',destination.size()) #[1, 1, 256])
        # print('cattion1', cattion1.size())#[1, 1, 256])
        # print('cattion2',cattion2.size())#[1, 1, 256])
        #print('destination_embedding',destination,destination.size())
        #print('cattion1_embedding', cattion1,cattion1.size())
        #print('cattion2_embedding',cattion2,cattion2.size())
        ai_ri=self.all_attention_TP(x=user,y1=user_title,y2=destination,y3=cattion1,y4=cattion2)
        ai=ai_ri[0]#View level-attention
        output=ai_ri[1] #TP表达向量
        return ai,output


####假定参数不共享的情况下
class Biattention_one(nn.Module):  #long_term的表达学习
    def __init__(self,input_size,hidden,num_steps):
        super(Biattention_one, self).__init__()
        self.input_size=input_size
        self.num_steps = num_steps
        self.hidden=hidden
        self.bilstm=nn.LSTM(input_size=input_size,hidden_size=hidden,batch_first=True,bidirectional=True)
    def build_shape(self,input_shape):
        self.w=nn.Parameter(torch.rand(size=(1,1,input_shape[2])))
        #self.w = nn.Parameter(0.1 * torch.rand(size=(1, num_steps, input_size)))
        self.b=nn.Parameter(torch.zeros(size=(1,1,input_shape[2])))
    def forward(self,q_u,L): #U:user;L:long-term-encoder
        #B,N,L]
        #print("q_u_long",q_u.size())

        ho=torch.zeros(size=(1*2,L.size(0),self.hidden)).to(device)
        co=torch.zeros(size=(1*2,L.size(0),self.hidden)).to(device)
        output,_=self.bilstm(L,(ho,co))
        #print('output_LSTM:', output)
        #print('output_LSTM:',output.size())  #B*L*256
        q_u=torch.squeeze(q_u,dim=2)
        #print("qu shape",q_u.size())
        self.build_shape(q_u.size())
        qi=torch.tanh(torch.add(torch.mul(q_u,self.w),self.b))
        ##[B,L,1]
        ai=torch.sum(torch.mul(output,qi),dim=-1,keepdim=True)
        #print('ai.size:',ai.size())
        ai=torch.softmax(ai,dim=1)
        #print('Item level-attention of long_term:',ai)
        ri=torch.sum(torch.mul(ai,output),dim=1)
        return ri
        #return output

class Biattention_two(nn.Module):  #long_term的表达学习
    def __init__(self,input_size,hidden,num_steps):
        super(Biattention_two, self).__init__()
        self.input_size=input_size
        self.num_steps = num_steps
        self.hidden=hidden
        self.bilstm=nn.LSTM(input_size=input_size,hidden_size=hidden,batch_first=True,bidirectional=True)
    def build_shape(self,input_shape):
        self.w=nn.Parameter(torch.rand(size=(1,1,input_shape[2])))
        #self.w = nn.Parameter(0.1 * torch.rand(size=(1, num_steps, input_size)))
        self.b=nn.Parameter(torch.zeros(size=(1,1,input_shape[2])))
    def forward(self,q_u,S): #U:user;L:long-term-encoder
        #B,N,L]
        ##[B,L,D]
        #print("q_u_short",q_u.size())
        ho=torch.zeros(size=(1*2,S.size(0),self.hidden)).to(device)
        co=torch.zeros(size=(1*2,S.size(0),self.hidden)).to(device)
        output,_=self.bilstm(S,(ho,co))
        #print('output_LSTM:',output.size())
        q_u=torch.squeeze(q_u,dim=2)
        #print("qu shape",q_u.size())
        self.build_shape(q_u.size())
        qi=torch.tanh(torch.add(torch.mul(q_u,self.w),self.b))
        ##[B,L,1]
        ai=torch.sum(torch.mul(output,qi),dim=-1,keepdim=True)
        #print('ai.size:',ai.size())
        ai=torch.softmax(ai,dim=1)
        #print('Item level-attention of long_term:',ai)
        ri=torch.sum(torch.mul(ai,output),dim=1)
        return ri
        #return output

class Gate_fusion(nn.Module):
    def __init__(self,hidden1):
        super(Gate_fusion, self).__init__()
        self.hidden1=hidden1
        # self.wq=nn.Linear(self.hidden1,self.hidden1)
        # self.ws=nn.Linear(self.hidden1,self.hidden1)
        # self.wl=nn.Linear(self.hidden1,self.hidden1)
        # self.b=nn.Parameter(torch.ones(size=(1,self.hidden1)))
        self.wq=nn.Linear(self.hidden1,1)
        self.ws=nn.Linear(self.hidden1,1)
        self.wl=nn.Linear(self.hidden1,1)
        self.b=nn.Parameter(torch.ones(size=(1,1)))
    def forward(self,q_u,L,S ):
        qu=q_u[:,0,:]
        q_u=torch.squeeze(q_u,dim=1)
        #print("qu 6666",q_u.size())
        Uq=self.wq(q_u)
        Ls=self.ws(L)  #long-term behvaiors
        #print('Ls',Ls.size(),Ls)
        mask_Ls = WeightbinaryMatrix(L, 1)  # ([B, L# , 1])
        #print('1Gated Fusion mask_Ls:', mask_Ls.size(), mask_Ls)
        mask_Ls=torch.squeeze(mask_Ls,dim=2)
        #print('2Gated Fusion mask_Ls:', mask_Ls.size(), mask_Ls)
        mask_Ls=mask_Ls[:,[0]]
        #print('3Gated Fusion mask_Ls:',mask_Ls.size(),mask_Ls)
        Ls=torch.add(Ls,mask_Ls)
        #print ('4long-term behavior encoder after mask:',Ls.size(),Ls)
        sl=self.wl(S)  #short-term behvaiors
        # print("UQ 3333SHAPE",Uq.size())
        # print("sl 3333shape",sl.size())
        # print("self.b 333333shape",self.b.size())
        F_U=torch.sigmoid(Uq+Ls+sl+self.b)
        #print('F_U.size:',F_U.size())
        #print("F_U",F_U)
        O_U=(1-F_U)*S+F_U*L  #F_u Long-term behavior
        return O_U,F_U


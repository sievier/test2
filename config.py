#from layer import encodertile
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Data

#from DataLoad import TPData
#TPData()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class conf ():
    def __init__(self):

        '''
        这里的参数都是要自己调的
        '''
        self.Short_N = 20  #N-gram # 取long-term的参数
        self.Long_N = 40 #N-gram 截取
        self.vocab1 = 89843 #user
        self.vocab2 =10025 #Title
        self.vocab3 = 150 #Destination
        self.vocab4 = 10 #Travel Region (TT)
        self.vocab5 =13  #Travel Type (TR)
        self.embedding_size1 = 300  #user
        self.embedding_size2 = 300  #Title #Destination
        self.embedding_size3 = 300 #Destination
        self.embedding_size4 = 300  #Travel Region (TR)
        self.embedding_size5 = 300 #Travel Type (TT)

        self.hidden1 = 128
        self.hidden2 = 256  #mlp 输出纬度 ，destination  categories
        self.hidden3=128
        self.hidden4=256     ##bi-lstm  long and short term
        self.hidden5=256         ##caps
        self.num_steps =23 #title词的最大长度
        self.batch_size=64#16,32,64,128,256
        self.epoch=1
        self.K_neagtive_sampling=31  #negative_sampling ratio
        self.word_Embedding_file='WordID_embedding.npy'#Title  word_embedding

cfg=conf()
#print(cfg.vocab1)
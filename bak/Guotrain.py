import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from model import  Net
from load_data_help import start_item_data
from  layer import caps_loss,one_hot
###train Subj 的
# file="data/Subj/Subj.npy"
#
# dataset=start_item_data(low_count=2,num_steps=19,batch_size=32,file='Subj')
# testdataset=start_item_data(low_count=2,num_steps=19,batch_size=32,file='Subj',training=False)
# model=Net(vocab_size=10939+4,embedding_size=300,hidden_size=128,out_size=128,num_length=19,num_class=2,file=None)
# optimizer=optim.Adam(model.parameters(),lr=1e-3)
# # #MR_old 阈值为2 vocab_size 9973
dataset=start_item_data(low_count=2,num_steps=20,batch_size=32,file='MR_old')
testdataset=start_item_data(low_count=2,num_steps=20,batch_size=32,file='MR_old',training=False)
model=Net(vocab_size=9973,embedding_size=128,hidden_size=300,out_size=128,num_length=20,num_class=2,file=None)
optimizer=optim.Adam(model.parameters(),lr=1e-3)
####TREC  ###97.3% 第9论
# dataset=start_item_data(low_count=0,num_steps=10,batch_size=32,file='TREC')
# testdataset=start_item_data(low_count=0,num_steps=10,batch_size=32,file='TREC',training=False)
# model=Net(vocab_size=9361,embedding_size=300,hidden_size=128,out_size=128,num_length=10,num_class=6,file=None)
# optimizer=optim.Adam(model.parameters(),lr=1e-3)
###SST-2
# dataset=start_item_data(low_count=2,num_steps=19,batch_size=32,file='SST2')
# testdataset=start_item_data(low_count=2,num_steps=19,batch_size=32,file='SST2',training=False)
# model=Net(vocab_size=7146,embedding_size=300,hidden_size=128,out_size=128,num_length=19,num_class=2,file=None)
# optimizer=optim.Adam(model.parameters(),lr=1e-2)

###agnews  ##10  27106 ###90.8  第9论
# dataset=start_item_data(low_count=10,num_steps=228,batch_size=64,file='agnews')
# testdataset=start_item_data(low_count=10,num_steps=228,batch_size=64,file='agnews',training=False)
# model=Net(vocab_size=27106,embedding_size=300,hidden_size=128,out_size=128,num_length=228,num_class=4,file=None)
# optimizer=optim.Adam(model.parameters(),lr=1e-2)

###CR_old
# dataset=start_item_data(low_count=0,num_steps=23,batch_size=32,file='CR_old')
# testdataset=start_item_data(low_count=0,num_steps=23,batch_size=32,file='CR_old',training=False)
# model=Net(vocab_size=5246,embedding_size=300,hidden_size=128,out_size=128,num_length=23,num_class=2,file=None)
#optimizer=optim.Adam(model.parameters(),lr=1e-3)

###
def trian(epochs,dataset,model,optimizer=None,testdataset=None):
    use_gpu=torch.cuda.is_available()
    for i in range(epochs):
        if optimizer is not  None:
            optimizers=optimizer
        else:
            if i< epochs-3:
                optimizers=optim.Adam(model.parameters(),lr=1e-3)
                print('第 %d steps, 使用学习率为1e-3'%i)
            else:
                optimizers=optim.Adam(model.parameters(),lr=1e-4)
                print("第 %d steps 使用学习率le-4"%i)

        train_accuracy_list = []
        train_loss_list = []
        for index, (b_x, b_y) in enumerate(dataset):
            model.train()
            model.cuda()
            b_x = b_x.type(torch.LongTensor)
            b_x = Variable(b_x)
            b_y = b_y.type(torch.LongTensor)
            b_y = Variable(b_y)
            capusle= model(b_x)
            length=capusle.norm(dim=-1)
            ###至于这里需不需要加超参数调节，在另一说
            preds = torch.max(length, dim=1)[1]

            correct = torch.eq(preds, b_y)
            correct = torch._cast_Double(correct)
            accuracy = torch.mean(correct)
            t_y = one_hot(value=b_y, depth=2)

            length=torch._cast_Double(length)
            loss=caps_loss(y_true=t_y,y_pred=length)
            train_loss_list.append(loss.item())
            train_accuracy_list.append(accuracy.item())
            loss.backward()
            optimizers.step()
            optimizers.zero_grad()

            print("epoch: %d   steps: %d   loss :%3.3f   accuracy:%3.3f " % (i, index, loss, accuracy.item()))
        print("epoch:%d average_loss:%3.3f average_accuracy:%3.3f" % (i, np.mean(train_loss_list), np.mean(train_accuracy_list)))

        torch.save(model, 'model/MR_old/model-{}.pkl'.format(i))


        if testdataset is not None:
            test_model=torch.load('model/MR_old/model-{}.pkl'.format(i))
            test_accuracy_list=[]

            for index_i,(test_x,test_y) in enumerate(testdataset):
                test_model.eval()
                test_x=test_x.type(torch.LongTensor)
                test_x=Variable(test_x)

                test_y=test_y.type(torch.LongTensor)
                test_y=Variable(test_y)

                test_capsule=test_model(test_x)
                test_legnth=test_capsule.norm(dim=-1)
                test_pred=torch.max(test_legnth,dim=1)[1]
                test_correct=torch.eq(test_pred,test_y)
                test_correct=torch._cast_Double(test_correct)
                test_accuracy=torch.mean(test_correct)
                test_accuracy_list.append(test_accuracy.item())

                print("steps: %d accuracy: %3.3f"%(index_i,test_accuracy.item()))

            test_average_accuracy=np.mean(test_accuracy_list)

            print('average accuracy: %3.3f'%test_average_accuracy)


def test(dataset):
    test_model = torch.load('model/MR_old/model-8.pkl')
    test_accuracy_list = []

    for index_i, (test_x, test_y) in enumerate(testdataset):
        test_model.eval()
        test_x = test_x.type(torch.LongTensor)
        test_x = Variable(test_x)

        test_y = test_y.type(torch.LongTensor)
        test_y = Variable(test_y)

        test_capsule = test_model(test_x)
        test_legnth = test_capsule.norm(dim=-1)
        test_pred = torch.max(test_legnth, dim=1)[1]
        test_correct = torch.eq(test_pred, test_y)
        test_correct = torch._cast_Double(test_correct)
        test_accuracy = torch.mean(test_correct)
        test_accuracy_list.append(test_accuracy.item())

        print("steps: %d accuracy: %3.3f" % (index_i, test_accuracy.item()))

    test_average_accuracy = np.mean(test_accuracy_list)
    print('average accuracy: %3.3f' % test_average_accuracy)


#trian(epochs=10,dataset=dataset,model=model,testdataset=testdataset)
#test(dataset=testdataset)


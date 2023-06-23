
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
from  config import  config


cfg=config()

dataset=start_item_data(low_count=cfg.low_count,num_steps=cfg.num_steps,batch_size=cfg.batch_size,file=cfg.file)
testdataset=start_item_data(low_count=cfg.low_count,num_steps=cfg.num_steps,batch_size=cfg.batch_size,file=cfg.file,training=False)
model=Net(cfg)
#print(model)

device=torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
print(device)
loss_fn=nn.CrossEntropyLoss()
def trian(epochs,dataset,model,optimizer=None,testdataset=None):
    for i in range(epochs):
        if optimizer is not None:
            optimizers=optimizer
            #optimizers= optim.lr_scheduler.StepLR(optimizers, 2, 0.1)
        else:
            optimizers=optim.Adam(model.parameters(),lr=cfg.learning_rate)
            #optimizers=optim.lr_scheduler.StepLR(optimizers,2,0.1)
        #optimizers=adjust_learning_rate(optimizers,i)
        train_accuracy_list = []
        train_loss_list = []
        for index, (b_x, b_y) in enumerate(dataset):
            model.train()
            model=model.to(device)
            b_x = b_x.type(torch.LongTensor)
            b_x = Variable(b_x).to(device)
            b_y = b_y.type(torch.LongTensor)
            b_y = Variable(b_y)

            logist=model(b_x)
            logist=torch.softmax(logist,dim=-1)

            preds=torch.argmax(logist,dim=1)
            preds=preds.to(device='cpu')

            correct = torch.eq(preds, b_y)
            correct = torch._cast_Double(correct)
            accuracy = torch.mean(correct)

            loss=loss_fn(logist,b_y)
            train_loss_list.append(loss.item())
            train_accuracy_list.append(accuracy.item())
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            #writer.add_graph(model=model,input_to_model=torch._cast_Long(torch.ones(1,10)))
            print("epoch: %d   steps: %d   loss :%3.3f   accuracy:%3.3f " % (i, index, loss, accuracy.item()))

        print("epoch:%d average_loss:%3.3f average_accuracy:%3.3f" % (i, np.mean(train_loss_list), np.mean(train_accuracy_list)))

        torch.save(model, cfg.save_model+'model-{}.pkl'.format(i))


        with torch.no_grad():
            if testdataset is not None:

                test_model=torch.load(cfg.save_model+'model-{}.pkl'.format(i))
                test_model=test_model.to(device)
                test_accuracy_list=[]

                for index_i,(test_x,test_y) in enumerate(testdataset):
                    test_model.eval()
                    test_x=test_x.type(torch.LongTensor)
                    test_x=Variable(test_x).to(device)

                    test_y=test_y.type(torch.LongTensor)
                    test_y=Variable(test_y)

                    test_logist=test_model(test_x)

                    test_pred=torch.softmax(test_logist,dim=1)
                    test_pred=torch.argmax(test_pred,dim=1)

                    test_pred=test_pred.to(device='cpu')
                    test_correct=torch.eq(test_pred,test_y)
                    test_correct=torch._cast_Double(test_correct)
                    test_accuracy=torch.mean(test_correct)
                    test_accuracy_list.append(test_accuracy.item())

                    print("steps: %d accuracy: %3.3f"%(index_i,test_accuracy.item()))

                test_average_accuracy=np.mean(test_accuracy_list)

                print('average accuracy: %3.3f'%test_average_accuracy)


def test(dataset):

    test_model = torch.load('model/agnews/model-0.pkl')
    test_accuracy_list = []
    for index_i, (test_x, test_y) in enumerate(dataset):
        test_model.eval()
        test_x = test_x.type(torch.LongTensor)
        test_x = Variable(test_x).to(device)

        test_y = test_y.type(torch.LongTensor)
        test_y = Variable(test_y)

        test_capsule= test_model(test_x)
        test_legnth = test_capsule.norm(dim=-1)
        test_pred = torch.max(test_legnth, dim=1)[1]
        test_pred=test_pred.to(device='cpu')
        test_correct = torch.eq(test_pred, test_y)
        test_correct = torch._cast_Double(test_correct)
        test_accuracy = torch.mean(test_correct)
        test_accuracy_list.append(test_accuracy.item())

        print("steps: %d accuracy: %3.3f" % (index_i, test_accuracy.item()))

    test_average_accuracy = np.mean(test_accuracy_list)
    print('average accuracy: %3.3f' % test_average_accuracy)

trian(epochs=10,dataset=dataset,model=model,testdataset=testdataset)

#test(dataset=testdataset)

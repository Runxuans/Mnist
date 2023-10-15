import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms



def getDataLoader(train=False):

    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.13,std=0.30)
    ])

    dataset = MNIST(root="./datasets",
                    train=train,
                    transform= transform_fn,
                    download=True)

    return DataLoader(dataset=dataset,batch_size=5,shuffle=True)




class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28,28)
        self.layer2 = nn.Linear(28,10)

    
    def forward(self,input):
        x = input.view(-1,28*28)
        x = self.layer1(x)
        x = F.relu(x)
        out = self.layer2(x)

        return F.log_softmax(out,dim=-1)
    



def train(model,epochs,optimizer):

    train_dataloader = getDataLoader(train=True)

    for epoch in range(1,epochs+1):

        for idx,(data,target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out,target)
            loss.backward()
            optimizer.step()


            if idx%1000==0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, idx*len(data), len(train_dataloader.dataset),
                    100.*idx/len(train_dataloader), loss.item()
                ))


def test(model):


    loss_list = []
    acc_list  =  []
    test_dataloader = getDataLoader(train=False)
    model.eval()

    with torch.no_grad():
        for data, target in test_dataloader:
            out = model(data)
            cur_loss = F.nll_loss(out,target)
            loss_list.append(cur_loss)

            pred = out.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)


        print(np.mean(acc_list),np.mean(loss_list))





   
  


if __name__=='__main__':

    MNIST_model = MNISTNet()
    optimizer = optim.Adam(params=MNIST_model.parameters(),lr=0.001)
    epoch = 1



    train(model=MNIST_model, epochs=epoch, optimizer=optimizer)
    test(model=MNIST_model)





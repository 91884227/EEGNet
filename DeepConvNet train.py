#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import datetime
from tqdm.notebook import tqdm

DEVICE = "cuda:0"


# # import self-define tool

# In[2]:


from dataloader import read_bci_data
from network_structure import DeepConvNet


# # prepare data loader

# In[3]:


BATCH_SIZE = 64

train_data, train_label, test_data, test_label = read_bci_data()
X_train = torch.FloatTensor(train_data)
y_train = torch.tensor(train_label, dtype=torch.int64)
X_test = torch.FloatTensor(test_data)
y_test = torch.tensor(test_label, dtype=torch.int64)

trainset = Data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = BATCH_SIZE, shuffle = True)
testset = Data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size = BATCH_SIZE, shuffle = True)


# # prediction function

# In[4]:


def get_prediction(data_loader_):
    data_loader = data_loader_
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = net(X)
            _, pred = torch.max(outputs, 1)

            correct = correct + torch.sum(pred == y).item()
            total = total + len(y)
    
    return( correct/total )


# In[ ]:


# prepare NN

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

net = DeepConvNet()
net = net.to(DEVICE)

# loss

criterion = nn.CrossEntropyLoss()

# optimizer

LR = 0.0002
optimizer = optim.Adam(net.parameters(), lr = LR)

EPOCH = 2000
train_acc = []
test_acc = []
loss_record = []
print( datetime.date.today().strftime('%Y-%m-%d %H:%M:%S') )
for epoch in range(EPOCH):
    running_loss = 0.0
    
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(X)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    
    print("==== EPOCH %d/%d ====" % (epoch+1, EPOCH))
    print("loss: %.4f" % running_loss)
    buf1, buf2 = get_prediction(train_loader), get_prediction(test_loader)
    train_acc = train_acc + [buf1]
    test_acc = test_acc + [buf2]
    loss_record = loss_record + [running_loss]
    print("train acc: %.4f ||| test acc: %.4f" % (buf1, buf2))

EEG_ELU_train_acc = train_acc
EEG_ELU_test_acc = test_acc 


# In[10]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("loss(DeepConvNet)")
plt.plot(loss_record)
plt.xlabel("iterations")
plt.legend()
plt.show()


# In[6]:


# prepare NN

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

net = DeepConvNet("ReLU")
net = net.to(DEVICE)

# loss

criterion = nn.CrossEntropyLoss()

# optimizer

LR = 0.0002
optimizer = optim.Adam(net.parameters(), lr = LR)

EPOCH = 200
train_acc = []
test_acc = []
print( datetime.date.today().strftime('%Y-%m-%d %H:%M:%S') )
for epoch in range(EPOCH):
    running_loss = 0.0
    
    for X, y in tqdm(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(X)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    
    print("==== EPOCH %d/%d ====" % (epoch+1, EPOCH))
    print("loss: %.4f" % running_loss)
    buf1, buf2 = get_prediction(train_loader), get_prediction(test_loader)
    train_acc = train_acc + [buf1]
    test_acc = test_acc + [buf2]
    
    print("train acc: %.4f ||| test acc: %.4f" % (buf1, buf2))

EEG_ReLU_train_acc = train_acc
EEG_ReLU_test_acc = test_acc 


# In[7]:


# prepare NN

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

net = DeepConvNet("LeakyReLU")
net = net.to(DEVICE)

# loss

criterion = nn.CrossEntropyLoss()

# optimizer

LR = 0.0002
optimizer = optim.Adam(net.parameters(), lr = LR)

EPOCH = 200
train_acc = []
test_acc = []
print( datetime.date.today().strftime('%Y-%m-%d %H:%M:%S') )
for epoch in range(EPOCH):
    running_loss = 0.0
    
    for X, y in tqdm(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(X)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    
    print("==== EPOCH %d/%d ====" % (epoch+1, EPOCH))
    print("loss: %.4f" % running_loss)
    buf1, buf2 = get_prediction(train_loader), get_prediction(test_loader)
    train_acc = train_acc + [buf1]
    test_acc = test_acc + [buf2]
    
    print("train acc: %.4f ||| test acc: %.4f" % (buf1, buf2))

EEG_LeakyReLU_train_acc = train_acc
EEG_LeakyReLU_test_acc = test_acc 


# In[8]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("Activation function comparision(DeepConvNet)")
plt.plot(EEG_ELU_test_acc, label = "test ELU")
plt.plot(EEG_ReLU_test_acc, label = "test ReLU")
plt.plot(EEG_LeakyReLU_test_acc, label = "test LeakyReLU")
plt.plot(EEG_ELU_train_acc, label = "train ELU")
plt.plot(EEG_ReLU_train_acc, label = "train ReLU")
plt.plot(EEG_LeakyReLU_train_acc, label = "train LeakyReLU")

plt.xlabel("iterations")
plt.ylabel("acc %")
plt.legend()
plt.show()


# In[ ]:





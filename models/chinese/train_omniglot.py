import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from func_omniglot import VAE, Loss, data_folders,get_data_loader, create_task
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
import random

def count_acc(logits,labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.FloatTensor).mean().item()

num_epochs = 5000
learning_rate = 0.001
cnt = 0
n_batch_train = 100 #(way+15)*shot
n_batch_val = 100 #(way+15)*shot 20 or 100

n_train_way = 40
n_val_way = 20 

n_train_shot = 1 
n_val_shot = 1
# MNIST dataset
#train_dataset = torchvision.datasets.MNIST(root='../../sharedLocal/',train=True,transform=transforms.ToTensor(),download=True)
#val_dataset = torchvision.datasets.MNIST(root='../../sharedLocal/',train=False,transform=transforms.ToTensor())

train_folders,test_folders = data_folders(data_folder = '../../sharedLocal/png')

pn_loss_list = []
pn_acc_list = []
val_loss_list = []
val_acc_list = []

torch.cuda.set_device(3)
model = VAE().cuda()
#model = nn.DataParallel(VAE())
#model = model.cuda()
crit = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('training....')
for epoch in range(num_epochs):
    model.train()
    if epoch%10 == 0:
        print("epoch {}...".format(epoch))
    task = create_task(train_folders,n_train_way,n_train_shot,15)
    degrees = random.choice([0])
    support_dataloader = get_data_loader(task,'../../sharedLocal/png',num_per_class=n_train_shot,split="train",shuffle=False,rotation=degrees)
    query_dataloader = get_data_loader(task,'../../sharedLocal/png',num_per_class=15,split="test",shuffle=False,rotation=degrees)

    support,support_labels = support_dataloader.__iter__().next()
    query,query_labels = query_dataloader.__iter__().next()
    total_loss = 0
    total_acc = 0

    x_support,x_support_labels = support.cuda(), support_labels.cuda()
    x_query,x_query_labels = query.cuda(), query_labels.cuda()

    change_support = []
    for j in range(n_train_shot):
        change_support += [i for i in range(j,len(x_support),n_train_shot)]
    x_support = x_support[change_support,:,:,:]
    change_query = []
    for j in range(15):
        change_query += [i for i in range(j,len(x_query),15)]
    x_query = x_query[change_query,:,:,:]

    optimizer.zero_grad()
    embedding_support, recon_support, mu_support, logvar_support = model([x_support,'support'])
    embedding_query, recon_query, mu_query, logvar_query = model([x_query,'query'])
    labels = torch.arange(n_train_way).repeat(15)
    labels = labels.type(torch.cuda.LongTensor)
    loss, logits = crit(embedding_support,recon_support, mu_support, logvar_support, x_support, embedding_query,recon_query, mu_query, logvar_query, x_query, labels)
    accuracy = count_acc(logits, labels)
    total_loss = total_loss + loss.item()
    total_acc += accuracy
    loss.backward()
    optimizer.step()
    pn_loss_list.append(total_loss)
    pn_acc_list.append(total_acc)

    if (epoch)%50 == 0:
        print('validation....')
        val_loss = 0
        val_acc = 0
        for _ in range(100):
            degrees = random.choice([0])
            task = create_task(test_folders,n_val_way,n_val_shot,15)
            #degrees = random.choice([0,90,180,270])
            support_dataloader = get_data_loader(task,'../../sharedLocal/png',num_per_class=n_val_shot,split="train",shuffle=False,rotation=degrees)
            test_dataloader = get_data_loader(task,'../../sharedLocal/png',num_per_class=15,split="test",shuffle=False,rotation=degrees)

            support_images,support_labels = support_dataloader.__iter__().next()
            test_images,test_labels = test_dataloader.__iter__().next()

            x_support,x_support_labels = support_images.cuda(), support_labels.cuda()
            x_query,x_query_labels = test_images.cuda(), test_labels.cuda()

            change_support = []
            for j in range(n_val_shot):
                change_support += [i for i in range(j,len(x_support),n_val_shot)]
            x_support = x_support[change_support,:,:,:]
            change_query = []
            for j in range(15):
                change_query += [i for i in range(j,len(x_query),15)]
            x_query = x_query[change_query,:,:,:]

            embedding_support, recon_support, mu_support, logvar_support = model([x_support,'support'])
            embedding_query, recon_query, mu_query, logvar_query = model([x_query,'query'])
            labels = torch.arange(n_val_way).repeat(15)
            labels = labels.type(torch.cuda.LongTensor)
#            print(embedding_support.shape,recon_support.shape,mu_support.shape,x_support.shape)
#            print(embedding_query.shape,recon_query.shape,mu_query.shape,x_query.shape)
            loss, logits = crit(embedding_support,recon_support, mu_support, logvar_support, x_support, embedding_query,recon_query, mu_query, logvar_query, x_query, labels)
            acc = count_acc(logits, labels)
            val_loss += loss.item()
            val_acc += acc

        val_loss_list.append(val_loss/100)
        val_acc_list.append(val_acc/100)
        print(pn_acc_list[-1],val_acc_list[-1])

with open('pn_loss_list.txt','w') as f:
    for item in pn_loss_list:
        f.write('%s\n'%item)
with open('pn_acc_list.txt','w') as f:
    for item in pn_acc_list:
        f.write('%s\n'%item)
with open('val_loss_list.txt','w') as f:
    for item in val_loss_list:
        f.write('%s\n'%item)
with open('val_acc_list.txt','w') as f:
    for item in val_acc_list:
        f.write('%s\n'%item)

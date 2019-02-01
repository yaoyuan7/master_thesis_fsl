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

num_epochs = 2500
learning_rate = 0.001
cnt = 0
n_batch_train = 100 #(way+15)*shot
n_batch_val = 100 #(way+15)*shot 20 or 100

n_train_way = 60
n_val_way = 10

n_train_shot = 1 
n_val_shot = 20
# MNIST dataset
#train_dataset = torchvision.datasets.MNIST(root='../../sharedLocal/',train=True,transform=transforms.ToTensor(),download=True)
#val_dataset = torchvision.datasets.MNIST(root='../../sharedLocal/',train=False,transform=transforms.ToTensor())

train_folders,test_folders = data_folders(data_folder = '../../sharedLocal/omniglot_resized')

torch.cuda.set_device(3)
model = VAE().cuda()
crit = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('training....')
for epoch in range(num_epochs):
    model.train()
    if epoch%10 == 0:
        print("epoch {}...".format(epoch))
    task = create_task(train_folders,n_train_way,n_train_shot,15)
    degrees = random.choice([0,90,180,270])
    support_dataloader = get_data_loader(task,'../../sharedLocal/omniglot_resized',num_per_class=n_train_shot,split="train",shuffle=False,rotation=degrees)
    query_dataloader = get_data_loader(task,'../../sharedLocal/omniglot_resized',num_per_class=15,split="test",shuffle=False,rotation=degrees)

    support,support_labels = support_dataloader.__iter__().next()
    query,query_labels = query_dataloader.__iter__().next()

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
    loss.backward()
    optimizer.step()

task = create_task(test_folders,n_val_way,n_val_shot,15)
degrees = random.choice([0])
support_dataloader = get_data_loader(task,'../../sharedLocal/omniglot_resized',num_per_class=n_val_shot,split="train",shuffle=False,rotation=degrees)
support_images,support_labels = support_dataloader.__iter__().next()
x_support = support_images.cuda()

embedding_support, recon_support, mu_support, logvar_support = model([x_support,'query'])

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.cm as cm
data = mu_support.cpu().detach().numpy()
n_sample,n_feature = data.shape
label = support_labels

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    sns.set()
    fig = plt.figure(figsize=(25,25))
    ax = plt.subplot(111)
    colors=sns.color_palette("deep", 15)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1],s=8,c=colors[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.axis('off')
    fig.savefig('test.jpg',bbox_inches='tight')
    return fig

tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(data)
fig = plot_embedding(result, label,'')
print(test_folders,label)

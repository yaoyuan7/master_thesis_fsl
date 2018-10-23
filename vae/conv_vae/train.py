import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from func import VAE, Loss, dataset_sampler,miniimagenet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np


def count_acc(logits,labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.FloatTensor).mean().item()

num_epochs = 1000
learning_rate = 0.001
cnt = 0
n_batch_train = 20
n_batch_val = 32

n_train_way = 5
n_val_way = 5

n_train_shot = 5
n_val_shot = 5
# MNIST dataset
#train_dataset = torchvision.datasets.MNIST(root='../../sharedLocal/',train=True,transform=transforms.ToTensor(),download=True)
#val_dataset = torchvision.datasets.MNIST(root='../../sharedLocal/',train=False,transform=transforms.ToTensor())

train_dataset = miniimagenet('train')
train_sampler = dataset_sampler(train_dataset.label, n_batch_train, n_train_way, 5 + 15)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=train_sampler, pin_memory=True)

val_dataset = miniimagenet('val')
val_sampler = dataset_sampler(val_dataset.label, n_batch_val, n_val_way, 5 + 15)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_sampler=val_sampler, pin_memory=True)


train_loss_list = []
val_loss_list = []

model = VAE().cuda()
crit = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()

for epoch in range(num_epochs):
    _loss = []
    __loss = []
    print("epoch {}...".format(epoch))
    print('training....')
    for batch_idx, (data, label) in enumerate(train_loader):
        data,label = data.cuda(), label.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = crit(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()
        _loss.append(loss.item())

    if epoch % 100 == 0:
        print('Iter-{}; Loss: {:.4}'.format(batch_idx, loss.item()))
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(recon_batch[0:25]):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            #plt.imshow(sample.data.cpu().numpy(),cmap='Greys_r')
            plt.imshow(np.transpose(sample.data.cpu().numpy(),(1,2,0)))#.reshape(28,28,3),cmap='Greys_r'
        if not os.path.exists('out_train/'):
            os.makedirs('out_train/')
        plt.savefig('out_train/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        print('save output....')
        plt.close(fig)
'''
    print('validation....')
    for batch_idx, (data, label) in enumerate(val_loader):
        data,label = data.cuda(), label.cuda()
        data = Variable(data)
        recon_batch, mu, logvar = model(data)
        loss = crit(recon_batch, data, mu, logvar)
        __loss.append(loss.item())

    if epoch % 20 == 0:
        print('Iter-{}; Loss: {:.4}'.format(batch_idx, loss.item()))
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(recon_batch[0:25]):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            #plt.imshow(sample.data.cpu().numpy(),cmap='Greys_r')
            plt.imshow(np.transpose(sample.data.cpu().numpy(),(1,2,0)))#.reshape(28,28,3),cmap='Greys_r'
        if not os.path.exists('out_val/'):
            os.makedirs('out_val/')
        plt.savefig('out_val/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)

    train_loss_list.append(np.mean(_loss))
    val_loss_list.append(np.mean(__loss))
'''
#torch.save(model.state_dict(), os.path.join('./model/','epoch-{}'.format(epoch) + '.pth'))
#with open('train_loss.txt','w') as f:
#    for item in train_loss_list:
#        f.write('%s\n'%item)

#with open('val_loss.txt','w') as f:
#    for item in val_loss_list:
#        f.write('%s\n'%item)






        # if batch_idx % 50 == 0:
        #     print('Iter-{}; Loss: {:.4}'.format(batch_idx, loss.item()))
        #     fig = plt.figure(figsize=(4, 4))
        #     gs = gridspec.GridSpec(4, 4)
        #     gs.update(wspace=0.05, hspace=0.05)
        #     for i, sample in enumerate(recon_batch[0:16]):
        #         ax = plt.subplot(gs[i])
        #         plt.axis('off')
        #         ax.set_xticklabels([])
        #         ax.set_yticklabels([])
        #         ax.set_aspect('equal')
        #         plt.imshow(sample.data.numpy().reshape(28, 28), cmap='Greys_r')
        #
        #     if not os.path.exists('out/'):
        #         os.makedirs('out/')
        #
        #     plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        #     cnt += 1
        #     plt.close(fig)



    #print("epoch {}: - loss: {}".format(epoch, np.mean(loss_list)))
#with open('loss.txt','w') as f:
#    for item in loss_list:
#        f.write('%s\n'%item)
#with open('accuracy.txt','w') as f:
#    for item in accuracy_list:
#        f.write('%s\n'%item)

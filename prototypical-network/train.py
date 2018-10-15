import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from func import Convnet, miniimagenet, dataset_sampler
from utils import calculate_dist

n_batch_train = 100
n_batch_val = 400
n_way = 5
n_shot = 5
n_query = 15
max_epoch = 1000

def count_acc(logits,label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

if __name__ == '__main__':


    model = Convnet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    train_set = miniimagenet('train')
    train_sampler = dataset_sampler(train_set.label, n_batch_train, n_way, n_shot + n_query)
    train_loader = DataLoader(dataset=train_set,batch_sampler=train_sampler,num_workers=8, pin_memory=True)

    val_set = miniimagenet('val')
    val_sampler = dataset_sampler(val_set.label, n_batch_val, n_way, n_shot + n_query)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler,num_workers=8, pin_memory=True)

    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for epoch in range(max_epoch):
        lr_scheduler.step()
        model.train()
        for step, batch in enumerate(train_loader, 1):
            data, label = [_.cuda() for _ in batch]
            support_data, query_data = data[:n_way*n_shot], data[n_way*n_shot:]

            support_embedding, query_embedding = model(support_data), model(query_data)
            support_embedding = support_embedding.reshape(n_shot, n_way, -1).mean(dim=0)
            #labels = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
            #labels = labels.cuda()
            labels = torch.arange(n_way).repeat(n_query)
            labels = labels.type(torch.cuda.LongTensor)
            #support_embedding: [5,1600], query_embedding: [75,1600]
            #support_embedding:[1,2,3,4,5], query_embedding: [12345,12345,...,12345]
            logits = calculate_dist(query_embedding,support_embedding)
            #log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
            #loss = -log_p_y.gather(2, labels).squeeze().view(-1).mean()
            loss = F.cross_entropy(logits,labels)
            acc = count_acc(logits, labels)
            train_loss.append(loss.item())
            train_accuracy.append(acc)
#            dist = calculate_dist(query_embedding,support_embedding)
#            log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
#            loss_val = -log_p_y.gather(2, labels).squeeze().view(-1).mean()
#            _, y_hat = log_p_y.max(2)
#            acc = torch.eq(y_hat, labels.squeeze()).float().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step()

        if (epoch)%100 == 0:
            for step, batch in enumerate(val_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                support_data, query_data = data[:n_way*n_shot], data[n_way*n_shot:]
                support_embedding, query_embedding = model(support_data), model(query_data)
                support_embedding = support_embedding.reshape(n_shot, n_way, -1).mean(dim=0)

                labels = torch.arange(n_way).repeat(n_query)
                labels = labels.type(torch.cuda.LongTensor)

                logits = calculate_dist(query_embedding,support_embedding)
                loss = F.cross_entropy(logits, labels)
                acc = count_acc(logits, labels)
                val_loss.append(loss.item())
                val_accuracy.append(acc)
        print(np.mean(train_loss[epoch:epoch+n_batch_train]),np.mean(val_loss[epoch:epoch+n_batch_val]))
#            print("epoch: ",epoch, "train_loss: ", np.mean(train_loss[epoch:epoch+n_batch_train]),\
#                                    "val_loss: ", np.mean(val_loss[epoch:epoch+n_batch_val]),\
#                                    "train_accuracy: ",np.mean(train_accuracy[epoch:epoch+n_batch_train]),\
#                                    "val_accuracy: ",np.mean(val_accuracy[epoch:epoch+n_batch_val]))

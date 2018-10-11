import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import utils
from func import *

import os
import math
import argparse
import random

parser = argparse.ArgumentParser(description="relation network for few shot learning")
parser.add_argument("-d","--directory",type = str, default = '../datas/omniglot_resized')
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--support_num",type = int, default = 5)
parser.add_argument("-b","--query_num",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-t","--test_episode", type = int, default = 400)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=2)
args = parser.parse_args()

DATA_FOLDER = args.directory
CLASS_NUM = args.class_num
SUPPORT_NUM = args.support_num
QUERY_NUM = args.query_num
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
dataset_name = DATA_FOLDER.split('/')[-1]

def main():
    print("init data folders")
    train_folders,test_folders = data_folders(data_folder = DATA_FOLDER)

    print("init neural networks")
    if dataset_name == 'omniglot_resized':
        encoder = utils.CNNEncoder(1)
        decoder = utils.RelationNetwork(1,64,8)
    else:
        encoder = utils.CNNEncoder(3)
        decoder = utils.RelationNetwork(3,64,8)

    encoder.apply(weights_init)
    decoder.apply(weights_init)
    encoder.cuda(GPU)
    decoder.cuda(GPU)

    encoder_optim = torch.optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
    decoder_optim = torch.optim.Adam(decoder.parameters(),lr=LEARNING_RATE)
    encoder_scheduler = StepLR(encoder_optim,step_size=100000,gamma=0.5)
    decoder_scheduler = StepLR(decoder_optim,step_size=100000,gamma=0.5)

    encoder_save_path = "./models/" + dataset_name + "_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM) +"shot.pkl"
    decoder_save_path = "./models/" + dataset_name + "_decoder_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM) +"shot.pkl"

    if os.path.exists(encoder_save_path):
        encoder.load_state_dict(torch.load(encoder_save_path))
        print("load feature encoder success")
    if os.path.exists(decoder_save_path):
        decoder.load_state_dict(torch.load(decoder_save_path))
        print("load relation network success")


    print("Training...")
    last_accuracy = 0.0
    losses = []
    accuracy = []
    for episode in range(EPISODE):

        encoder_scheduler.step(episode)
        decoder_scheduler.step(episode)

        degrees = random.choice([0])
        task = utils.create_task(train_folders,CLASS_NUM,SUPPORT_NUM,QUERY_NUM)
        support_dataloader = get_data_loader(task,dataset_name,num_per_class=SUPPORT_NUM,split="train",shuffle=False,rotation=degrees)
        query_dataloader = get_data_loader(task,dataset_name,num_per_class=QUERY_NUM,split="test",shuffle=True,rotation=degrees)

        support,support_labels = support_dataloader.__iter__().next()
        query,query_labels = query_dataloader.__iter__().next()

        if dataset_name == 'omniglot_resized':
            support_features = encoder(Variable(support).cuda(GPU))
            support_features = support_features.view(CLASS_NUM,SUPPORT_NUM,64,5,5)
            support_features = torch.sum(support_features,1).squeeze(1)
            query_features = encoder(Variable(query).cuda(GPU))
            support_features_ext = support_features.unsqueeze(0).repeat(QUERY_NUM*CLASS_NUM,1,1,1,1)
            query_features_ext = query_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
            query_features_ext = torch.transpose(query_features_ext,0,1)
            relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,64*2,5,5)
            relations = decoder(relation_pairs).view(-1,CLASS_NUM)

        else:
            support_features = encoder(Variable(support).cuda(GPU))
            support_features = support_features.view(CLASS_NUM,SUPPORT_NUM,64,19,19)
            support_features = torch.sum(support_features,1).squeeze(1)
            query_features = encoder(Variable(query).cuda(GPU))
            support_features_ext = support_features.unsqueeze(0).repeat(QUERY_NUM*CLASS_NUM,1,1,1,1)
            query_features_ext = query_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
            query_features_ext = torch.transpose(query_features_ext,0,1)
            relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,64*2,19,19)
            relations = decoder(relation_pairs).view(-1,CLASS_NUM)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(QUERY_NUM*CLASS_NUM, CLASS_NUM).scatter_(1, query_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)
        losses.append(loss.item())

        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(),0.5)
        encoder_optim.step()
        decoder_optim.step()

        if (episode+1)%100 == 0:
            print("episode:",episode+1,"loss",loss.item())

        if (episode+1)%TEST_EPISODE == 0:
            print("Testing...")
            total_rewards = 0
            #train_folders,test_folders = data_folders(data_folder = DATA_FOLDER)
            for _ in range(int(TEST_EPISODE//4)):
                degrees = random.choice([0])
                task = utils.create_task(test_folders,CLASS_NUM,SUPPORT_NUM,SUPPORT_NUM,)
                support_dataloader = get_data_loader(task,dataset_name,num_per_class=SUPPORT_NUM,split="train",shuffle=False,rotation=degrees)
                test_dataloader = get_data_loader(task,dataset_name,num_per_class=SUPPORT_NUM,split="test",shuffle=True,rotation=degrees)

                support_images,support_labels = support_dataloader.__iter__().next()
                test_images,test_labels = test_dataloader.__iter__().next()

                if dataset_name == 'omniglot_resized':
                    support_features = encoder(Variable(support_images).cuda(GPU))
                    support_features = support_features.view(CLASS_NUM,SUPPORT_NUM,64,5,5)
                    support_features = torch.sum(support_features,1).squeeze(1)
                else:
                    support_features = encoder(Variable(support_images).cuda(GPU))
                    support_features = support_features.view(CLASS_NUM,SUPPORT_NUM,64,19,19)
                    support_features = torch.sum(support_features,1).squeeze(1)

                test_features = encoder(Variable(test_images).cuda(GPU))

                support_features_ext = support_features.unsqueeze(0).repeat(SUPPORT_NUM*CLASS_NUM,1,1,1,1)
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
                test_features_ext = torch.transpose(test_features_ext,0,1)

                if dataset_name == 'omniglot_resized':
                    relation_pairs = torch.cat((support_features_ext,test_features_ext),2).view(-1,64*2,5,5)
                    relations = decoder(relation_pairs).view(-1,CLASS_NUM)
                else:
                    relation_pairs = torch.cat((support_features_ext,test_features_ext),2).view(-1,64*2,19,19)
                    relations = decoder(relation_pairs).view(-1,CLASS_NUM)

                _,predict_labels = torch.max(relations.data,1)

                rewards = [1 if predict_labels[j].type(torch.LongTensor)==test_labels[j] else 0 for j in range(CLASS_NUM*SUPPORT_NUM)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards/CLASS_NUM/SUPPORT_NUM/TEST_EPISODE*4
            accuracy.append(test_accuracy)
            print("test accuracy:",test_accuracy)

            if test_accuracy > last_accuracy:
                torch.save(encoder.state_dict(),encoder_save_path)
                torch.save(decoder.state_dict(),decoder_save_path)
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
    with open('accuracy.txt','w') as f:
        for item in accuracy:
            f.write('{}\n'.format(item))
    with open('loss.txt','w') as f:
        for item in losses:
            f.write('{}\n'.format(item))
if __name__ == '__main__':
    main()

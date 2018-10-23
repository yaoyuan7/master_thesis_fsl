import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset

class miniimagenet(Dataset):
    def __init__(self, setname):
        ROOT_PATH = '../../sharedLocal/miniimagenet/'
        csv_path = os.path.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor()])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class dataset_sampler():
    def __init__(self, label, n_batch, n_cls, n_per,shuffle=True):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(21 * 21 * 64, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc21 = nn.Linear(21 * 21 * 64, 256)
        self.fc22 = nn.Linear(21 * 21 * 64, 256)
        #self.fc21 = nn.Linear(128, 128)
        #self.fc22 = nn.Linear(128, 128)

        # Decoder
        self.fc3 = nn.Linear(256, 256)
        self.fc_bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 21 * 21 * 64)
        self.fc_bn4 = nn.BatchNorm1d(21 * 21 * 64)

        self.conv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encode(self, x_support,x_query):
        conv1_support = self.relu(self.bn1(self.conv1(x_support)))
        conv2_support = self.relu(self.bn2(self.conv2(conv1_support)))
        conv3_support = self.relu(self.bn3(self.conv3(conv2_support)))
        conv4_support = self.relu(self.bn4(self.conv4(conv3_support))).view(-1, 21 * 21 * 64) #[batch,16,7,7]

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4_support)))

        conv1_query = self.relu(self.bn1(self.conv1(x_query)))
        conv2_query = self.relu(self.bn2(self.conv2(conv1_query)))
        conv3_query = self.relu(self.bn3(self.conv3(conv2_query)))
        conv4_query = self.relu(self.bn4(self.conv4(conv3_query))).view(-1, 21 * 21 * 64) #[batch,16,7,7]

        return self.fc21(conv4_support), self.fc22(conv4_support), conv4_support.view(conv4_support.size(0), -1), conv4_query.view(conv4_query.size(0), -1)
        #return self.fc21(fc1), self.fc22(fc1), conv4_support.view(conv4_support.size(0), -1), conv4_query.view(conv4_query.size(0), -1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def calculate_dist(self, a, b):
        way = int(b.shape[0]/5)
        shot = 5
        b = b.reshape(shot, way, -1).mean(dim=0)
        a = a.unsqueeze(1).expand(a.shape[0], b.shape[0], -1)
        b = b.unsqueeze(0).expand(a.shape[0], b.shape[0], -1)
        logits = -torch.pow(a-b,2).sum(2)
        return logits

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 64, 21, 21)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1,3, 84, 84)

    def forward(self, x_support,x_query):
        mu, logvar, support_embedding, query_embedding = self.encode(x_support,x_query)
        z = self.reparameterize(mu, logvar)
        logits = self.calculate_dist(query_embedding, support_embedding)
        return self.decode(z), mu, logvar, logits


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, recon_x, x, mu, logvar,logits, labels):
        MSE = self.mse_loss(recon_x, x)
        CE = torch.nn.functional.cross_entropy(logits,labels)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD 
'''
class Trainer:
    def __init__(self, model, loss, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = loss
        self.optimizer = self.get_optimizer()

    def train(self):
        self.model.train()
        for epoch in range(5):
            loss_list = []
            print("epoch {}...".format(epoch))
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = Variable(data)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss(recon_batch, data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.data[0])

            print("epoch {}: - loss: {}".format(epoch, np.mean(loss_list)))

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
'''

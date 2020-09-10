from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
#from pytorch_memlab import profile, set_target_gpu, MemReporter


class PointMaxPool(nn.Module):
    def __init__(self):
        super(PointMaxPool, self).__init__()


    def forward(self, x):
        x = torch.max(x, 2, keepdim=True)[0]
        return x


class PointMeanPool(nn.Module):
    def __init__(self, mean_pool=0):
        super(PointMeanPool, self).__init__()


    def forward(self, x):
        x = torch.mean(x, 2, keepdim=True)
        return x


class PointMeanMaxPool(nn.Module):
    def __init__(self, mean_pool=0):
        super(MeanMaxPool, self).__init__()
        self.mean_pool_split = mean_pool


    def forward(self, x):
        x1 = torch.mean(x[:,:self.mean_pool_split,:], 2, keepdim=True)
        x2 = torch.max(x[:,self.mean_pool_split:,:], 2, keepdim=True)[0]
        x = torch.cat((x1, x2), 1)
        return x


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
#        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#        self.fc1 = nn.Linear(1024, 512)
#        self.fc2 = nn.Linear(512, 256)
#        self.fc3 = nn.Linear(256, 9)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
#        self.bn3 = nn.BatchNorm1d(1024)
#        self.bn4 = nn.BatchNorm1d(512)
#        self.bn5 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        #x = torch.max(x, 2, keepdim=True)[0]
        x = torch.mean(x, 2, keepdim=True)
#        x = x.view(-1, 1024)
        x = x.view(-1, 256)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64, mean_pool = 0):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
#        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#        self.fc1 = nn.Linear(1024, 512)
#        self.fc2 = nn.Linear(512, 256)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
#        self.bn3 = nn.BatchNorm1d(1024)
#        self.bn4 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k
        if mean_pool <= 0:
            self.pool = PointMaxPool()
        elif mean_pool >= 1024:
            self.pool = PointMeanPool()
        else:
            self.pool = PointMeanMaxPool(mean_pool)
        print("using", self.pool, "in STNkd, k =", k)

#    @profile
    def forward(self, x):
#        reporter = MemReporter()
#        print("==========  1  ==========")
#        reporter.report()
        batchsize = x.size()[0]
#        sumhits = (x[:,0,:]!=0).sum(-1, keepdim=True).float()
        x = F.relu(self.bn1(self.conv1(x)))
#        print("==========  2  ==========")
#        reporter.report()
        x = F.relu(self.bn2(self.conv2(x)))
#        print("==========  3  ==========")
#        reporter.report()
        x = F.relu(self.bn3(self.conv3(x)))
#        print("==========  4  ==========")
#        reporter.report()
        x = self.pool(x)
#        print("==========  5  ==========")
#        reporter.report()
        #x = torch.max(x, 2, keepdim=True)[0]
        #x = torch.mean(x, 2, keepdim=True)
#        x = x.view(-1, 1024)
        x = x.view(-1, 256)
#        print("==========  6  ==========")
#        reporter.report()

        x = F.relu(self.bn4(self.fc1(x)))
#        print("==========  7  ==========")
#        reporter.report()
        x = F.relu(self.bn5(self.fc2(x)))
#        print("==========  8  ==========")
#        reporter.report()
#        x = torch.cat((x, sumhits), 1)
        x = self.fc3(x)
#        print("==========  9  ==========")
#        reporter.report()

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, mean_pool = 0):
        super(PointNetfeat, self).__init__()
#        self.stn = STN3d()
        k=5
        self.stn = STNkd(k=k, mean_pool=mean_pool)
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
#        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
#        self.bn3 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if mean_pool <= 0:
            self.pool = PointMaxPool()
        elif mean_pool >= 1024:
            self.pool = PointMeanPool()
        else:
            self.pool = PointMeanMaxPool(mean_pool)
        print("using", self.pool, "in PointNetfeat")
        if self.feature_transform:
            self.fstn = STNkd(k=64, mean_pool=mean_pool)

#    @profile
    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        if self.global_feat:
            pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
#        if self.mean_pool:
#            x = torch.mean(x, 2, keepdim=True)
#        else:
#            x = torch.max(x, 2, keepdim=True)[0]
        x = self.pool(x)
        #x = torch.max(x, 2, keepdim=True)[0]
        #x = torch.mean(x, 2, keepdim=True)
#        x = x.view(-1, 1024)
        x = x.view(-1, 256)
        if self.global_feat:
            return x, trans, trans_feat
        else:
#            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False, mean_pool=0):
#        set_target_gpu(6)
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, mean_pool=mean_pool)
#        self.fc1 = nn.Linear(1024, 512)
#        self.fc2 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
#        self.bn1 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        #sumhits = (x[:,0,:] != 0).sum(-1, keepdim=True).float()
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        #x = torch.cat((x, sumhits), 1)
        x = self.fc3(x)
        return x


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    I = I.to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 8)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())

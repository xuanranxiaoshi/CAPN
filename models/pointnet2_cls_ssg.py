import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_cls_ssg_util import  PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3     # 带有法线特征，输入channel则为6
        self.normal_channel = normal_channel

        """全局位置编码"""
        self.mlp0 = nn.Sequential(
            nn.Conv1d(in_channel,32,1),
            nn.BatchNorm1d(32)
        )

        """层次化点云特征提取模块"""
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=32, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128,256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024],  group_all=True)

        """最后多层感知机处理全局特征得到识别结果"""
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.5)#0.4
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape         # B*N*C
        xyz = xyz.transpose(2, 1)
        if self.normal_channel:
            l0_xyz = xyz[:, :3, :]  # 坐标
            """是否是全局空间编码过程"""
            l0_points = self.mlp0(xyz)  # 坐标和特征机型卷积操作，B*32*N

        else:
            l0_xyz = xyz
            l0_points = self.mlp0(xyz)

        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
     
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x,None



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, gold, trans_feat, smoothing=True):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

if __name__=="__main__":
    data = torch.rand(10,3,1024).cuda()
    model = get_model(40,False).cuda()
    out = model(data)
    print(out.shape)
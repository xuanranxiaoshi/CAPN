import torch
import torch.nn as nn
from torch.autograd import Function
import pointnet2_ops._ext as _ext
import torch.nn.functional as F
from models.pointnet_util import *


class eca_layer(nn.Module):
    """Constructs a ECA module.
    论文中的特征提取函数 E
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, in_channel, out_channel, k_size=3):
        super(eca_layer, self).__init__()  # super类的作用是继承的时候，调用含super的哥哥的基类__init__函数。
        # AdaptiveAvgPool2d（二元自适应均值汇聚层）
        # AdaptiveAvgPool2d(1)
        # torch.Size([2, 32, 16, 16])----->torch.Size([2, 32, 1, 1])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)  # 一维卷积
        self.sigmoid = nn.Sigmoid()
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()  # b代表b个样本，c为通道数，h为高度，w为宽度

        # feature descriptor on the global spatial information
        # 相当于每个点云集合的全局特征
        y = self.avg_pool(x)  # b*c*1*1

        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        # y.squeeze(-1)——>B*C*1——>B*1*C——>B*1*C——>B*C*1——>B*C*1*1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)  # B*C*1*1  归一化，每个通道的权重
        # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算
        # y.expand_as(x) y拓展为x的size, B*C*H*W， 是将C中的数在后两个维度进行重复
        x = x * y.expand_as(x)  # 乘以权重
        x = self.mlp1(x)
        return x


class _PositionAttentionModule(nn.Module):
    """对应于论文中局部位置编码部分，属于注意力机制的一部分"""
    def __init__(self, channel=3, ratio=8):
        super(_PositionAttentionModule, self).__init__()
        self.b_channel = max(8, channel // ratio)   # 8
        self.mlp0 = nn.Sequential(
            nn.Conv2d(3, self.b_channel, 1),        # B*b_channel*npoint*nsample
            nn.BatchNorm2d(self.b_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.b_channel, channel, 1),  # B*in_channel*npoint*nsample
            nn.BatchNorm2d(channel)
        )

        self.mlp1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.BatchNorm2d(channel)
        )

    def forward(self, pos, x):
        """
        pos: (B, C=3, npoint, nsample) 各邻域点集坐标信息
        x  : (B, D, npoint, nsample) 各点的特征信息
        """
        pos = self.mlp0(pos)    # MLP
        """将点云坐标直接作为点云的位置编码； """
        x = x + pos
        x = self.mlp1(x)        # 在进行一个MLP编码     (B, D, npoint, nsample)
        return x


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel       # 点的特征维度
        for out_channel in mlp[:-1]:    # 只到倒数第二层
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        """增加注意力和eca层"""
        self.attn = _PositionAttentionModule(in_channel)
        self.eca = eca_layer(mlp[-2], mlp[-1])  # mlp的最后一层

    def forward(self, xyz, points):
        B, C, N = xyz.shape
        xyz = xyz.contiguous()
        if points is not None:
            points = points.contiguous()
        xyz_flipped = xyz.transpose(1, 2).contiguous()  # B*N*C

        """1. 采样与分组"""
        if self.group_all:  # 将所有点作为一个邻域
            new_xyz = torch.zeros(B, C, 1).cuda()
            grouped_xyz = xyz.view(B, C, 1, N).contiguous()
            if points is not None:
                # new_features = torch.cat([grouped_xyz, points.view(B, -1, 1,N)], dim=1)
                """这里没有拼接坐标信息"""
                new_features = points.view(B, -1, 1, N)
            else:
                new_features = grouped_xyz  # 没有额外的特征直接返回坐标
        else:
            fps_idx = furthest_point_sample(xyz_flipped, self.npoint)   # (B, npoint) 中心点索引
            new_xyz = gather_operation(xyz, fps_idx)    # (B, C, npoint) 提取出中心点坐标

            # xyz_gumbel = torch.norm(xyz_flipped,dim=-1)
            # idx = torch.sort(F.gumbel_softmax(xyz_gumbel, tau=1, hard=False))[1][:,::2]
            # new_xyz = gather_operation(xyz, idx.int())

            new_xyz_flipped = new_xyz.transpose(1, 2).contiguous()  # B*npoint*C

            idx = ball_query(self.radius, self.nsample, xyz_flipped, new_xyz_flipped)   # 各邻域采样点的索引 (B, npoint, nsample)
            # grouping_operation(xyz, idx) 提取出邻域中的每个点，(B, C, npoint, nsample)

            """对应论文中P_center 和 P_neighbor 之间的关系"""
            grouped_xyz = grouping_operation(xyz, idx) - new_xyz.unsqueeze(-1)

            if points is not None:
                points_flipped = points.transpose(1, 2).contiguous()    # B*N*D
                grouped_points = grouping_operation(points, idx).contiguous()   # # (B, D, npoint, nsample)
                # new_features = torch.cat([grouped_xyz, grouped_points], dim=1)  # [B,C+D, npoint, nsample]
                """这里也没有拼接坐标"""
                new_features = grouped_points
            else:
                new_features = grouped_xyz

        """2. 基于采样后点云空间坐标和点云特征进行局部位置编码"""
        new_features = self.attn(grouped_xyz, new_features) #  (B, D, npoint, nsample)

        """3. ，提取点云特征"""
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features = F.relu(bn(conv(new_features)))   #  (B, D'', npoint, nsample)

        """4. 基于点云的通道注意力提取点云特征"""
        new_features = self.eca(new_features)               # (B, D', npoint, nsample)

        """5. 最大池化"""
        new_features = torch.max(new_features, -1)[0]       # (B, D', npoint)

        return new_xyz, new_features


if __name__ == "__main__":
    model = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128],
                                   group_all=False).cuda()
    data = torch.zeros(10, 3, 1024).cuda()
    feature = torch.zeros(10, 3, 1024).cuda()
    out = model(data, feature)

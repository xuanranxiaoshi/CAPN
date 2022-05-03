import torch
import torch.nn as nn
from torch.autograd import Function
import pointnet2_ops._ext as _ext
import torch.nn.functional as F


def sample_gumbel1(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax1(logits, dim, k=1, rand=False, temperature=1):
    """
    ST-gumple-softmax w/o random gumbel samplings
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    if rand:
        logits = logits + sample_gumbel(logits.size())
    y = F.softmax(logits / temperature, dim=dim)
    shape = y.size()
    _, ind = torch.topk(y, k, dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, k), 1)
    y_hard = y_hard.view(*shape)

    y_hard = (y_hard - y).detach() + y
    return ind, y_hard


def gumbel_softmax2(logits, dim, k=1, stride=1, rand=False, temperature=1):
    """
    ST-gumple-softmax w/o random gumbel samplings
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    if rand:
        logits = logits + sample_gumbel(logits.size())
    y = F.softmax(logits / temperature, dim=dim)
    shape = y.size()
    ind = torch.sort(y, dim=-1)[1][:, ::stride]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, k), 1)
    y_hard = y_hard.view(*shape)

    y_hard = (y_hard - y).detach() + y
    return ind, y_hard


def knn1(x, k):
    B, _, N = x.size()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def knn2(x, y, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), y)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -yy - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def knn3(x, k):
    k = k + 1
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


# def square_distance(src, dst):
#     """
#     Calculate Euclid distance between each two points.
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist

# def knn_point(nsample, xyz, new_xyz):
#     """
#     Input:
#         nsample: max sample number in local region
#         xyz: all points, [B, N, C]
#         new_xyz: query points, [B, S, C]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     sqrdists = square_distance(new_xyz, xyz)
#     _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
#     return group_idx
class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint, dim=None):
        # todo ctx 是什么
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        # todo 这里调用的是c++代码？
        if dim is not None:
            out = _ext.furthest_point_sampling2(xyz, npoint)
        else:
            out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor 为点云特征
            (B, C, N) tensor

        idx : torch.Tensor      为采样中心点的索引
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor            提取出采样中心点
            (B, C, npoint) tensor
        """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor     为点云特征
            (B, C, N) tensor of features to group
        idx : torch.Tensor          为每个邻域的采样点集
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor                为每个邻域提取出特征
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz, fps_idx=None, dim=None):
        # type: (Any, float, int, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        采样函数
        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        if fps_idx is not None:
            if dim is not None:
                idx = _ext.ball_query3(new_xyz, xyz, radius, nsample, fps_idx)
            else:
                idx = _ext.ball_query2(new_xyz, xyz, radius, nsample, fps_idx)
            output = torch.cat([fps_idx.unsqueeze(2), idx], dim=2)
        else:
            output = _ext.ball_query(new_xyz, xyz, radius, nsample)

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
            为know中的每个点在unknown中找到最近的三个点
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            对三个点的权重进行线性插值
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features  插值后的特征
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        """一维卷积实现全连接"""
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        B, C, N = xyz.shape
        # contiguous() → Tensor 返回一个内存连续的有相同数据的 tensor，如果原 tensor 内存连续则返回原 tensor
        xyz = xyz.contiguous()
        points = points.contiguous()
        xyz_flipped = xyz.transpose(1, 2).contiguous()  # B*N*C
        if self.group_all:
            grouped_xyz = xyz.view(B, C, 1, N).contiguous()  # 全部点作为一个邻域
            if points is not None:
                new_features = torch.cat([grouped_xyz, points.view(B, -1, 1, N)], dim=1)  # 点的坐标和原特征进行拼接获得新的特征
            else:
                new_features = grouped_xyz
        else:  # 采样
            # furthest_point_sample(xyz_flipped, self.npoint) 采样得到中心点索引， (B, npoint)
            # gather_operation(xyz, furthest_point_sample(xyz_flipped, self.npoint)) , 提取出中心点，(B, C, npoint)

            new_xyz = (
                gather_operation(
                    xyz, furthest_point_sample(xyz_flipped, self.npoint)
                ))
            new_xyz_flipped = new_xyz.transpose(1, 2).contiguous()  # B*npoint*3

            idx = ball_query(self.radius, self.nsample, xyz_flipped, new_xyz_flipped)  # 获得各领域中点的索引，(B, npoint, nsample)
            # grouping_operation(xyz, idx) 提取出邻域中的每个点，(B, C, npoint, nsample)
            grouped_xyz = grouping_operation(xyz, idx) - new_xyz.unsqueeze(-1)  # 减去中心点坐标

            if points is not None:
                points_flipped = points.transpose(1, 2).contiguous()  # B*N*D
                grouped_points = grouping_operation(points, idx).contiguous()  # (B, D, npoint, nsample)
                # 坐标与特征拼接作为新的特征输入
                new_features = torch.cat([grouped_xyz, grouped_points], dim=1)  # [B,C+D, npoint, nsample]
            else:
                new_features = grouped_xyz
        """用一维卷积实现MLP，提取局部特征"""
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features = F.relu(bn(conv(new_features)))
        new_features = torch.max(new_features, -1)[0]
        if self.group_all:

            return new_features
        else:

            return new_xyz, new_features


if __name__ == "__main__":
    model = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128],
                                   group_all=False).cuda()
    data = torch.zeros(10, 3, 1024).cuda()
    feature = torch.zeros(10, 3, 1024).cuda()
    out = model(data, feature)

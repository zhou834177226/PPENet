import torch
import torch.nn as nn
import torch.nn.functional as F

from GEMZSL.modeling.backbone import resnet101_features
import GEMZSL.modeling.utils as utils

from os.path import join
import pickle
import os
# import numpy as np
# np.set_printoptions(threshold=np.inf)


base_architecture_to_features = {
    'resnet101': resnet101_features,
}
# class MultiGrained_generator(nn.Module):
#     def __init__(self):
#         super(MultiGrained_generator, self).__init__()
#         self.trans = nn.Sequential(
#             nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(2048),
#             nn.LeakyReLU(0.2, True),
#         )
#         self.pool_1 = nn.AvgPool2d(2)
#         self.pool_2 = nn.AvgPool2d(3)
#         self.pool_3 = nn.AvgPool2d(5)
#         self.pool_4 = nn.AvgPool2d(7)
#
#     def forward(self, x):
#
#         x1 = self.trans(x)
#         x2 = self.pool_1(x1)
#         x3 = self.pool_2(x1)
#         x4 = self.pool_3(x1)
#         x5 = self.pool_4(x1)
#
#         return x1,x2,x3,x4,x5


class PIP(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num, attritube_num):
        super(PIP, self).__init__()

        self.conv_num = conv_num
        conv1x1s = []
        # 添加1x1卷积
        # for i in range(conv_num):

        self.convS = nn.Conv2d(in_channels, conv_num, kernel_size=1)
        # 修改conv添加中间层
        # self.convS = nn.Sequential(
        #     nn.Conv2d(in_channels, int(in_channels / 2), 1, bias=False),
        #     nn.BatchNorm2d(int(in_channels / 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(int(in_channels / 2), conv_num, 1, bias=False),
        # )

        # self.p_linear = nn.Linear(312 * conv_num, 312, False)
        # self.drop = nn.Dropout(0.5)

        # 修改concat
        self.conv_layer = nn.Sequential(
            nn.Conv1d(conv_num * attritube_num, attritube_num, 1, bias=False),
            # nn.BatchNorm2d(312),
            # nn.ReLU(),
        )

        # self.part_concat = nn.Sequential(
        #     nn.Conv2d(self.conv_num * in_channels, out_channels, 1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        # )

        # self.weight = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight.data.fill_(0.0)
        # self.weight_clamped = torch.clamp(self.weight, 0.0, 1.0)

        # 修改3 添加自适应学习
        # self.fuse_weight = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.fuse_weight.data.fill_(0.5)
        # self.fuse_weight_clamped = torch.clamp(self.fuse_weight, 0.0, 1.0)

    def forward(self, x, query, n, c, w, h):
        atten_attrs = []
        x_weights = []
        weight = self.convS(x)
        weight = torch.sigmoid(weight)

        # 修改阈值置0
        # batch, parts, width, height = weight.size()
        # weights_layout = weight.view(batch, -1)
        # threshold_value, _ = weights_layout.max(dim=1)
        # local_max, _ = weight.view(batch, parts, -1).max(dim=2)
        # threshold_value = 0.8 * threshold_value.view(batch, 1).expand(batch, parts)
        # # local_max为14*14中的最大值，threshold_value为全num通道最大值的0.8，若最大值低于该阈值则舍弃
        # weight = weight * local_max.ge(threshold_value).view(batch, parts, 1, 1).float()\
        #     .expand(batch, parts, width, height)

        x_temp = x

        for i in range(self.conv_num):
            # x_weighted = weight * x
            # x_weighted = x_weighted.reshape(n, c, w * h)

            x_weighted = x * weight[:, i, :, :].unsqueeze(dim=1).expand(n, c, w, h)
            x_weights.append(x_weighted)
            x_weighted = x_weighted.reshape(n, c, w * h)

            atten_map = torch.einsum('lv,bvr->blr', query, x_weighted)

            atten_map = F.softmax(atten_map, -1)
            atten_map = atten_map.view(n, -1, w, h)  # N*312*W*H

            # 修改加权
            # atten_map = atten_map * weight[:, i, :, :].unsqueeze(dim=1).expand(n, 312, w, h)

            atten_attr = F.max_pool2d(atten_map, kernel_size=(w, h))  # N*312*1
            atten_attr = atten_attr.view(n, -1)  # N*312

            # 修改topk求均值
            # top_values, indices = torch.topk(atten_map, k=5, dim=2)
            # atten_attr = torch.mean(top_values, dim=2)
            # atten_attr = atten_attr.view(n, -1)

            # 修改融合特征值
            # inputs = [x_temp, x_weighted]
            # x_concat = torch.cat(inputs, dim=1)
            # x_temp = self.part_concat(x_concat)

            atten_attrs.append(atten_attr)

        # 修改2 concat
        concatenated_tensor = torch.cat(atten_attrs, dim=1).unsqueeze(2)
        concatenated_tensor = self.conv_layer(concatenated_tensor)
        concatenated_tensor = torch.squeeze(concatenated_tensor)

        # 修改先concat再提特征
        # x_weights.append(x)
        # x_concat = torch.cat(x_weights, dim=1)
        # x_concat = self.part_concat(x_concat)
        #
        # x_weighted = x_concat.reshape(n, c, w * h)
        # part_map = torch.einsum('lv,bvr->blr', query, x_weighted)
        # part_map = F.softmax(part_map, -1)
        # part_map = part_map.view(n, -1, w, h)  # N*312*W*H
        #
        # part_attr = F.max_pool2d(part_map, kernel_size=(w, h))  # N*312*1
        # part_attr = part_attr.view(n, -1)  # N*312

        # 修改添加cls分支
        # part_concat = torch.cat(x_weights, dim=1)
        # part_concat = self.part_concat(part_concat)

        # concatenated_tensor = torch.cat(atten_attrs, dim=1)
        # concatenated_tensor = self.drop(self.p_linear(concatenated_tensor))

        # 将这些张量按照第0维度进行堆叠
        # atten_attrs_mid = torch.stack(atten_attrs, dim=0)
        # # 将其转换为n*312*conv_num的张量
        # atten_attrs_mid = atten_attrs_mid.permute(1, 2, 0)
        # # 在第2维度上取最大值
        # atten_attrs_final, _ = torch.max(atten_attrs_mid, dim=2)
        # # 去除维度大小为1的维度
        # atten_attrs_final = torch.squeeze(atten_attrs_final)

        # 修改1 取均值
        # atten_attrs_final = torch.mean(atten_attrs_mid, dim=2)

        # 保留原始信息
        x_trans = x.reshape(n, c, w * h)
        atten_map = torch.einsum('lv,bvr->blr', query, x_trans)
        atten_map_x = F.softmax(atten_map, -1)
        atten_map_x = atten_map_x.view(n, -1, w, h)

        atten_attr_x = F.max_pool2d(atten_map_x, kernel_size=(w, h))  # N*312*1
        atten_attr_x = atten_attr_x.view(n, -1)  # N*312

        # top_values, indices = torch.topk(atten_map_x, k=5, dim=2)
        # atten_attr_x = torch.mean(top_values, dim=2)
        # atten_attr_x = atten_attr_x.view(n, -1)

        # atten_attr = self.fuse_weight_clamped * atten_attr_x + (1-self.fuse_weight_clamped) * atten_attrs_final
        # atten_attr = (1 - self.weight) * atten_attr_x + self.weight * concatenated_tensor
        atten_attr = 0.7 * atten_attr_x + 0.3 * concatenated_tensor
        # atten_attr = 0.7 * atten_attr_x + 0.3 * atten_attrs_final
        # atten_attr = concatenated_tensor
        # atten_attr = atten_attr_x

        # part_sum = 0
        # for i in range(self.conv_num):
        #     part_sum += atten_attrs[i]
        # part_avg = part_sum / self.conv_num
        # atten_attr = 0.7 * atten_attr_x + 0.3 * part_avg

        # atten_attrs.append(atten_attr_x)
        # concatenated_tensor = torch.cat(atten_attrs, dim=1).unsqueeze(2)
        # concatenated_tensor = self.conv_layer(concatenated_tensor)
        # atten_attr = torch.squeeze(concatenated_tensor)

        # return self.conv_num, atten_attrs, atten_attr
        return atten_attr
        # return part_concat, atten_attr

    # def forward_getmap(self, x, query, n, c, w, h):
    #     atten_attrs = []
    #     x_weights = []
    #     weight = self.convS(x)
    #     weight = torch.sigmoid(weight)
    #     x_temp = x
    #
    #     for i in range(self.conv_num):
    #         # x_weighted = weight * x
    #         # x_weighted = x_weighted.reshape(n, c, w * h)
    #
    #         x_weighted = x * weight[:, i, :, :].unsqueeze(dim=1).expand(n, c, w, h)
    #         x_weights.append(x_weighted)
    #         x_weighted = x_weighted.reshape(n, c, w * h)
    #
    #         atten_map = torch.einsum('lv,bvr->blr', query, x_weighted)
    #
    #         atten_map = F.softmax(atten_map, -1)
    #         atten_map = atten_map.view(n, -1, w, h)  # N*312*W*H
    #
    #         # 修改加权
    #         # atten_map = atten_map * weight[:, i, :, :].unsqueeze(dim=1).expand(n, 312, w, h)
    #
    #         atten_attr = F.max_pool2d(atten_map, kernel_size=(w, h))  # N*312*1
    #         atten_attr = atten_attr.view(n, -1)  # N*312
    #
    #
    #         atten_attrs.append(atten_attr)
    #
    #     # 修改2 concat
    #     concatenated_tensor = torch.cat(atten_attrs, dim=1).unsqueeze(2)
    #     concatenated_tensor = self.conv_layer(concatenated_tensor)
    #     concatenated_tensor = torch.squeeze(concatenated_tensor)
    #
    #
    #     # 保留原始信息
    #     x_trans = x.reshape(n, c, w * h)
    #     atten_map = torch.einsum('lv,bvr->blr', query, x_trans)
    #     atten_map_x = F.softmax(atten_map, -1)
    #     atten_map_x = atten_map_x.view(n, -1, w, h)
    #
    #     atten_attr_x = F.max_pool2d(atten_map_x, kernel_size=(w, h))  # N*312*1
    #     atten_attr_x = atten_attr_x.view(n, -1)  # N*312
    #
    #     atten_attr = 0.7 * atten_attr_x + 0.3 * concatenated_tensor
    #
    #     # return self.conv_num, atten_attrs, atten_attr
    #     return atten_attr, atten_map_x
    #     # return part_concat, atten_attr


# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# # 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化 不会影响
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='nearest')

    # 整个 ASPP 架构


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = [3, 6, 9]
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class GEMNet(nn.Module):
    def __init__(self, res101, img_size, c, w, h,
                 attritube_num, cls_num, ucls_num, attr_group, w2v,
                 scale=20.0, device=None):

        super(GEMNet, self).__init__()
        self.device = device

        self.img_size = img_size
        # self.prototype_shape = prototype_shape
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.attr_group = attr_group

        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        assert self.w2v_att.shape[0] == self.attritube_num

        if scale <= 0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        self.backbone = res101

        # 修改部分
        self.fuse_weight4 = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        self.fuse_weight4.data.fill_(0.0)
        # self.fuse_weight4 = torch.clamp(self.fuse_weight4, 0.0, 1.0)
        self.fuse_weight4_clamped = torch.clamp(self.fuse_weight4, 0.0, 1.0)
        # self.fuse_weight4_clamped = nn.Parameter(self.fuse_weight4)
        # self.fuse_weight4_clamped = self.fuse_weight4

        self.fuse_weight3 = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        self.fuse_weight3.data.fill_(0.0)
        # self.fuse_weight3 = torch.clamp(self.fuse_weight3, 0.0, 1.0)
        self.fuse_weight3_clamped = torch.clamp(self.fuse_weight3, 0.0, 1.0)
        # self.fuse_weight3_clamped = nn.Parameter(self.fuse_weight3)
        # self.fuse_weight3_clamped = self.fuse_weight3

        # self.part_weight3 = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.part_weight3.data.fill_(0.0)
        # self.part_weight4 = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.part_weight4.data.fill_(0.0)


        # self.part1 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        # self.part2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        # self.part3 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        # self.part4 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=Falsce)
        # self.part5 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False)

        # self.aspp = ASPP(in_channels=2048, out_channels=2048)
        self.bn2048 = nn.BatchNorm2d(2048)
        self.bn1024 = nn.BatchNorm2d(1024)
        # self.bn2048 = nn.BatchNorm2d(2048, eps=1e-05, track_running_stats=False)
        # self.bn1024 = nn.BatchNorm2d(1024, eps=1e-05, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        # self.elu = nn.ELU()

        # self.rfb1L3 = RFB_modified(in_channel=1024, out_channel=1024)
        # self.rfb1L4 = RFB_modified(in_channel=2048, out_channel=2048)

        # self.asppL1 = ASPP(in_channels=256, out_channels=256)
        # self.asppL2 = ASPP(in_channels=512, out_channels=512)
        self.asppL3 = ASPP(in_channels=1024, out_channels=1024)
        self.asppL4 = ASPP(in_channels=2048, out_channels=2048)
        self.pip3 = PIP(1024, 1024, 5, self.attritube_num)
        self.pip4 = PIP(2048, 2048, 5, self.attritube_num)

        # self.fuseL1 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     # nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        # )
        # self.fuseL2 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     # nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
        #     # nn.BatchNorm2d(512),
        #     # nn.ReLU(),
        #     nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        # )
        # self.fuseL3 = nn.Sequential(
        #     # nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
        #     # nn.BatchNorm2d(1024),
        #     # nn.ReLU(),
        #     nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(),
        # )
        #
        # self.fuseL4 = nn.Sequential(
        #     nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2,  padding=1,
        #                                         output_padding=1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        # )
        self.fuseL3 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
        self.fuseL4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        # self.connectLayer3 = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )
        # self.connectLayer4 = nn.Sequential(
        #     nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        #     nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )

        # self.downL3 = nn.Sequential(
        #     nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        # )
        # self.upL4 = nn.Sequential(
        #     nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(),
        # )

        # 300 * 2048
        # self.prototype_vectors = nn.Parameter(nn.init.normal_(torch.empty(self.prototype_shape)), requires_grad=True)

        # self.W1 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 256)),
        #                       requires_grad=True)
        # self.W2 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 512)),
        #                       requires_grad=True)
        self.W3 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 1024)),
                              requires_grad=True)
        self.W4 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 2048)),
                              requires_grad=True)
        # self.Wx4 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 2048)),
        #                       requires_grad=True)
        # 2048 * 312
        # 使用单个隐藏层MLP将属性词向量e转换为视觉属性特征e (e)∈R K×C
        # self.V1 = nn.Parameter(nn.init.normal_(torch.empty(256, self.attritube_num)), requires_grad=True)
        # self.V2 = nn.Parameter(nn.init.normal_(torch.empty(512, self.attritube_num)), requires_grad=True)
        self.V3 = nn.Parameter(nn.init.normal_(torch.empty(1024, self.attritube_num)), requires_grad=True)
        self.V4 = nn.Parameter(nn.init.normal_(torch.empty(2048, self.attritube_num)), requires_grad=True)
        # self.V3part = nn.Parameter(nn.init.normal_(torch.empty(1024, self.attritube_num)), requires_grad=True)
        # self.V4part = nn.Parameter(nn.init.normal_(torch.empty(2048, self.attritube_num)), requires_grad=True)
        # self.Vx4 = nn.Parameter(nn.init.normal_(torch.empty(2048, self.attritube_num)), requires_grad=True)
        #  嵌入层，视觉嵌入词向量空间
        # 将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。
        # 即在定义网络时这个tensor就是一个可以训练的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化

        # loss
        self.Reg_loss = nn.MSELoss()
        self.CLS_loss = nn.CrossEntropyLoss()
        # nn.CrossEntropyLoss() 会自动先经过softmax处理

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x1, x2, x3, x4 = self.backbone(x)
        # x4 = self.backbone(x)

        return x1, x2, x3, x4

    # 计算 cos(h(x),y) = σ y*h(x)/|x|*|y|
    def base_module(self, x, seen_att, layerV):

        N, C, W, H = x.shape

        # global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        # # N*C*1
        # global_feat = global_feat.view(N, C)
        # # N*C
        # # 嵌入层，视觉嵌入语义空间
        # gs_feat = torch.einsum('bc,cd->bd', global_feat, layerV)
        # # N*C * C*312 = N*312

        # 修改3
        # fuse_atten_map = F.avg_pool2d(atten_map, kernel_size=(W, H))  # N*312*1
        # fuse_atten_map = fuse_atten_map.view(N, 312)  # N * 312
        # print(f'debug_x: {x.detach().cpu().numpy()}')
        global_feat = F.avg_pool2d(x, kernel_size=(W, H))  # N,C,1
        global_feat = global_feat.view(N, C)
        gs_feat = torch.einsum('bc,cd->bd', global_feat, layerV)
        # print(gs_feat.shape,atten_att.shape)

        # print(f'debug_gs_feat: {gs_feat.detach().cpu().numpy()}')
        # print(f'debug_atten_att: {atten_att.detach().cpu().numpy()}')
        # gs_feat = gs_feat * atten_att

        # gs_feat = x

        # h(x)/|x|
        gs_feat_norm = torch.norm(gs_feat, p=2, dim=1).unsqueeze(1).expand_as(gs_feat)
        # norm求指定维度上的范数 可以用keepdim保持维度
        # p为指定几范数
        # unsqueeze在指定维度上增维
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)
        # 归一化 防止除0
        # print(f'debug_gs_feat_normalized: {gs_feat_normalized.detach().cpu().numpy()}')

        # y/|y|
        temp_norm = torch.norm(seen_att, p=2, dim=1).unsqueeze(1).expand_as(seen_att)
        seen_att_normalized = seen_att.div(temp_norm + 1e-5)

        # y*h(x)/|x|*|y|
        # scale 作为缩放因子
        cos_dist = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        score = cos_dist * self.scale

        # print(f'debug_score: {score.detach().cpu().numpy()}')

        # zero_error = torch.full((N, 150), 0.0001)
        # zero_error = zero_error.to(self.device)
        # score = score + 1e-5
        # print(f'debug_gs_feat: {gs_feat}')
        # print(f'debug_gs_feat_normalized: {gs_feat_normalized}')
        # print(f'debug_gs_seen_att_normalized: {seen_att_normalized}')
        # print(f'debug_score: {score}')
        # 在余弦相似度中使用图像特征和类语义嵌入归一化有助于减少类内方差，提高未见类的准确率

        return score

    # 计算注意力图 属性
    def attentionModule(self, x, layerW):

        N, C, W, H = x.shape
        query = torch.einsum('lw,wv->lv', self.w2v_att, layerW)
        # *******************************************************************************************************
        # x = x.reshape(N, C, W * H)  # B, V, R=WH
        #
        # atten_map = torch.einsum('lv,bvr->blr', query, x)
        # # batch * L * r N*312*R
        #
        # # 激活
        # atten_map = F.softmax(atten_map, -1)
        # atten_map = atten_map.view(N, -1, W, H)  # N*312*W*H
        #
        # atten_attr = F.max_pool2d(atten_map, kernel_size=(W, H))  # N*312*1
        # atten_attr = atten_attr.view(N, -1)  # N*312
        # *******************************************************************************************************

        # top_values, indices = torch.topk(atten_map, k=1, dim=2)
        # atten_attr = torch.mean(top_values, dim=2)
        # atten_attr = atten_attr.view(N, -1)

        # pip*******************************************************************************************************
        if C == 1024:
            # pip_num, part_attrs, atten_attr = self.pip3(x, query, N, C, W, H)
            # part_feature, atten_attr = self.pip3(x, query, N, C, W, H)
            atten_attr = self.pip3(x, query, N, C, W, H)
        if C == 2048:
            # pip_num, part_attrs, atten_attr = self.pip4(x, query, N, C, W, H)
            # part_feature, atten_attr = self.pip4(x, query, N, C, W, H)
            atten_attr = self.pip4(x, query, N, C, W, H)
        # pip*******************************************************************************************************

        # return atten_map, atten_attr, query
        # return pip_num, part_attrs, atten_attr, query
        # return part_feature, atten_attr, query
        return atten_attr, query

    # def attentionModule_getmap(self, x, layerW):
    #
    #     N, C, W, H = x.shape
    #     query = torch.einsum('lw,wv->lv', self.w2v_att, layerW)
    #     if C == 1024:
    #         # pip_num, part_attrs, atten_attr = self.pip3(x, query, N, C, W, H)
    #         # part_feature, atten_attr = self.pip3(x, query, N, C, W, H)
    #         atten_attr, atten_map = self.pip3.forward_getmap(x, query, N, C, W, H)
    #     if C == 2048:
    #         # pip_num, part_attrs, atten_attr = self.pip4(x, query, N, C, W, H)
    #         # part_feature, atten_attr = self.pip4(x, query, N, C, W, H)
    #         atten_attr, atten_map = self.pip4.forward_getmap(x, query, N, C, W, H)
    #     return atten_attr, query, atten_map
    def attr_decorrelation(self, query):

        loss_sum = 0

        for key in self.attr_group:
            group = self.attr_group[key]
            proto_each_group = query[group]  # g1 * v
            channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
            loss_sum += channel_l2_norm.mean()

        loss_sum = loss_sum.float()/len(self.attr_group)

        return loss_sum

    def CPT(self, atten_map):
        """

        :param atten_map: N, L, W, H
        :return:
        """

        N, L, W, H = atten_map.shape
        xp = torch.tensor(list(range(W))).long().unsqueeze(1).to(self.device)
        yp = torch.tensor(list(range(H))).long().unsqueeze(0).to(self.device)

        xp = xp.repeat(1, H)
        yp = yp.repeat(W, 1)

        atten_map_t = atten_map.view(N, L, -1)
        value, idx = atten_map_t.max(dim=-1)

        # tx = idx // H
        tx = torch.div(idx, H, rounding_mode='floor')
        ty = idx - H * tx

        xp = xp.unsqueeze(0).unsqueeze(0)
        yp = yp.unsqueeze(0).unsqueeze(0)
        tx = tx.unsqueeze(-1).unsqueeze(-1)
        ty = ty.unsqueeze(-1).unsqueeze(-1)

        pos = (xp - tx) ** 2 + (yp - ty) ** 2

        loss = atten_map * pos

        loss = loss.reshape(N, -1).mean(-1)
        loss = loss.mean()

        return loss

    def coattention(self, layera, layerb):
        layera_mid = self.fuseL3(layera)  # batch*2048*28*28
        layerb_mid = self.fuseL4(layerb)  # batch*1024*14*14
        # print(f'debug_layerb_mid: {torch.isnan(layerb_mid).any()}')

        N1, C1, W1, H1 = layera.shape  # batch 1024 28 28
        N2, C2, W2, H2 = layerb.shape  # batch 2048 14 14

        layera_reshape = layera.reshape(N1, C1, W1 * H1)  # batch*1024*(28*28)
        layerb_reshape = layerb.reshape(N2, C2, W2 * H2)  # batch*2048*(14*14)
        layera_mid = layera_mid.reshape(N1, C2, W1 * H1)  # batch*2048*(28*28)
        layerb_mid = layerb_mid.reshape(N2, C1, W2 * H2)  # batch*1024*(14*14)

        layera_con_mid = torch.norm(layera_reshape, p=2, dim=1).unsqueeze(1).expand_as(layera_reshape)
        layera_con_normalized = layera_reshape.div(layera_con_mid + 1e-5)

        layerb_con_mid = torch.norm(layerb_mid, p=2, dim=1).unsqueeze(1).expand_as(layerb_mid)
        layerb_con_normalized = layerb_mid.div(layerb_con_mid + 1e-5)

        matrix = torch.einsum('bcq,bcp->bqp', layera_con_normalized, layerb_con_normalized)  # batch（28*28）*（14*14）

        min_vals, _ = torch.min(matrix, dim=1, keepdim=True)
        max_vals, _ = torch.max(matrix, dim=1, keepdim=True)
        similarity_matrix = (matrix - min_vals) / (max_vals - min_vals)

        layera_final = torch.einsum('bcp,bqp->bcq', layerb_mid,
                                    similarity_matrix)  # batch*1024*（14*14） * batch*（28*28）*（14*14）= batch*1024*（28*28）
        layerb_final = torch.einsum('bcq,bqp->bcp', layera_mid,
                                    similarity_matrix)  # batch*2048*（28*28） * batch*（28*28）*（14*14）= batch*2048*（14*14）

        layera_final = self.relu(self.bn1024(layera_final.view(N1, C1, W1, H1)))
        layerb_final = self.relu(self.bn2048(layerb_final.view(N2, C2, W2, H2)))

        return layera_final, layerb_final

    def forward(self, x, att=None, label=None, seen_att=None):

        x1, x2, x3, x4 = self.conv_features(x)  # N， 2048， 14， 14
        # x4 = self.conv_features(x)  # N， 2048， 14， 14

        # print(f'debug_x4_res: {x4.detach().cpu().numpy()}')

        # x3 = self.relu(self.bn1024(xx1+xx2+x3))
        # xx4 = self.relu(self.bn2048(xx1 + xx2 + xx3 + x4))

        # print(f'debug_x4_ASPP: {x4.detach().cpu().numpy()}')
        # print(f'debug_x4_beforeco: {torch.isnan(x4).any()}')

        # *******************************************************************************************************
        x4 = self.asppL4(x4)
        x3 = self.asppL3(x3)
        x3_final, x4_final = self.coattention(x3, x4)
        x4 = (1 - self.fuse_weight4_clamped) * x4 + self.fuse_weight4_clamped * x4_final
        x3 = (1 - self.fuse_weight3_clamped) * x3 + self.fuse_weight3_clamped * x3_final
        # *******************************************************************************************************

        # x4 = self.asppL4(x4)
        # x3 = self.asppL3(x3)
        # x4 = (1 - self.fuse_weight4) * x4 + self.fuse_weight4 * x4_final
        # x3 = (1 - self.fuse_weight3) * x3 + self.fuse_weight3 * x3_final
        # print(f'debug_fuse_weight4: {self.fuse_weight4}')
        # print(f'debug_fuse_weight4_clamped: {self.fuse_weight4_clamped}')
        # print(f'debug_fuse_weight3: {self.fuse_weight3}')
        # print(f'debug_fuse_weight3_clamped: {self.fuse_weight3_clamped}')
        # # x4 = 0.9 * x4 + 0.1 * x4_final
        # x3 = 0.9 * x3 + 0.1 * x3_final
        # print(f'debug_x4_coattention: {x4_final.detach().cpu().numpy()}')

        # x3 = x3_final
        # x4 = x4_final
        # print("x4is:", x4)
        # print("x4_final is:", x4_final)

        # atten_map1, atten_attr1, query1 = self.attentionModule(x1, self.W1)
        # atten_map2, atten_attr2, query2 = self.attentionModule(x2, self.W2)
        # atten_attr3, query3 = self.attentionModule(x3, self.W3)
        # atten_attr4, query4 = self.attentionModule(x4, self.W4)
        # pip_num3, part_attrs3, atten_attr3, query3 = self.attentionModule(x3, self.W3)
        # pip_num4, part_attrs4, atten_attr4, query4 = self.attentionModule(x4, self.W4)
        # part_feature3, atten_attr3, query3 = self.attentionModule(x3, self.W3)
        # part_feature4, atten_attr4, query4 = self.attentionModule(x4, self.W4)

        atten_attr3, query3 = self.attentionModule(x3, self.W3)
        atten_attr4, query4 = self.attentionModule(x4, self.W4)


        # x4 = self.asppL4(x4)
        # x3 = self.asppL3(x3)
        # x3 = (1-self.part_weight3) * x3 + self.part_weight3 * part_feature3
        # x4 = (1-self.part_weight4) * x4 + self.part_weight4 * part_feature4
        # print(f'debug_part_weight3: {self.part_weight3}')
        # print(f'debug_part_weight4: {self.part_weight4}')

        # score1 = self.base_module(x1, seen_att, self.V1, atten_map1)
        # score2 = self.base_module(x2, seen_att, self.V2, atten_map2)
        # atten_attr3 = self.split(atten_map3)
        # atten_attr4 = self.split(atten_map4)
        # print(f'debug_x4: {torch.isnan(x4).any()}')
        score3 = self.base_module(x3, seen_att, self.V3)
        score4 = self.base_module(x4, seen_att, self.V4)  # N, d  # N, d
        # score3part = self.base_module(part_feature3, seen_att, self.V3part)
        # score4part = self.base_module(part_feature4, seen_att, self.V4part)
        # scorex4 = self.base_module(xx4, seen_att, self.Vx4, atten_mapx4)  # N, d

        # score3 = 0.5 * score3 + 0.5 * score3part
        # score4 = 0.5 * score4 + 0.5 * score4part
        # score = self.base_module(x4, s                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 een_att, self.V4, xx4)

        score = []
        for i in range(0, 11):
            score.append((score3 * i + score4 * (10-i))*0.1)

        if not self.training:
            return score
        # return x3, x4, score

        # Lcls1 = self.CLS_loss(score1, label)
        # Lcls2 = self.CLS_loss(score2, label)
        Lcls3 = self.CLS_loss(score3, label)
        Lcls4 = self.CLS_loss(score4, label)
        # Lcls3part = self.CLS_loss(score3part, label)
        # Lcls4part = self.CLS_loss(score4part, label)
        # print(f'debug_score4: {torch.isnan (score4).any()}')
        # Lclsx4 = self.CLS_loss(scorex4, label)

        # print(f'debug_label: {label}')

        # log_info_Lcls3 = 'Lcls3: %.4f' % (Lcls3)
        # log_info_Lcls4 = 'Lcls4: %.4f' % (Lcls4)
        # print(log_info_Lcls3)
        # print(log_info_Lcls4)

        # Lcls4 = self.CLS_loss(score, label)

        # Lreg1 = self.Reg_loss(atten_attr1, att)
        # Lreg2 = self.Reg_loss(atten_attr2, att)

        # 修改part loss
        Lreg3 = self.Reg_loss(atten_attr3, att)
        Lreg4 = self.Reg_loss(atten_attr4, att)
        # Lparts3 = []
        # Lparts4 = []
        # lpart3_num = 0
        # lpart4_num = 0

        # for i in range(pip_num3):
        #     temp_loss3 = self.Reg_loss(part_attrs3[i], att)
        #     lpart3_num += temp_loss3
        #     # Lparts3.append(temp_loss3)
        # for i in range(pip_num4):
        #     temp_loss4 = self.Reg_loss(part_attrs4[i], att)
        #     lpart4_num += temp_loss4
            # Lparts4.append(temp_loss4)

        # lpart3_num = lpart3_num / pip_num3
        # lpart4_num = lpart4_num / pip_num4

        # Lreg3 = 0.9 * Lreg3 + 0.1 * lpart3_num
        # Lreg4 = 0.9 * Lreg4 + 0.1 * lpart4_num

        if self.attr_group is not None:
            # Lad1 = self.attr_decorrelation(query1)
            # Lad2 = self.attr_decorrelation(query2)
            Lad3 = self.attr_decorrelation(query3)
            Lad4 = self.attr_decorrelation(query4)
            # Ladx4 = self.attr_decorrelation(queryx4)
        else:
            # Lad1 = torch.tensor(0).float().to(self.device)
            # Lad2 = torch.tensor(0).float().to(self.device)
            Lad3 = torch.tensor(0).float().to(self.device)
            Lad4 = torch.tensor(0).float().to(self.device)
            # Ladx4 = torch.tensor(0).float().to(self.device)

        # Lcpt1 = self.CPT(atten_map1)
        # Lcpt2 = self.CPT(atten_map2)
        # Lcpt3 = self.CPT(atten_map3)
        # Lcpt4 = self.CPT(atten_map4)
        # Lcptx4 = self.CPT(atten_mapx4)

        scale = self.scale.item()


        # loss_dict1 = {
        #     'Reg_loss': Lreg1,
        #     'Cls_loss': Lcls1,
        #     'AD_loss': Lad1,
        #     # 'CPT_loss': Lcpt1,
        # }
        # loss_dict2 = {
        #     'Reg_loss': Lreg2,
        #     'Cls_loss': Lcls2,
        #     'AD_loss': Lad2,
        #     # 'CPT_loss': Lcpt2,
        # }
        loss_dict3 = {
            'Reg_loss': Lreg3,
            'Cls_loss': Lcls3,
            'AD_loss': Lad3,
            # 'part_loss': lpart3_num,
            # 'CPT_loss': Lcpt3,
        }
        loss_dict4 = {
            'Reg_loss': Lreg4,
            'Cls_loss': Lcls4,
            'AD_loss': Lad4,
            # 'part_loss': lpart4_num,
            # 'CPT_loss': Lcpt4,
            'scale': scale
        }
        # loss_dictx4 = {
        #     'Reg_loss': Lregx4,
        #     'Cls_loss': Lclsx4,
        #     'AD_loss': Ladx4,
        #     'CPT_loss': Lcptx4,
        # }

        # return loss_dict1, loss_dict2, loss_dict3, loss_dict4
        return loss_dict3, loss_dict4


    # def forward_getmap(self, x, att=None, label=None, seen_att=None):
    #
    #     x1, x2, x3, x4 = self.conv_features(x)  # N， 2048， 14， 14
    #     # x4 = self.conv_features(x)  # N， 2048， 14， 14
    #
    #     # *******************************************************************************************************
    #     x4 = self.asppL4(x4)
    #     x3 = self.asppL3(x3)
    #     x3_final, x4_final = self.coattention(x3, x4)
    #     x4 = (1 - self.fuse_weight4_clamped) * x4 + self.fuse_weight4_clamped * x4_final
    #     x3 = (1 - self.fuse_weight3_clamped) * x3 + self.fuse_weight3_clamped * x3_final
    #     # *******************************************************************************************************
    #
    #     atten_attr3, query3, atten_map3 = self.attentionModule_getmap(x3, self.W3)
    #     atten_attr4, query4, atten_map4 = self.attentionModule_getmap(x4, self.W4)
    #
    #     # score3 = self.base_module(x3, seen_att, self.V3)
    #     # score4 = self.base_module(x4, seen_att, self.V4)  # N, d  # N, d                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             een_att, self.V4, xx4)
    #     #
    #     # score = []
    #     # for i in range(0, 11):
    #     #     score.append((score3 * i + score4 * (10-i))*0.1)
    #     #
    #     # if not self.training:
    #     #     return score
    #     #     # return score4
    #     #
    #     # Lcls3 = self.CLS_loss(score3, label)
    #     # Lcls4 = self.CLS_loss(score4, label)
    #     #
    #     # # 修改part loss
    #     # Lreg3 = self.Reg_loss(atten_attr3, att)
    #     # Lreg4 = self.Reg_loss(atten_attr4, att)
    #     # # Lparts3 = []
    #     # # Lparts4 = []
    #     # # lpart3_num = 0
    #     # # lpart4_num = 0
    #     #
    #     # if self.attr_group is not None:
    #     #     # Lad1 = self.attr_decorrelation(query1)
    #     #     # Lad2 = self.attr_decorrelation(query2)
    #     #     Lad3 = self.attr_decorrelation(query3)
    #     #     Lad4 = self.attr_decorrelation(query4)
    #     #     # Ladx4 = self.attr_decorrelation(queryx4)
    #     # else:
    #     #     # Lad1 = torch.tensor(0).float().to(self.device)
    #     #     # Lad2 = torch.tensor(0).float().to(self.device)
    #     #     Lad3 = torch.tensor(0).float().to(self.device)
    #     #     Lad4 = torch.tensor(0).float().to(self.device)
    #     #     # Ladx4 = torch.tensor(0).float().to(self.device)
    #     #
    #     # scale = self.scale.item()
    #     #
    #     # loss_dict3 = {
    #     #     'Reg_loss': Lreg3,
    #     #     'Cls_loss': Lcls3,
    #     #     'AD_loss': Lad3,
    #     #     # 'part_loss': lpart3_num,
    #     #     # 'CPT_loss': Lcpt3,
    #     # }
    #     # loss_dict4 = {
    #     #     'Reg_loss': Lreg4,
    #     #     'Cls_loss': Lcls4,
    #     #     'AD_loss': Lad4,
    #     #     # 'part_loss': lpart4_num,
    #     #     # 'CPT_loss': Lcpt4,
    #     #     'scale': scale
    #     # }
    #
    #     return atten_map3, atten_map4

    def getAttention(self, x):
        feat = self.conv_features(x)
        part_feat, atten_map, atten_attr, query = self.attentionModule(feat)
        return atten_map


def build_GEMNet(cfg):
    dataset_name = cfg.DATASETS.NAME
    # 数据集名称
    info = utils.get_attributes_info(dataset_name)
    # get_attributes_info用于获得数据集属性dim、类别n、未知类m
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]

    attr_group = utils.get_attr_group(dataset_name)
    # 将属性分组，同一区域不同属性类分组

    img_size = cfg.DATASETS.IMAGE_SIZE
    # 数据集的图像大小，CUB：448

    # res101 feature size
    c, w, h = 2048, img_size//32, img_size//32
    # //取整运算符 确认特征图大小，c=2048 w,h=img_size//32
    # 为何除32？？？？？？？

    scale = cfg.MODEL.SCALE
    # 标尺？CUB中是20.0
    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    # 赋是否训练
    model_dir = cfg.PRETRAINED_MODELS
    # 赋预训练数据地址
    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    # 读取resnet-101模型,使用预训练模型参数
    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)
    # 读取w2v文件路径

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)
    # 读取改写的属性
    device = torch.device(cfg.MODEL.DEVICE)
    # 分配GPU

    return GEMNet(res101=res101, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attritube_num=attritube_num,
                  attr_group=attr_group, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)
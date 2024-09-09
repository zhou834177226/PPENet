import torch
import torch.nn as nn
import torch.nn.functional as F

from GEMZSL.modeling.backbone import resnet101_features
import GEMZSL.modeling.utils as utils

from os.path import join
import pickle
import os


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

# 空洞卷积
# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)
#
#
# # 池化 -> 1*1 卷积 -> 上采样
# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),  # 自适应均值池化 不会影响
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         # 上采样
#         return F.interpolate(x, size=size, mode='nearest')
#
#     # 整个 ASPP 架构
#
#
# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels=256):
#         super(ASPP, self).__init__()
#         modules = []
#         # 1*1 卷积
#         modules.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()))
#
#         # 多尺度空洞卷积
#         rates = [3, 6, 9]
#         for rate in rates:
#             modules.append(ASPPConv(in_channels, out_channels, rate))
#
#         # 池化
#         modules.append(ASPPPooling(in_channels, out_channels))
#
#         self.convs = nn.ModuleList(modules)
#
#         # 拼接后的卷积
#         self.project = nn.Sequential(
#             nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
#
#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)

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
        # self.aspp = ASPP(in_channels=2048, out_channels=2048)
        # self.bn2048 = nn.BatchNorm2d(2048)
        self.bn1024 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

        # self.asppL1 = ASPP(in_channels=256, out_channels=256)
        # self.asppL2 = ASPP(in_channels=512, out_channels=512)
        # self.asppL3 = ASPP(in_channels=1024, out_channels=1024)
        # self.asppL4 = ASPP(in_channels=2048, out_channels=2048)

        self.fuseL1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.fuseL2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        # self.fuseL3 = nn.Sequential(
        #     nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        #     nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(),
        # )

        # self.fuseL4 = nn.Sequential(
        #     nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2,  padding=1,
        #                                         output_padding=1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        # )

        # 300 * 2048
        # self.prototype_vectors = nn.Parameter(nn.init.normal_(torch.empty(self.prototype_shape)), requires_grad=True)

        self.W1 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 256)),
                              requires_grad=True)
        self.W2 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 512)),
                              requires_grad=True)
        self.W3 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 1024)),
                              requires_grad=True)
        self.W4 = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], 2048)),
                              requires_grad=True)
        # 2048 * 312
        # 使用单个隐藏层MLP将属性词向量e转换为视觉属性特征e (e)∈RK×C
        self.V1 = nn.Parameter(nn.init.normal_(torch.empty(256, self.attritube_num)), requires_grad=True)
        self.V2 = nn.Parameter(nn.init.normal_(torch.empty(512, self.attritube_num)), requires_grad=True)
        self.V3 = nn.Parameter(nn.init.normal_(torch.empty(1024, self.attritube_num)), requires_grad=True)
        self.V4 = nn.Parameter(nn.init.normal_(torch.empty(2048, self.attritube_num)), requires_grad=True)
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
    def base_module(self, x, seen_att, layerV, atten_map):

        N, C, W, H = x.shape

        # global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        # # N*C*1
        # global_feat = global_feat.view(N, C)
        # # N*C
        # # 嵌入层，视觉嵌入语义空间
        # gs_feat = torch.einsum('bc,cd->bd', global_feat, layerV)
        # # N*C * C*312 = N*312

        # 修改3
        fuse_atten_map = F.avg_pool2d(atten_map, kernel_size=(W, H))  # N*312*1
        fuse_atten_map = fuse_atten_map.view(N, 312)  # N * 312
        gs_feat = F.avg_pool2d(x, kernel_size=(W, H))  # N,C,1
        gs_feat = gs_feat.view(N, C)
        gs_feat = torch.einsum('bc,cd->bd', gs_feat, layerV)
        gs_feat = gs_feat * fuse_atten_map

        # gs_feat = x

        # h(x)/|x|
        gs_feat_norm = torch.norm(gs_feat, p=2, dim=1).unsqueeze(1).expand_as(gs_feat)
        # norm求指定维度上的范数 可以用keepdim保持维度
        # unsqueeze在指定维度上增维
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)
        # 归一化 防止除0

        # y/|y|
        temp_norm = torch.norm(seen_att, p=2, dim=1).unsqueeze(1).expand_as(seen_att)
        seen_att_normalized = seen_att.div(temp_norm + 1e-5)

        # y*h(x)/|x|*|y|
        # scale 作为缩放因子
        cos_dist = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        score = cos_dist * self.scale
        # 在余弦相似度中使用图像特征和类语义嵌入归一化有助于减少类内方差，提高未见类的准确率

        return score

    # 计算注意力图 属性
    def attentionModule(self, x, layerW):

        N, C, W, H = x.shape
        x = x.reshape(N, C, W * H)  # B, V, R=WH

        query = torch.einsum('lw,wv->lv', self.w2v_att, layerW)
        # L * V 312*2048 = 312*300 * 300*2048

        atten_map = torch.einsum('lv,bvr->blr', query, x)
        # batch * L * r N*312*R

        # ————————————————————————————修改0——————————————————————————————
        # xx = torch.einsum('bcr,cd->bdr', x, layerV)  # N*312*R
        # atten_map = F.normalize(atten_map)  # N*312*R
        # xx = F.normalize(xx)
        # atten_map = F.normalize(atten_map * xx)  # N*312*R N*312*R

        # 激活
        atten_map = F.softmax(atten_map, -1)
        atten_map = atten_map.view(N, -1, W, H)  # N*312*W*H

        # ————————————————————————————暂时屏蔽——————————————————————————————
        # x = x.transpose(2, 1)
        # # batch, WH=r, V 更换维度 N*R*2048
        # part_feat = torch.einsum('blr,brv->blv', atten_map, x)
        # # batch * L * V  N*312*2048??????
        # part_feat = F.normalize(part_feat, dim=-1)

        atten_attr = F.max_pool2d(atten_map, kernel_size=(W, H))  # N*312*1
        atten_attr = atten_attr.view(N, -1)  # N*312

        # ————————————————————————————修改1——————————————————————————————
        # fuse_atten_map = F.avg_pool2d(atten_map, kernel_size=(W, H))  # N*312*1
        # fuse_atten_map = fuse_atten_map.view(N, 312)  # N * 312
        # xx = F.avg_pool2d(xx, kernel_size=(W, H))  # N,C,1
        # xx = xx.view(N, C)
        # xx = torch.einsum('bc,cd->bd', xx, layerV)
        # xx = xx * fuse_atten_map

        # return part_feat, atten_map, atten_attr, query
        return atten_map, atten_attr, query

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

    def forward(self, x, att=None, label=None, seen_att=None):

        x1, x2, x3, x4 = self.conv_features(x)  # N， 2048， 14， 14
        # x4 = self.conv_features(x)  # N， 2048， 14， 14

        xx1 = self.fuseL1(x1)
        xx2 = self.fuseL2(x2)
        # xx3 = self.fuseL3(x3)

        # x4 = self.asppL4(x4)
        x3 = self.relu(self.bn1024(xx1+xx2+x3))
        # x3 = self.asppL3(x3)

        atten_map1, atten_attr1, query1 = self.attentionModule(x1, self.W1)
        atten_map2, atten_attr2, query2 = self.attentionModule(x2, self.W2)
        atten_map3, atten_attr3, query3 = self.attentionModule(x3, self.W3)
        atten_map4, atten_attr4, query4 = self.attentionModule(x4, self.W4)

        score1 = self.base_module(x1, seen_att, self.V1, atten_map1)
        score2 = self.base_module(x2, seen_att, self.V2, atten_map2)
        score3 = self.base_module(x3, seen_att, self.V3, atten_map3)
        score4 = self.base_module(x4, seen_att, self.V4, atten_map4)  # N, d

        # score = self.base_module(x4, seen_att, self.V4, xx4)

        score = []
        for i in range(0, 10):
            score.append((score3 * i + score4 * (10-i))*0.1)

        if not self.training:
            return score1, score2, score3, score4, score
            # return score4

        Lcls3 = self.CLS_loss(score3, label)
        Lcls4 = self.CLS_loss(score4, label)

        # Lcls4 = self.CLS_loss(score, label)

        Lreg1 = self.Reg_loss(atten_attr1, att)
        Lreg2 = self.Reg_loss(atten_attr2, att)
        Lreg3 = self.Reg_loss(atten_attr3, att)
        Lreg4 = self.Reg_loss(atten_attr4, att)

        if self.attr_group is not None:
            Lad4 = self.attr_decorrelation(query4)
        else:
            Lad4 = torch.tensor(0).float().to(self.device)

        Lcpt1 = self.CPT(atten_map1)
        Lcpt2 = self.CPT(atten_map2)
        Lcpt3 = self.CPT(atten_map3)
        Lcpt4 = self.CPT(atten_map4)

        scale = self.scale.item()


        loss_dict1 = {
            'Reg_loss': Lreg1,
            'CPT_loss': Lcpt1,
        }
        loss_dict2 = {
            'Reg_loss': Lreg2,
            'CPT_loss': Lcpt2,
        }
        loss_dict3 = {
            'Cls_loss': Lcls3,
            'Reg_loss': Lreg3,
            'CPT_loss': Lcpt3,
        }
        loss_dict4 = {
            'Reg_loss': Lreg4,
            'Cls_loss': Lcls4,
            'AD_loss': Lad4,
            'CPT_loss': Lcpt4,
            'scale': scale
        }

        return loss_dict1, loss_dict2, loss_dict3, loss_dict4
        # return loss_dict4

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
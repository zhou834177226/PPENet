import torch

def make_optimizer(cfg, model):

    lr = cfg.SOLVER.BASE_LR
    # 设定学习率0.001
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    # 设定权重衰减，防止过拟合
    momentum = cfg.SOLVER.MOMENTUM
    # 设定动量，防止陷入局部最优解

    params_to_update = []
    params_names = []
    # named_parameters()返回的list中，每个元组（与list相似，只是数据不可修改）打包了2个内容，
    # 分别是layer-name和layer-param（网络层的名字和参数的迭代器）
    for name, param in model.named_parameters():
        # 各层中参数名称和数据
        if param.requires_grad == True:
            params_to_update.append(param)
            params_names.append(name)

    optimizer = torch.optim.SGD(params_to_update, lr=lr,
                weight_decay=weight_decay, momentum=momentum)
    # 设置SGD梯度更新策略

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    step_size = cfg.SOLVER.STEPS
    # 设置每多少epochs更新一次lr
    gamma = cfg.SOLVER.GAMMA
    # 更新lr的系数gamma
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # 将预训练的参数权重加载到新的模型之中
    # 当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合
    # 如果新构建的模型在层数上进行了部分微调，则上述代码就会报错：说key对应不上。
    # 此时，如果采用strict=False 与训练权重中与新构建网络中匹配层的键值就进行使用，没有的就默认初始化。

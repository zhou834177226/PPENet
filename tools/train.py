import sys
import argparse
import torch
import numpy as np
import random
import os
from os.path import join

sys.path.append("E:/PROJECT/Gem")

seed = 822 # 822 for CUB
# print("seed : {}".format(seed))
# 设置随机种子
random.seed(seed)                                # PYTHON随机
os.environ['PYTHONHASHSEED'] = str(seed)         # 设置python哈希种子，禁止hash随机化
torch.manual_seed(seed)                          # torch CPU种子
torch.cuda.manual_seed(seed)                     # torch GPU种子
torch.cuda.manual_seed_all(seed)                 # torch 多GPU种子
np.random.seed(seed)                             # numpy 种子
torch.backends.cudnn.benchmark = False           # True为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
torch.backends.cudnn.deterministic = True        # 选择确定性算法
torch.backends.cudnn.enabled = False             # 为每一轮寻找最优算法？？True使用非确定性算法
# torch.use_deterministic_algorithms(True)         # 使用非确定算法时报错
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 10.2以上cuda需要设置，为保证use_deterministic_algorithms(True)可用
# cuDNN 是英伟达专门为深度神经网络所开发出来的 GPU 加速库，针对卷积、池化等等常见操作做了非常多的底层优化，
# 比一般的 GPU 程序要快很多。在使用 cuDNN 的时候，torch.backends.cudnn.benchmark 模式是为 False。
# 哪些因素会影响到卷积层的运行时间：
# 1、首先，当然是卷积层本身的参数，常见的包括卷积核大小，stride，dilation，padding ，输出通道的个数等；
# 2、其次，是输入的相关参数，包括输入的宽和高，输入通道的个数等；
# 3、最后，还有一些其他的因素，比如硬件平台，输入输出精度、布局等等。
# 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，
# 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
# 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其
# 实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间

from GEMZSL.data import build_dataloader  # 可以直接从文件目录而非py文件中找函数
from GEMZSL.modeling import build_zsl_pipeline
from GEMZSL.solver import make_optimizer, make_lr_scheduler
from GEMZSL.engine.trainer import do_train
from GEMZSL.config import cfg
from GEMZSL.utils.comm import *
from GEMZSL.utils import ReDirectSTD
from torch.cuda.amp import GradScaler

def train_model(cfg, local_rank, distributed):

    model = build_zsl_pipeline(cfg)
    # 建立模型，读取w2v改写的属性，分配GPU，返回res101，输入尺寸，通道数，特征图大小，scale？属性，属性组，已知类未知类
    device = torch.device(cfg.MODEL.DEVICE)
    # 分配GPU，已经在之前步骤分配过
    model = model.to(device)
    # 将模型输入GPU
    optimizer = make_optimizer(cfg, model)
    # 设置优化器，SGD梯度更新策略
    scheduler = make_lr_scheduler(cfg, optimizer)
    # 设置学习率衰减
    scaler = GradScaler()
    # torch.amp设定

    # # 如果多GPU训练进行分配
    # if distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[local_rank], output_device=local_rank,
    #         # this should be removed if we update BatchNorm stats
    #         broadcast_buffers=False,
    #     )

    tr_dataloader, tu_loader, ts_loader, res = build_dataloader(cfg, is_distributed=distributed)
    # 读数据

    # 设置输出文件
    output_dir = cfg.OUTPUT_DIR
    model_file_name = cfg.MODEL_FILE_NAME
    model_file_path = join(output_dir, model_file_name)

    test_gamma = cfg.TEST.GAMMA
    # 超参数
    max_epoch = cfg.SOLVER.MAX_EPOCH
    # 迭代轮数

    lamd = {
        1: cfg.MODEL.LOSS.LAMBDA1,
        2: cfg.MODEL.LOSS.LAMBDA2,
        3: cfg.MODEL.LOSS.LAMBDA3,
    }
    # 设置loss的超参数

    do_train(
        model,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        device,
        max_epoch,
        model_file_path,
        scaler,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Zero-Shot Learning Training")  # 描述该解析器是做什么的，该参数可以为空
    # 添加参数
    parser.add_argument("--config_file", default="E:/PROJECT/Gem/config/GEMZSL/cub_4w_2s.yaml", metavar="FILE", help="path to config file", type=str, )
    # 更改运行数据集需要更改配置的路径名  ？？如果设置相对路径，当前路径值是什么m
    parser.add_argument("--local_rank", type=int, default=0)
    # 本机默认0？
    args = parser.parse_args()
    # print(args.config_file)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    args.distributed = num_gpus > 1
    # (如果gpu数量大于1则分布计算, 此处为布尔值)

# 分GPU训练下的？
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    # print(sys.path)

    cfg.merge_from_file(args.config_file)
    # 对于不同的模型配置，有不同的超参设置，可以使用yaml文件来管理不同的configs，
    # merge_from_file()会比较每个模型特有的confi` g和默认参数的区别，将默认参数与特定参数不同的部分，用特定参数覆盖
    cfg.freeze()
    # 冻结参数

    output_dir = cfg.OUTPUT_DIR
    log_file_name = cfg.LOG_FILE_NAME
    log_file_path = join(output_dir, log_file_name)

# 记录输出文件？，这个类覆盖sys.Stdout或sys.Stderr，以便控制台日志可以也可以写入文件
    if is_main_process():
        ReDirectSTD(log_file_path, 'stdout', True)

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
        # 输出yaml文件内容
    print("Running with config:\n{}".format(cfg))
    print("The seed is {}".format(seed))
    # 输出训练配置 yaml+defaults.py

    model = train_model(cfg, args.local_rank, args.distributed)
    # local_rank == 0 distributed == False


if __name__ == '__main__':
    main()
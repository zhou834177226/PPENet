from os.path import join

import torch
from torch.utils.data import DataLoader
import numpy as np

from scipy import io
from sklearn import preprocessing

from GEMZSL.data.random_dataset import RandDataset
from GEMZSL.data.episode_dataset import EpiDataset, CategoriesSampler, DCategoriesSampler
from GEMZSL.data.test_dataset import TestDataset

from GEMZSL.data.transforms import data_transform

from GEMZSL.utils.comm import get_world_size


#
class ImgDatasetParam(object):
    DATASETS = {
        "imgroot": 'datasets',
        "dataroot": 'datasets/Data',
        "image_embedding": 'res101',
        "class_embedding": 'att'
    }

    @staticmethod
    # 使用 @staticmethod或 @classmethod，可以不需要实例化，直接类名.方法名()来调用
    # 静态方法不可以引用类中的属性或方法，其参数列表也不需要约定的默认参数self
    def get(dataset):
        attrs = ImgDatasetParam.DATASETS
        # 类嵌套？？
        # attrs["imgroot"] = join(attrs["imgroot"], dataset)
        attrs["imgroot"] = join("E:/PROJECT/GemDataset", dataset)
        # 合并图像数据路径
        args = dict(
            dataset=dataset
        )
        # 创立了字典args{'dataset':'CUB'},此处为传入关键字方法创建字典
        args.update(attrs)
        # args ： {'dataset': 'CUB',
        # 'imgroot': 'datasets\\CUB',
        # 'dataroot': 'datasets/Data',
        # 'image_embedding': 'res101',
        # 'class_embedding': 'att'}

        # update() 方法可使用一个字典dict所包含的键值对来更新己有的字典。
        # 在执行 update() 方法时，如果被更新的字典中己包含对应的键值对，那么原 value 会被覆盖；
        # 如果被更新的字典中不包含对应的键值对，则该键值对被添加进

        # update() 方法也用于修改当前集合set，可以添加新的元素或集合到当前集合中，
        # 如果添加的元素在集合中已存在，则该元素只会出现一次，重复的会忽略

        return args


def build_dataloader(cfg, is_distributed=False):
    args = ImgDatasetParam.get(cfg.DATASETS.NAME)
    # 创建对象
    imgroot = args['imgroot']
    dataroot = args['dataroot']
    image_embedding = args['image_embedding']
    class_embedding = args['class_embedding']
    dataset = args['dataset']
    # 读值

    matcontent = io.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    # datasets/Data/CUB/res101.mat 借用APN网络

    img_files = np.squeeze(matcontent['image_files'])
    # 1）numpy.squeeze(a,axis = None) a表示输入的数组；
    # 2）axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；
    # 3）axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目；
    # 4）返回值：数组
    # 5) 不会修改原数组
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]
        if dataset=='CUB':
            img_path = join(imgroot, '/'.join(img_path.split('/')[5:]))
        elif dataset=='AwA2':
            eff_path = img_path.split('/')[5:]
            eff_path.remove('')
            img_path = join(imgroot, '/'.join(eff_path))
        elif dataset=='SUN':
            img_path = join(imgroot, '/'.join(img_path.split('/')[7:]))
        new_img_files.append(img_path)

    new_img_files = np.array(new_img_files)
    # 转换为np数组
    label = matcontent['labels'].astype(int).squeeze() - 1
    # 创建标签

    matcontent = io.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    # 数据划分
    trainvalloc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    # 读取数据划分

    att_name = 'att'
    # if dataset == 'AwA2':
    #     att_name = 'original_att'
    cls_name = matcontent['allclasses_names']
    # 读取类别名称

    attribute = matcontent[att_name].T
    # 读类别的属性值

# 读训练数据
    train_img = new_img_files[trainvalloc]  # 图像路径
    train_label = label[trainvalloc].astype(int)
    train_att = attribute[train_label]
    # 划分训练集图片，类别，类别属性

    train_id, idx = np.unique(train_label, return_inverse=True)
    # numpy.unique(arr, return_index, return_inverse, return_counts)
    # arr：输入数组，如果不是一维数组则会展开
    # return_index：如果为 true，返回新列表元素在旧列表中的位置（下标），并以列表形式存储。
    # return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式存储。
    # return_counts：如果为 true，返回去重数组中的元素在原数组中的出现次数。

    train_att_unique = attribute[train_id]
    train_clsname = cls_name[train_id]
    # 训练集所含属性和训练集所含类别

    num_train = len(train_id)  # 训练的类别数,150
    train_label = idx # 训练样本的标签 50
    train_id = np.unique(train_label)  # 再赋值一遍训练样本的标签？

# 读测试未知类数据
    test_img_unseen = new_img_files[test_unseen_loc]
    test_label_unseen = label[test_unseen_loc].astype(int)
    # 读测试图像和标签
    test_id, idx = np.unique(test_label_unseen, return_inverse=True)
    att_unseen = attribute[test_id]
    test_clsname = cls_name[test_id]
    test_label_unseen = idx + num_train
    test_id = np.unique(test_label_unseen)
    # ？？？？对数据进行了怎样的处理

    # np.concatenate是numpy中对array进行拼接的函数，axis可以指定拼接的维度
    train_test_att = np.concatenate((train_att_unique, att_unseen))
    train_test_id = np.concatenate((train_id, test_id))

# 读测试已知类样本
    test_img_seen = new_img_files[test_seen_loc]
    test_label_seen = label[test_seen_loc].astype(int)
    _, idx = np.unique(test_label_seen, return_inverse=True)
    test_label_seen = idx

    att_unseen = torch.from_numpy(att_unseen).float()
    test_label_seen = torch.tensor(test_label_seen)
    test_label_unseen = torch.tensor(test_label_unseen)
    train_label = torch.tensor(train_label)
    att_seen = torch.from_numpy(train_att_unique).float()

    res = {
        'train_label': train_label,
        'train_att': train_att,
        'test_label_seen': test_label_seen,
        'test_label_unseen': test_label_unseen,
        'att_unseen': att_unseen,
        'att_seen': att_seen,
        'train_id': train_id,
        'test_id': test_id,
        'train_test_id': train_test_id,
        'train_clsname': train_clsname,
        'test_clsname': test_clsname
    }
# 'train_label':7057
# 'train_label': 7057
# 'train_att': 7057*312
# 'test_label_seen': 1764
# 'test_label_unseen': 2967
# 'att_unseen': 50*312
# 'att_seen': 150*312
# 'train_id': 150（0~149）
# 'test_id': 50 （150~199）
# 'train_test_id': 200（0~199
# 'train_clsname': 150（类名称）
# 'test_clsname':50（类名称）

    num_gpus = get_world_size()
# GPU群

    # train dataloader
    ways = cfg.DATASETS.WAYS
    # ways = 4 四类
    shots = cfg.DATASETS.SHOTS
    # shots = 2 两张
    data_aug_train = cfg.SOLVER.DATA_AUG
    # DATA_AUG: "resize_random_crop" 确认transform的类型
    img_size = cfg.DATASETS.IMAGE_SIZE
    # 448
    transforms = data_transform(data_aug_train, size=img_size)
    # 进行transform

    # MODE: 'episode'
    if cfg.DATALOADER.MODE == 'random':
        dataset = RandDataset(train_img, train_att, train_label, transforms)
        # 随机读入？
        if not is_distributed:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            # 随机抽取元素。如果没有替换，则从打乱的数据集中采样。 如果有替换，则用户可以指定:attr:num_samples
            # data_source (Dataset): 采样的数据集
            # replacement (bool): 如果为 True抽取的样本是有放回的。默认是False
            # num_samples (int): 抽取样本的数量，默认是len(dataset)。当replacement是 True的时应该被被实例化

            batch = ways*shots
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch, drop_last=True)
            tr_dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=8,
                batch_sampler=batch_sampler,
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            batch = ways * shots
            tr_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=8)

    elif cfg.DATALOADER.MODE == 'episode':
        n_batch = cfg.DATALOADER.N_BATCH  # = 300
        ep_per_batch = cfg.DATALOADER.EP_PER_BATCH  # = 1 episodes for each batch
        dataset = EpiDataset(train_img, train_att, train_label, transforms)
        # 重写dataset读入
        # 实现episode策略，选择support set和query set类别的过程
        # is_distributed默认distributed，有值，not is_distributed为False，执行else
        if not is_distributed:
            sampler = CategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )
        else:
            sampler = DCategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )
        tr_dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)

    data_aug_test = cfg.TEST.DATA_AUG
    # 设置transform格式
    transforms = data_transform(data_aug_test, size=img_size)
    # 进行transform
    test_batch_size = cfg.TEST.IMS_PER_BATCH
    # 设置测试大小

    if not is_distributed:
        # test unseen dataloader
        tu_data = TestDataset(test_img_unseen, test_label_unseen, transforms)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)

        # test seen dataloader
        ts_data = TestDataset(test_img_seen, test_label_seen, transforms)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)
    else:
        # test unseen dataloader
        tu_data = TestDataset(test_img_unseen, test_label_unseen, transforms)
        tu_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tu_data, shuffle=False)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, sampler=tu_sampler,
            num_workers=4, pin_memory=False)

        # test seen dataloader
        ts_data = TestDataset(test_img_seen, test_label_seen, transforms)
        ts_sampler = torch.utils.data.distributed.DistributedSampler(dataset=ts_data, shuffle=False)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, sampler=ts_sampler,
            num_workers=4, pin_memory=False)

    return tr_dataloader, tu_loader, ts_loader, res


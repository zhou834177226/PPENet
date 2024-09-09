import torch
import numpy as np

import torch.distributed as dist
from GEMZSL.utils.comm import *
from GEMZSL.engine.inferencer import eval_zs_gzsl

from torch.cuda.amp import autocast as autocast

# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for multi-precision via apex.amp')


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_train(
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
    ):

    best_performance = [0, 0, 0, 0]
    best_epoch = -1

    att_seen = res['att_seen'].to(device)

    losses = []
    cls_losses = []
    reg_losses = []
    ad_losses = []
    cpt_losses = []
    scale_all = []

    model.train()

    for epoch in range(0, max_epoch):
        print("开始训练")

        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):
            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)

            with autocast():
                loss_dict1, loss_dict2, loss_dict3, loss_dict4 = model(x=batch_img, att=batch_att, label=batch_label,
                                                                       seen_att=att_seen,)
                # loss_dict4 = model(x=batch_img, att=batch_att, label=batch_label, seen_att=att_seen, )

            scale = loss_dict4['scale']
            loss_dict4.pop('scale')

            # reduce losses over all GPUs for logging purposes
            # layer1
            loss_dict_reduced1 = reduce_loss_dict(loss_dict1)
            lreg1 = loss_dict_reduced1['Reg_loss']
            lcpt1 = loss_dict_reduced1['CPT_loss']
            losses_reduced1 = lamd[1] * lreg1 + lamd[3] * lcpt1

            # layer2
            loss_dict_reduced2 = reduce_loss_dict(loss_dict2)
            lreg2 = loss_dict_reduced2['Reg_loss']
            lcpt2 = loss_dict_reduced2['CPT_loss']
            losses_reduced2 = lamd[1] * lreg2 + lamd[3] * lcpt2

            # layer3
            loss_dict_reduced3 = reduce_loss_dict(loss_dict3)
            lreg3 = loss_dict_reduced3['Reg_loss']
            lcls3 = loss_dict_reduced3['Cls_loss']
            lcpt3 = loss_dict_reduced3['CPT_loss']
            losses_reduced3 = lcls3 + lamd[1] * lreg3 + lamd[3] * lcpt3

            loss_dict_reduced4 = reduce_loss_dict(loss_dict4)
            lreg4 = loss_dict_reduced4['Reg_loss']
            lcls4 = loss_dict_reduced4['Cls_loss']
            lad4 = loss_dict_reduced4['AD_loss']
            lcpt4 = loss_dict_reduced4['CPT_loss']
            losses_reduced4 = lcls4 + lamd[1] * lreg4 + lamd[2] * lad4 + lamd[3] * lcpt4

            losses_reduced = (losses_reduced1 + losses_reduced2 + losses_reduced3 + losses_reduced4)
            # losses_reduced = losses_reduced3

            optimizer.zero_grad()
            scaler.scale(losses_reduced).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        # 更改学习率

        if is_main_process():
            log_info = 'epoch: %d  |   lr: %.6f' % \
                       (epoch + 1,  optimizer.param_groups[0]["lr"])
            log_info1 = 'layer1 | reg_loss: %.4f, cpt_loss: %.4f, losses_reduced: %.4f ' % \
                       (lreg1, lcpt1, losses_reduced1)
            log_info2 = 'layer2 | reg_loss: %.4f, cpt_loss: %.4f, losses_reduced: %.4f ' % \
                       (lreg2, lcpt2, losses_reduced2)
            log_info3 = 'layer3 | cls_loss: %.4f , reg_loss: %.4f, cpt_loss: %.4f, cpt_loss: %.4f, losses_reduced: %.4f ' % \
                       (lcls3, lreg3, lcpt3, lad3, losses_reduced3)
            log_info4 = 'layer4 | cls_loss: %.4f , reg_loss: %.4f, cpt_loss: %.4f, ad_loss: %.4f, losses_reduced: %.4f ' % \
                       (lcls4, lreg4, lcpt4, lad4, losses_reduced4)
            print(log_info)
            print(log_info1)
            print(log_info2)
            print(log_info3)
            print(log_info4)

        synchronize()
        # 分布训练的同步文件
        torch.cuda.empty_cache()
        acc_zsl1, acc_zsl2, acc_zsl3, acc_zsl4, acc_unseen1, acc_unseen2, acc_unseen3, acc_unseen4 , acc_seen1, acc_seen2,acc_seen3,acc_seen4,H1,H2,H3,H4,acc_zsl,acc_unseen,acc_seen,H = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            device)

        # ———————————————————————修改清空内存———————————————————————————————
        # torch.cuda.empty_cache()

        # acc_seen4, acc_unseen4, H4, acc_zsl4 = eval_zs_gzsl(
        #     tu_loader,
        #     ts_loader,
        #     res,
        #     model,
        #     test_gamma,
        #     device)


        if is_main_process():
            print('layer1: zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zsl1, acc_seen1, acc_unseen1, H1))
            print('layer2: zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zsl2, acc_seen2, acc_unseen2, H2))
            print('layer3: zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zsl3, acc_seen3, acc_unseen3, H3))
            print('layer4: zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zsl4, acc_seen4, acc_unseen4, H4))
            for i in range(0, 10):
                print('fuse%.0f: zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (i, acc_zsl[i], acc_seen[i], acc_unseen[i], H[i]))

            if H4 > best_performance[-1]:
                # 储存的是H最高的
                best_performance[1:] = [acc_seen4, acc_unseen4, H4]
                # best_epoch = epoch + 1
                # data = {}
                # data["model"] = model.state_dict()
                # torch.save(data, model_file_path)
                # print('save model: ' + model_file_path)

            if acc_zsl4 > best_performance[0]:
                best_performance[0] = acc_zsl4
                best_epoch = epoch + 1
                data = {}
                data["model"] = model.state_dict()
                torch.save(data, model_file_path)
                print('save model: ' + model_file_path)

        # if is_main_process():
        #     print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zs, acc_seen, acc_novel, H))
        #
        # if H > best_performance[-1]:
        #     # 储存的是H最高的
        #     best_performance[1:] = [acc_seen, acc_novel, H]
        #     # best_epoch = epoch + 1
        #     # data = {}
        #     # data["model"] = model.state_dict()
        #     # torch.save(data, model_file_path)
        #     # print('save model: ' + model_file_path)
        #
        # if acc_zs > best_performance[0]:
        #     best_performance[0] = acc_zs
        #     best_epoch = epoch + 1
        #     data = {}
        #     data["model"] = model.state_dict()
        #     torch.save(data, model_file_path)
        #     print('save model: ' + model_file_path)

    if is_main_process():
        print("best: ep: %d" % best_epoch)
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % tuple(best_performance))
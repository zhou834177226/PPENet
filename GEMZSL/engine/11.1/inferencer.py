import torch
import numpy as np
from sklearn.metrics import accuracy_score

def cal_accuracy(model, dataloadr, att, test_id, device, bias=None):

    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores = [list() for i in range(0, 10)]
    pred = [list() for i in range(0, 10)]
    outpred = [list() for i in range(0, 10)]
    acc = [list() for i in range(0, 10)]

    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, label) in enumerate(dataloadr):
        img = img.to(device)
        score1, score2, score3, score4, score = model(img, seen_att=att)

        scores1.append(score1)
        scores2.append(score2)
        scores3.append(score3)
        scores4.append(score4)
        for i in range(0, 10):
            scores[i].append(score[i])
        labels.append(label)

    scores1 = torch.cat(scores1, dim=0)
    scores2 = torch.cat(scores2, dim=0)
    scores3 = torch.cat(scores3, dim=0)
    scores4 = torch.cat(scores4, dim=0)
    for i in range(0, 10):
        scores[i] = torch.cat(scores[i], dim=0)

    labels = torch.cat(labels, dim=0)

    if bias is not None:
        scores1 = scores1-bias
        scores2 = scores2 - bias
        scores3 = scores3 - bias
        scores4 = scores4 - bias
        for i in range(0, 10):
            scores[i] = scores[i] - bias

    _, pred1 = scores1.max(dim=1)
    _, pred2 = scores2.max(dim=1)
    _, pred3 = scores3.max(dim=1)
    _, pred4 = scores4.max(dim=1)
    for i in range(0, 10):
        _, pred[i] = scores[i].max(dim=1)
    pred1 = pred1.view(-1).to(cpu)
    pred2 = pred2.view(-1).to(cpu)
    pred3 = pred3.view(-1).to(cpu)
    pred4 = pred4.view(-1).to(cpu)
    for i in range(0, 10):
        pred[i] = pred[i].view(-1).to(cpu)

    outpred1 = test_id[pred1]
    outpred2 = test_id[pred2]
    outpred3 = test_id[pred3]
    outpred4 = test_id[pred4]
    for i in range(0, 10):
        outpred[i] = test_id[pred[i]]
    outpred1 = np.array(outpred1, dtype='int')
    outpred2 = np.array(outpred2, dtype='int')
    outpred3 = np.array(outpred3, dtype='int')
    outpred4 = np.array(outpred4, dtype='int')
    for i in range(0, 10):
        outpred[i] = np.array(outpred[i], dtype='int')
    labels = labels.numpy()
    unique_labels = np.unique(labels)
    acc1 = 0
    acc2 = 0
    acc3 = 0
    acc4 = 0
    for i in range(0, 10):
        acc[i] = 0
    for l in unique_labels:
        idx = np.nonzero(labels == l)[0]
        acc1 += accuracy_score(labels[idx], outpred1[idx])
        acc2 += accuracy_score(labels[idx], outpred2[idx])
        acc3 += accuracy_score(labels[idx], outpred3[idx])
        acc4 += accuracy_score(labels[idx], outpred4[idx])
        for i in range(0, 10):
            acc[i] += accuracy_score(labels[idx], outpred[i][idx])
    acc1 = acc1 / unique_labels.shape[0]
    acc2 = acc2 / unique_labels.shape[0]
    acc3 = acc3 / unique_labels.shape[0]
    acc4 = acc4 / unique_labels.shape[0]
    for i in range(0, 10):
        acc[i] = acc[i] / unique_labels.shape[0]
    return acc1, acc2, acc3, acc4, acc

def eval(
        tu_loader,
        ts_loader,
        att_unseen,
        att_seen,
        cls_unseen_num,
        cls_seen_num,
        test_id,
        train_test_id,
        model,
        test_gamma,
        device
):

    acc_zsl1, acc_zsl2, acc_zsl3, acc_zsl4, acc_zsl = cal_accuracy(model=model, dataloadr=tu_loader, att=att_unseen, test_id=test_id, device=device, bias=None)

    bias_s = torch.zeros((1, cls_seen_num)).fill_(test_gamma).to(device)
    bias_u = torch.zeros((1, cls_unseen_num)).to(device)
    bias = torch.cat([bias_s, bias_u], dim=1)

    att = torch.cat((att_seen, att_unseen), dim=0)
    acc_unseen1, acc_unseen2, acc_unseen3, acc_unseen4, acc_unseen = cal_accuracy(model=model, dataloadr=tu_loader, att=att,
                                   test_id=train_test_id, device=device, bias=bias)
    acc_seen1, acc_seen2, acc_seen3, acc_seen4, acc_seen = cal_accuracy(model=model, dataloadr=ts_loader, att=att,
                                   test_id=train_test_id, device=device, bias=bias)

    H1 = 2 * acc_seen1 * acc_unseen1 / (acc_seen1 + acc_unseen1)
    H2 = 2 * acc_seen2 * acc_unseen2 / (acc_seen2 + acc_unseen2)
    H3 = 2 * acc_seen3 * acc_unseen3 / (acc_seen3 + acc_unseen3)
    H4 = 2 * acc_seen4 * acc_unseen4 / (acc_seen4 + acc_unseen4)

    H = [list() for i in range(0, 10)]
    for i in range(0, 10):
        H[i] = 2 * acc_seen[i] * acc_unseen[i] / (acc_seen[i] + acc_unseen[i])


    return acc_zsl1, acc_zsl2, acc_zsl3, acc_zsl4, acc_unseen1, acc_unseen2, acc_unseen3, acc_unseen4 , acc_seen1, acc_seen2,acc_seen3,acc_seen4,H1,H2,H3,H4,acc_zsl,acc_unseen,acc_seen,H


def eval_zs_gzsl(
        tu_loader,
        ts_loader,
        res,
        model,
        test_gamma,
        device
):
    model.eval()
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)

    test_id = res['test_id']
    train_test_id = res['train_test_id']

    cls_seen_num = att_seen.shape[0]
    cls_unseen_num = att_unseen.shape[0]

    with torch.no_grad():
        acc_zsl1, acc_zsl2, acc_zsl3, acc_zsl4, acc_unseen1, acc_unseen2, acc_unseen3, acc_unseen4 , acc_seen1, acc_seen2,acc_seen3,acc_seen4,H1,H2,H3,H4,acc_zsl,acc_unseen,acc_seen,H = eval(
            tu_loader,
            ts_loader,
            att_unseen,
            att_seen,
            cls_unseen_num,
            cls_seen_num,
            test_id,
            train_test_id,
            model,
            test_gamma,
            device
        )

    model.train()

    return acc_zsl1, acc_zsl2, acc_zsl3, acc_zsl4, acc_unseen1, acc_unseen2, acc_unseen3, acc_unseen4 , acc_seen1, acc_seen2,acc_seen3,acc_seen4,H1,H2,H3,H4,acc_zsl,acc_unseen,acc_seen,H
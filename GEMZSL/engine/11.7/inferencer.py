import torch
import numpy as np
from sklearn.metrics import accuracy_score

def cal_accuracy(model, dataloadr, att, test_id, device, bias=None):

    scores = [list() for i in range(0, 11)]
    pred = [list() for i in range(0, 11)]
    outpred = [list() for i in range(0, 11)]
    acc = [list() for i in range(0, 11)]

    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, label) in enumerate(dataloadr):
        img = img.to(device)
        score = model(img, seen_att=att)
        for i in range(0, 11):
            scores[i].append(score[i])
        labels.append(label)

    for i in range(0, 11):
        scores[i] = torch.cat(scores[i], dim=0)

    labels = torch.cat(labels, dim=0)

    if bias is not None:
        for i in range(0, 11):
            scores[i] = scores[i] - bias

    for i in range(0, 11):
        _, pred[i] = scores[i].max(dim=1)

    for i in range(0, 11):
        pred[i] = pred[i].view(-1).to(cpu)

    for i in range(0, 11):
        outpred[i] = test_id[pred[i]]

    for i in range(0, 11):
        outpred[i] = np.array(outpred[i], dtype='int')

    labels = labels.numpy()
    unique_labels = np.unique(labels)

    for i in range(0, 11):
        acc[i] = 0
    for l in unique_labels:
        idx = np.nonzero(labels == l)[0]
        for i in range(0, 11):
            acc[i] += accuracy_score(labels[idx], outpred[i][idx])

    for i in range(0, 11):
        acc[i] = acc[i] / unique_labels.shape[0]
    return acc

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

    acc_zsl = cal_accuracy(model=model, dataloadr=tu_loader, att=att_unseen, test_id=test_id, device=device, bias=None)

    bias_s = torch.zeros((1, cls_seen_num)).fill_(test_gamma).to(device)
    bias_u = torch.zeros((1, cls_unseen_num)).to(device)
    bias = torch.cat([bias_s, bias_u], dim=1)

    att = torch.cat((att_seen, att_unseen), dim=0)
    acc_unseen = cal_accuracy(model=model, dataloadr=tu_loader, att=att,
                                   test_id=train_test_id, device=device, bias=bias)
    acc_seen = cal_accuracy(model=model, dataloadr=ts_loader, att=att,
                                   test_id=train_test_id, device=device, bias=bias)

    H = [list() for i in range(0, 11)]
    for i in range(0, 11):
        H[i] = 2 * acc_seen[i] * acc_unseen[i] / (acc_seen[i] + acc_unseen[i])

    return acc_zsl, acc_unseen, acc_seen, H


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
        acc_zsl, acc_unseen, acc_seen, H = eval(
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

    return acc_zsl, acc_unseen, acc_seen, H
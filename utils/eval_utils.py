import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn

# ====================Metrics==============================
def RecallPrecision_atK(test, r, k):
    tp = r[:, :k].sum(1)
    precision = np.sum(tp) / k
    recall_n = np.array([len(test[i]) for i in range(len(test))])
    recall = np.sum(tp / recall_n)
    return precision, recall


def MRR_atK(test, r, k):
    pred = r[:, :k]
    weight = np.arange(1, k+1)
    MRR = np.sum(pred / weight, axis=1) / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MRR = np.sum(MRR)
    return MRR


def MAP_atK(test, r, k):
    pred = r[:, :k]
    rank = pred.copy()
    for i in range(k):
        rank[:, k - i - 1] = np.sum(rank[:, :k - i], axis=1)
    weight = np.arange(1, k+1)
    AP = np.sum(pred * rank / weight, axis=1)
    AP = AP / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MAP = np.sum(AP)
    return MAP


def NDCG_atK(test, r, k):
    pred = r[:, :k]
    test_mat = np.zeros((len(pred), k))
    for i, items in enumerate(test):
        length = k if k <= len(items) else len(items)
        test_mat[i, :length] = 1

    idcg = np.sum(test_mat * (1. / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    dcg = pred * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    ndcg = np.sum(ndcg)
    return ndcg


def AUC(all_item_scores, dataset, test):
    r_all = np.zeros((dataset.m_item, ))
    r_all[test] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test, pred):
    r = []
    for i in range(len(test)):
        groundTruth, predTopK = test[i], pred[i]
        hits = list(map(lambda x: x in groundTruth, predTopK))
        hits = np.array(hits).astype("float")
        r.append(hits)
    return np.array(r).astype('float')
# ====================end Metrics=============================
def get_sample_scores(pred_list):
    pred_list = (-pred_list).argsort().argsort()[:, 0]
    HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
    HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
    HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
    return HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def choose_predict(predict_d1,predict_d2,domain_id):
    predict_d1_cse, predict_d2_cse = [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i] == 0:
            predict_d1_cse.append(predict_d1[i,:])
        else:
            predict_d2_cse.append(predict_d2[i,:])
    if len(predict_d1_cse)!=0:
        predict_d1_cse = np.array(predict_d1_cse)
    if len(predict_d2_cse)!=0:
        predict_d2_cse = np.array(predict_d2_cse)
    return predict_d1_cse, predict_d2_cse

def choose_predict_overlap(predict_d1,predict_d2,domain_id,overlap_label):
    predict_d1_cse_over, predict_d1_cse_nono, predict_d2_cse_over, predict_d2_cse_nono = [], [], [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i] == 0:
            if overlap_label[i][0]==0:
                predict_d1_cse_nono.append(predict_d1[i,:])
            else:
                predict_d1_cse_over.append(predict_d1[i,:])
        else:
            if overlap_label[i][0]==0:
                predict_d2_cse_nono.append(predict_d2[i,:])
            else:
                predict_d2_cse_over.append(predict_d2[i,:])
    if len(predict_d1_cse_over)!=0:
        predict_d1_cse_over = np.array(predict_d1_cse_over)
    if len(predict_d1_cse_nono)!=0:
        predict_d1_cse_nono = np.array(predict_d1_cse_nono)
    if len(predict_d2_cse_over)!=0:
        predict_d2_cse_over = np.array(predict_d2_cse_over)
    if len(predict_d2_cse_nono)!=0:
        predict_d2_cse_nono = np.array(predict_d2_cse_nono)
    return predict_d1_cse_over, predict_d1_cse_nono, predict_d2_cse_over, predict_d2_cse_nono

def compute_metrics(pred):
    logits = pred.predictions
    # print("logits shape:{}".format(logits.shape))
    if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
        HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = -1, -1, -1, -1, -1, -1, -1
    elif pred.label_ids is not None and logits.ndim == 2:
        labels = np.asarray(pred.label_ids).reshape(-1)
        logits = logits.copy()
        logits[:, 0] = -np.inf
        ranks = (-logits).argsort(axis=1)
        answer_ranks = np.array([np.where(ranks[i] == labels[i])[0][0] for i in range(len(labels))])
        HIT_1, NDCG_1, MRR = get_metric(answer_ranks, 1)
        HIT_5, NDCG_5, MRR = get_metric(answer_ranks, 5)
        HIT_10, NDCG_10, MRR = get_metric(answer_ranks, 10)
    else:
        HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = get_sample_scores(logits)
    return {
        'hit@1':HIT_1,
        'hit@5':HIT_5,
        'ndcg@5':NDCG_5,
        'hit@10':HIT_10,
        'ndcg@10':NDCG_10,
        'mrr':MRR,
    }

def compute_metrics_multiple(pred):
    logits = pred.predictions 
    print(logits.shape)
    loss = logits[:,:,-1]
    predict = logits[:,:,:-1]
    # if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
    #     HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = -1, -1, -1, -1, -1, -1, -1
    # else:
    HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = [],[],[],[],[],[],[]
    for i in range(predict.shape[1]):
        HIT_1_tmp, NDCG_1_tmp, HIT_5_tmp, NDCG_5_tmp, HIT_10_tmp, NDCG_10_tmp, MRR_tmp = get_sample_scores(predict[:,i,:])
        HIT_1.append(HIT_1_tmp)
        NDCG_1.append(NDCG_1_tmp)
        HIT_5.append(HIT_5_tmp)
        NDCG_5.append(NDCG_5_tmp)
        HIT_10.append(HIT_10_tmp)
        NDCG_10.append(NDCG_10_tmp)
        MRR.append(MRR_tmp)
    print("mrr:{}".format(MRR))
    return {
        'hit@1':HIT_1,
        'hit@5':HIT_5,
        'ndcg@5':NDCG_5,
        'hit@10':HIT_10,
        'ndcg@10':NDCG_10,
        'mrr':MRR,
        'loss':np.mean(loss,axis=0),
    }

def get_full_sort_score(answers, pred_list):
    recall, ndcg, mrr = [], [], []
    for k in [5, 10, 15, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
        mrr.append(MRR_atK(answers, pred_list, k))
    return recall, ndcg, mrr

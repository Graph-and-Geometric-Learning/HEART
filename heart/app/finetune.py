import os
import copy
import wandb
import random
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from heart.utils import set_random_seed
from heart.utils.token_utils import EHRTokenizer
from heart.utils.dataset_utils import HBERTFinetuneEHRDataset, batcher
from heart.models.HBERT import HBERT_Finetune
from heart.utils.metric_utils import eval_precisionk, eval_recallk, eval_ndcgk

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@torch.no_grad()
def evaluate(model, dataloader, device, task_type="binary"):
    model.eval()
    predicted_scores, gt_labels = [], []
    for step, batch in enumerate(dataloader):
        batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        labels = batch[-1]
        output_logits = model(*batch[:-1])
        predicted_scores.append(output_logits)
        gt_labels.append(labels)
    
    if task_type == "binary":
        predicted_scores = torch.cat(predicted_scores, dim=0).view(-1)
        gt_labels = torch.cat(gt_labels, dim=0).view(-1).cpu().numpy()
        scores = predicted_scores.cpu().numpy()
        predicted_labels = (predicted_scores > 0).float().cpu().numpy()

        precision = (predicted_labels * gt_labels).sum() / (predicted_labels.sum() + 1e-8)
        recall = (predicted_labels * gt_labels).sum() / (gt_labels.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        roc_auc = roc_auc_score(gt_labels, scores)
        precision_curve, recall_curve, _ = precision_recall_curve(gt_labels, scores)
        pr_auc = auc(recall_curve, precision_curve)

        return {"precision":precision, "recall":recall, "f1":f1, "roc_auc":roc_auc, "pr_auc":pr_auc}
    else:
        predicted_scores = torch.cat(predicted_scores, dim=0).cpu()  # [B, -1]
        gt_labels = torch.cat(gt_labels, dim=0).cpu()

        ave_f1, ave_auc, ave_prauc, ave_recall, ave_precision = [], [], [], [], []
        for i in range(predicted_scores.size(0)):
            scores, labels = predicted_scores[i].squeeze().clone(), gt_labels[i].squeeze().clone()

            predicted_labels = (scores > 0).float().cpu().numpy()
            labels = labels.float().cpu().numpy()
            precision = (predicted_labels * labels).sum() / (predicted_labels.sum() + 1e-8)
            recall = (predicted_labels * labels).sum() / (labels.sum() + 1e-8)
            ave_f1.append(2 * precision * recall / (precision + recall + 1e-8))
            ave_auc.append(roc_auc_score(labels, scores))
            precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
            ave_prauc.append(auc(recall_curve, precision_curve))
            ave_recall.append(recall)
            ave_precision.append(precision)

        ave_f1, ave_auc, ave_prauc, ave_recall, ave_precision = np.mean(ave_f1), np.mean(ave_auc), np.mean(ave_prauc), np.mean(ave_recall), np.mean(ave_precision)
        return {"recall":ave_recall, "precision":ave_precision, "f1":ave_f1, "auc":ave_auc, "prauc":ave_prauc}


def read_data(args, all_data_path, finetune_data_path):
    ehr_data = pickle.load(open(all_data_path, 'rb'))
    diag_sentences = ehr_data["ICD9_CODE"].values.tolist()
    med_sentences = ehr_data["NDC"].values.tolist()
    lab_sentences = ehr_data["LAB_TEST"].values.tolist()
    if args.dataset == "mimic":
        pro_sentences = ehr_data["PRO_CODE"].values.tolist()
        gender_set = [["M"], ["F"]]
        age_gender_set = [[str(c) + "_" + gender] for c in set(ehr_data["AGE"].values.tolist()) for gender in ["M", "F"]]
    else:
        pro_sentences = None
        gender_set = [["Female"], ["Male"], ["Unknown"], ["Other"]]
        age_gender_set = [[str(c) + "_" + gender] for c in set(ehr_data["AGE"].values.tolist()) for gender in ["Female", "Male", "Unknown", "Other"]]
    age_set = [[c] for c in set(ehr_data["AGE"].values.tolist())]    

    tokenizer = EHRTokenizer(diag_sentences, med_sentences, lab_sentences, pro_sentences, gender_set, age_set, age_gender_set, special_tokens=args.special_tokens)
    if args.dataset == "mimic":
        tokenizer.build_tree()

    train_data, val_data, test_data = pickle.load(open(finetune_data_path, 'rb'))
    train_dataset = HBERTFinetuneEHRDataset(train_data, tokenizer, token_type=args.predicted_token_type, task=args.task)
    val_dataset = HBERTFinetuneEHRDataset(val_data, tokenizer, token_type=args.predicted_token_type, task=args.task)
    test_dataset = HBERTFinetuneEHRDataset(test_data, tokenizer, token_type=args.predicted_token_type, task=args.task)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=batcher(tokenizer, is_train=False), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, collate_fn=batcher(tokenizer, is_train=False), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=batcher(tokenizer, is_train=False), shuffle=False)

    return tokenizer, train_dataloader, val_dataloader, test_dataloader


def main():
    parser = argparse.ArgumentParser(description='HBERT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="mimic", help="mimic,eicu")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--task', type=str, default="death", help="death,stay,readmission,next_diag_6m,next_diag_12m")
    parser.add_argument('--pretrain_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--encoder', type=str, default="hi_edge", help='hi,hi_edge')
    parser.add_argument('--pretrain_mask_rate', type=float, default=0.5)
    parser.add_argument('--pretrain_anomaly_rate', type=float, default=0.05)
    parser.add_argument('--pretrain_anomaly_loss_weight', type=float, default=1)
    parser.add_argument('--pretrain_pos_weight', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--num_hidden_layers', type=int, default=5)
    parser.add_argument('--num_attention_heads', type=int, default=6)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--edge_hidden_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=288, help='ensure that hidden_size is divisible by num_attention_heads')
    parser.add_argument('--intermediate_size', type=int, default=288)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--gnn_n_heads', type=int, default=1)
    parser.add_argument('--gnn_temp', type=float, default=1)
    parser.add_argument('--gat', type=str, default="dotattn", help="dotattn")
    parser.add_argument('--diag_med_emb', type=str, default="tree", help="simple,tree")
    args = parser.parse_args()
    print(args)

    set_random_seed(args.seed)

    exp_name = "HBERT" \
        + "-" + str(args.encoder) \
        + "-" + str(args.pretrain_mask_rate) \
        + "-" + str(args.pretrain_anomaly_rate) \
        + "-" + str(args.pretrain_anomaly_loss_weight) \
        + "-" + str(args.pretrain_pos_weight) \
        + "-" + str(args.hidden_size) \
        + "-" + str(args.edge_hidden_size) \
        + "-" + str(args.num_hidden_layers) \
        + "-" + str(args.num_attention_heads) \
        + "-" + str(args.attention_probs_dropout_prob) \
        + "-" + str(args.hidden_dropout_prob) \
        + "-" + str(args.intermediate_size) \
        + "-" + str(args.gat) \
        + "-" + str(args.gnn_n_heads) \
        + "-" + str(args.gnn_temp) \
        + "-" + str(args.diag_med_emb) \

    pretrained_weight_path = "/home/username/ehr_bert/saved_model/" + "Pretrain-" + exp_name + f"/pretrained_{args.pretrain_epoch}.pt"
    
    finetune_exp_name = f"Finetune-{args.task}-" + exp_name
    save_path = "/home/username/ehr_bert/saved_model/" + finetune_exp_name
    if args.save_model and not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == "mimic":
        args.predicted_token_type = ["diag", "med", "pro", "lab"]
        args.special_tokens = ("[PAD]", "[CLS]", "[SEP]", "[MASK0]", "[MASK1]", "[MASK2]", "[MASK3]")
        all_data_path = "/home/username/ehr_bert/dataset/mimic.pkl"  # for tokenizer
        if args.task == "next_diag_6m":
            finetune_data_path = "/home/username/ehr_bert/dataset/mimic_nextdiag_6m_0.1.pkl"
        elif args.task == "next_diag_12m":
            finetune_data_path = "/home/username/ehr_bert/dataset/mimic_nextdiag_12m_0.1.pkl"
        else:    
            finetune_data_path = "/home/username/ehr_bert/dataset/mimic_downstream_0.1.pkl"
        args.max_visit_size = 15
    else:
        args.predicted_token_type = ["diag", "med", "lab"]
        args.special_tokens = ("[PAD]", "[CLS]", "[SEP]", "[MASK0]", "[MASK1]", "[MASK2]")
        all_data_path = "/home/username/ehr_bert/dataset/eicu.pkl"  # for tokenizer
        finetune_data_path = "/home/username/ehr_bert/dataset/eicu_downstream.pkl"
        args.max_visit_size = 24

    tokenizer, train_dataloader, val_dataloader, test_dataloader = read_data(args, all_data_path, finetune_data_path)
    logging.info(f"load data from {finetune_data_path}")

    args.vocab_size = len(args.special_tokens) + \
                    len(tokenizer.diag_voc.idx2word) + \
                    len(tokenizer.pro_voc.idx2word) + \
                    len(tokenizer.med_voc.idx2word) + \
                    len(tokenizer.lab_voc.idx2word) + \
                    len(tokenizer.age_voc.idx2word) + \
                    len(tokenizer.gender_voc.idx2word) + \
                    len(tokenizer.age_gender_voc.idx2word)

    args.label_vocab_size = len(tokenizer.diag_voc.idx2word)  # only for diagnosis

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = HBERT_Finetune(args, tokenizer)
    logging.info(f"initialize model")
    if args.pretrain_epoch > 0:
        model.load_weight(torch.load(pretrained_weight_path))
        logging.info(f"load pretrained model from pretrained_{args.pretrain_epoch}.pt")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.task in ["death", "stay"]:
        eval_metric = "f1"
        task_type = "binary"
        loss_fn = F.binary_cross_entropy_with_logits
    elif args.task == "readmission":
        eval_metric = "f1"
        task_type = "binary"
        loss_fn = F.binary_cross_entropy_with_logits
    else:
        eval_metric = "prauc"
        task_type = "l2r"
        loss_fn = lambda x, y: F.binary_cross_entropy_with_logits(x, y)

    if args.use_wandb:
        wandb.init(project="ehr_bert", name=finetune_exp_name)
        wandb.config.update(args)
        wandb.watch(model, log='all')

    best_score, best_val_metric, best_test_metric = 0., None, None

    for epoch in range(1, 1 + args.epochs):
        train_iter = tqdm(train_dataloader, ncols=140)
        model.train()
        ave_loss = 0.

        for step, batch in enumerate(train_iter):
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            labels = batch[-1].float()
            output_logits = model(*batch[:-1])

            loss = loss_fn(output_logits.view(-1), labels.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_iter.set_description(f"Epoch:{epoch: 03d}, Step:{step: 03d}, loss:{loss.item():.4f}")
            ave_loss += loss.item()

        ave_loss /= (step + 1)
        val_metric = evaluate(model, val_dataloader, device, task_type=task_type)
        test_metric = evaluate(model, test_dataloader, device, task_type=task_type)
        print(f"Epoch:{epoch: 03d}, average loss:{ave_loss:.4f}")
        print(val_metric)
        print(test_metric)

        if val_metric[eval_metric] > best_score:
            best_score = val_metric[eval_metric]
            best_val_metric = val_metric
            best_test_metric = test_metric

        if args.use_wandb:
            record_dict = {f"loss": ave_loss}
            record_dict.update({f"val_{k}":v for k, v in val_metric})
            record_dict.update({f"test_{k}":v for k, v in test_metric})
            wandb.log(record_dict)

        if args.save_model:
            torch.save(model.cpu().state_dict(), f"{save_path}/{epoch}.pt")
            logging.info(f"save model to {save_path}/{epoch}.pt")
            model.to(device)

    print("-----------------")
    print(f"best val metric: {best_val_metric}")
    print(f"best test metric: {best_test_metric}")

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
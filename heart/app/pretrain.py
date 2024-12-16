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

import torch
from torch.utils.data import DataLoader

from heart.utils import set_random_seed
from heart.utils.token_utils import EHRTokenizer
from heart.utils.dataset_utils import HBERTPretrainEHRDataset, batcher
from heart.model.HBERT import HBERT_Pretrain

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def read_data(args, all_data_path, pretrain_data_path):
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
    
    ehr_pretrain_data = pickle.load(open(pretrain_data_path, 'rb'))
    tokenizer = EHRTokenizer(diag_sentences, med_sentences, lab_sentences, pro_sentences, gender_set, age_set, age_gender_set, special_tokens=args.special_tokens)
    if args.dataset == "mimic":
        tokenizer.build_tree()
    dataset = HBERTPretrainEHRDataset(ehr_pretrain_data, tokenizer, token_type=args.predicted_token_type, mask_rate=args.mask_rate)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=batcher(tokenizer, n_token_type=len(args.predicted_token_type)), shuffle=True)
    return tokenizer, dataloader


def main():
    parser = argparse.ArgumentParser(description='HBERT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="mimic", help="mimic,eicu")
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--use_wandb', action="store_true", default=False)
    parser.add_argument('--encoder', type=str, default="hi_edge", help='hi,hi_edge')
    parser.add_argument('--mask_rate', type=float, default=0.7)
    parser.add_argument('--anomaly_rate', type=float, default=0.05)
    parser.add_argument('--anomaly_loss_weight', type=float, default=1)
    parser.add_argument('--pos_weight', type=float, default=1)
    parser.add_argument('--num_hidden_layers', type=int, default=5)
    parser.add_argument('--num_attention_heads', type=int, default=6)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--edge_hidden_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=288, help='ensure that hidden_size is divisible by num_attention_heads')
    parser.add_argument('--intermediate_size', type=int, default=288)
    parser.add_argument('--gnn_n_heads', type=int, default=1)
    parser.add_argument('--gnn_temp', type=float, default=1)
    parser.add_argument('--gat', type=str, default="dotattn", help="dotattn,None")
    parser.add_argument('--diag_med_emb', type=str, default="tree", help="simple,tree")
    args = parser.parse_args()
    print(args)

    set_random_seed(args.seed)

    exp_name = "Pretrain-HBERT" \
        + "-" + str(args.dataset) \
        + "-" + str(args.encoder) \
        + "-" + str(args.mask_rate) \
        + "-" + str(args.anomaly_rate) \
        + "-" + str(args.anomaly_loss_weight) \
        + "-" + str(args.pos_weight) \
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
    
    save_path = "/home/username/ehr_bert/saved_model/" + exp_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == "mimic":
        args.max_visit_size = 15
        args.predicted_token_type = ["diag", "med", "pro", "lab"]
        args.mask_token_id = {"diag":3, "med":4, "pro":5, "lab":6}  # {token_type: masked_id}
        args.special_tokens = ("[PAD]", "[CLS]", "[SEP]", "[MASK0]", "[MASK1]", "[MASK2]", "[MASK3]")
        
        data_path = "/home/username/ehr_bert/dataset/mimic.pkl"  # for tokenizer
        pretrain_data_path = "/home/username/ehr_bert/dataset/mimic_pretrain.pkl"
        tokenizer, dataloader = read_data(args, data_path, pretrain_data_path)
        logging.info(f"load data from {pretrain_data_path}")

        args.vocab_size = 7 + len(tokenizer.diag_voc.idx2word) + \
                        len(tokenizer.pro_voc.idx2word) + \
                        len(tokenizer.med_voc.idx2word) + \
                        len(tokenizer.lab_voc.idx2word) + \
                        len(tokenizer.age_voc.idx2word) + \
                        len(tokenizer.gender_voc.idx2word) + \
                        len(tokenizer.age_gender_voc.idx2word)

        args.label_vocab_size = {"diag":len(tokenizer.diag_voc.idx2word), 
                                "pro":len(tokenizer.pro_voc.idx2word), 
                                "med":len(tokenizer.med_voc.idx2word), 
                                "lab":len(tokenizer.lab_voc.idx2word)}  # {token_type: vocab_size}

        loss_entity = ["diag", "med", "pro", "lab", "anomaly"]
    
    elif args.dataset == "eicu":
        args.max_visit_size = 24
        args.predicted_token_type = ["diag", "med", "lab"]
        args.mask_token_id = {"diag":3, "med":4, "lab":5}  # {token_type: masked_id}
        args.special_tokens = ("[PAD]", "[CLS]", "[SEP]", "[MASK0]", "[MASK1]", "[MASK2]")
        
        data_path = "/home/username/ehr_bert/dataset/eicu.pkl"  # for tokenizer
        pretrain_data_path = "/home/username/ehr_bert/dataset/eicu_pretrain.pkl"
        tokenizer, dataloader = read_data(args, data_path, pretrain_data_path)
        logging.info(f"load data from {pretrain_data_path}")

        args.vocab_size = 6 + len(tokenizer.diag_voc.idx2word) + \
                        len(tokenizer.med_voc.idx2word) + \
                        len(tokenizer.lab_voc.idx2word) + \
                        len(tokenizer.age_voc.idx2word) + \
                        len(tokenizer.gender_voc.idx2word) + \
                        len(tokenizer.age_gender_voc.idx2word)

        args.label_vocab_size = {"diag":len(tokenizer.diag_voc.idx2word), 
                                "med":len(tokenizer.med_voc.idx2word), 
                                "lab":len(tokenizer.lab_voc.idx2word)}  # {token_type: vocab_size}

        loss_entity = ["diag", "med", "lab", "anomaly"]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = HBERT_Pretrain(args, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logging.info(f"initialize model")

    if args.use_wandb:
        wandb.init(project="ehr_bert", name=exp_name)
        wandb.config.update(args)
        wandb.watch(model, log='all')

    for epoch in range(1, 1 + args.epochs):
        train_iter = tqdm(dataloader, ncols=140)
        model.train()
        ave_loss, ave_loss_dict = 0., {token_type: 0. for token_type in loss_entity}

        for step, batch in enumerate(train_iter):
            # torch.autograd.set_detect_anomaly(True)

            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            loss, loss_dict = model(*batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if args.dataset == "mimic":
                train_iter.set_description(f"Epoch:{epoch: 03d}, Step:{step: 03d}, loss:{loss.item():.4f}, diag:{loss_dict['diag']:.4f}, med:{loss_dict['med']:.4f}, pro:{loss_dict['pro']:.4f}, lab:{loss_dict['lab']:.4f}, anomaly:{loss_dict['anomaly']:.4f}")
            elif args.dataset == "eicu":
                train_iter.set_description(f"Epoch:{epoch: 03d}, Step:{step: 03d}, loss:{loss.item():.4f}, diag:{loss_dict['diag']:.4f}, med:{loss_dict['med']:.4f}, lab:{loss_dict['lab']:.4f}, anomaly:{loss_dict['anomaly']:.4f}")
            
            ave_loss += loss.item()
            ave_loss_dict = {token_type: ave_loss_dict[token_type] + loss_dict[token_type] for token_type in loss_entity}

        ave_loss /= (step + 1)
        ave_loss_dict = {token_type: ave_loss_dict[token_type] / (step + 1) for token_type in loss_entity}

        if args.dataset == "mimic":
            print(f"Epoch:{epoch: 03d}, average loss:{ave_loss:.4f}, diag:{ave_loss_dict['diag']:.4f}, med:{ave_loss_dict['med']:.4f}, pro:{ave_loss_dict['pro']:.4f}, lab:{ave_loss_dict['lab']:.4f}, anomaly:{ave_loss_dict['anomaly']:.4f}")
        elif args.dataset == "eicu":
            print(f"Epoch:{epoch: 03d}, average loss:{ave_loss:.4f}, diag:{ave_loss_dict['diag']:.4f}, med:{ave_loss_dict['med']:.4f}, lab:{ave_loss_dict['lab']:.4f}, anomaly:{ave_loss_dict['anomaly']:.4f}")
        
        if args.use_wandb:
            record_dict = {f"loss": ave_loss}
            record_dict.update({f"loss_{token_type}": ave_loss_dict[token_type] for token_type in loss_entity})
            wandb.log(record_dict)

        if epoch % 5 == 0 or epoch == 1:
            torch.save(model.cpu().state_dict(), f"{save_path}/pretrained_{epoch}.pt")
            logging.info(f"save model to {save_path}/pretrained_{epoch}.pt")
            model.to(device)

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
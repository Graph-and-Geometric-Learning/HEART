from time import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
from utils.token_utils import EHRTokenizer
from utils.bert_dataset_utils import PretrainEHRDataset, FinetuneEHRDataset


def _pad_sequence(seqs, pad_id=0):
    # seqs: a list of tensor [n, m]
    max_len = max([x.shape[1] for x in seqs])
    return torch.cat([F.pad(x, (0, max_len - x.shape[1]), "constant", pad_id) for x in seqs], dim=0)


class HBERTPretrainEHRDataset(PretrainEHRDataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, token_type=['diag', 'med', 'pro', 'lab'], mask_rate=0.15, anomaly_rate=0.1):
        super().__init__(data_pd, tokenizer, token_type)
        
        self.mask_rate = mask_rate
        self.anomaly_rate = anomaly_rate
        self.token_type = token_type
        self.token_type_map = {i:t for i, t in enumerate(token_type)}

    def _id2multi_hot(self, ids, dim):
        multi_hot = torch.zeros(dim)
        multi_hot[ids] = 1
        return multi_hot

    def __getitem__(self, item):
        subject_id = list(self.records.keys())[item]

        input_tokens, token_types, masked_labels, anomaly_labels = [], [], [None for _ in range(len(self.token_type))], []
        for idx, adm in enumerate(self.records[subject_id]):  # each subject have multiple admissions, idx:visit id
            adm_tokens, adm_token_types, adm_masked_labels = [str(self.ages[subject_id][idx]) + "_" + str(self.genders[subject_id][0])], [0], []  # replace [CLS] token with age
            adm_anomaly_labels = []

            for i in range(len(adm)):  # one admission have many kinds of entities, [[diag], [med], ...]
                cur_tokens = list(adm[i])

                # randomly mask tokens
                non_special_tokens_idx = [idx for idx, x in enumerate(cur_tokens)]
                masked_tokens_idx = np.random.choice(non_special_tokens_idx, max(1, int(len(non_special_tokens_idx) * self.mask_rate)))
                masked_tokens = [cur_tokens[idx] for idx in masked_tokens_idx]
                masked_tokens_idx_ = set(masked_tokens_idx.tolist())  # for fast lookup
                non_masked_tokens = [cur_tokens[idx] for idx in non_special_tokens_idx if idx not in masked_tokens_idx_]

                # randomly replace tokens with other tokens
                if self.anomaly_rate > 0 and len(non_masked_tokens) > 0:
                    candidate_token_idx = [idx for idx, x in enumerate(non_masked_tokens)]
                    anomaly_tokens_idx = np.random.choice(candidate_token_idx, max(1, int(len(candidate_token_idx) * self.anomaly_rate)))
                    for ano_idx in anomaly_tokens_idx:
                        non_masked_tokens[ano_idx] = self.tokenizer.random_token(voc_type=self.token_type_map[i])
                        adm_anomaly_labels.append(len(adm_tokens) + ano_idx + 1)  # the position of the anomaly token, +1 for [MASK] tokens

                adm_tokens.extend([f"[MASK{i}]"] + non_masked_tokens)  # [[MASK1], diag1, diag2, [MASK2], med1, med2]
                adm_token_types.extend([i + 1] * (len(non_masked_tokens) + 1))  # [0, 1, 1, 2, 2]
                adm_masked_labels.append(masked_tokens)  # [[diag1, diag2], [med1, med2]]

            input_tokens.append(torch.tensor([self.tokenizer.convert_tokens_to_ids(adm_tokens)]))
            token_types.append(torch.tensor([adm_token_types]))
            for i in range(len(self.token_type)):
                label_ids = self.tokenizer.convert_tokens_to_ids(adm_masked_labels[i], voc_type=self.token_type_map[i])
                label_hop = self._id2multi_hot(label_ids, dim=self.tokenizer.token_number(self.token_type_map[i])).unsqueeze(dim=0)
                if masked_labels[i] is None:
                    masked_labels[i] = label_hop
                else:
                    masked_labels[i] = torch.cat([masked_labels[i], label_hop])
            
            if len(adm_anomaly_labels) > 0:
                anomaly_labels.append(self._id2multi_hot(adm_anomaly_labels, dim=len(adm_tokens)).unsqueeze(dim=0))
            else:
                anomaly_labels.append(torch.zeros(len(adm_tokens)).unsqueeze(dim=0))

        visit_positions = torch.tensor(list(range(len(input_tokens))))  # [0, 1, 2, ...]
        input_tokens = _pad_sequence(input_tokens, pad_id=self.tokenizer.vocab.word2idx["[PAD]"])
        token_types = _pad_sequence(token_types, pad_id=0)
        anomaly_labels = _pad_sequence(anomaly_labels, pad_id=0) if len(anomaly_labels) > 0 else None
        n_adms = len(input_tokens)
        if n_adms > 1:
            # create a fully connected graph between admission
            edge_index = torch.tensor([[i, j] for i in range(n_adms) for j in range(n_adms)]).t()  # [2, n_adms * n_adms]
        else:
            edge_index = torch.tensor([])
        return input_tokens, token_types, edge_index, visit_positions, masked_labels, anomaly_labels


class HBERTFinetuneEHRDataset(FinetuneEHRDataset):
    def __init__(self, data_pd, tokenizer, token_type=['diag', 'med', 'pro', 'lab'], task='death'):
        super().__init__(data_pd, tokenizer, token_type, task)

    def __getitem__(self, item):
        hadm_id = list(self.records.keys())[item]

        input_tokens, token_types = [], []
        for idx, adm in enumerate(self.records[hadm_id]):  # each subject have multiple admissions, idx:visit id
            adm_tokens = [str(self.ages[hadm_id][idx]) + "_" + self.genders[hadm_id][0]]  # replace [CLS] token with age
            # adm_tokens = [self.ages[hadm_id][idx]]  # replace [CLS] token with age
            adm_token_types = [0]

            for i in range(len(adm)):
                cur_tokens = list(adm[i])
                adm_tokens.extend(cur_tokens)
                adm_token_types.extend([i + 1] * len(cur_tokens))
            
            # input_tokens.append(torch.tensor([self.tokenizer.convert_tokens_to_ids(adm_tokens)]))
            input_tokens.append(adm_tokens)
            # token_types.append(torch.tensor([adm_token_types]))
            token_types.append(adm_token_types)

        if self.task == "death":
            # predict if the patient will die in the hospital
            labels = torch.tensor([self.labels[hadm_id][0]]).float()
        elif self.task == "stay":
            # predict if the patient will stay in the hospital for more than 7 days
            labels = (torch.tensor([self.labels[hadm_id][1]]) > 7).float()
        elif self.task == "readmission":
            # predict if the patient will be readmitted within 1 month
            labels = torch.tensor([self.labels[hadm_id][2]]).float()
        else:
            # predict the next diagnosis in 6 months or 12 months
            input_tokens[-1] = [input_tokens[-1][0]] + ["[MASK0]"] + input_tokens[-1][1:]
            token_types[-1] = [token_types[-1][0]] + [1] + token_types[-1][1:]
            label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.labels[hadm_id], voc_type='diag'))
            labels = torch.zeros(self.tokenizer.token_number(voc_type='diag')).long()
            labels[label_ids] = 1  # multi-hop vector

        visit_positions = torch.tensor(list(range(len(input_tokens))))  # [0, 1, 2, ...]
        input_tokens = [torch.tensor([self.tokenizer.convert_tokens_to_ids(x)]) for x in input_tokens]
        token_types = [torch.tensor([x]) for x in token_types]
        input_tokens = _pad_sequence(input_tokens, pad_id=self.tokenizer.vocab.word2idx["[PAD]"])
        token_types = _pad_sequence(token_types, pad_id=0)
        n_adms = len(input_tokens)
        if n_adms > 1:
            # create a fully connected graph between admission
            edge_index = torch.tensor([[i, j] for i in range(n_adms) for j in range(n_adms)]).t()  # [2, n_adms * n_adms]
        else:
            edge_index = torch.tensor([])
        
        return input_tokens, token_types, edge_index, visit_positions, labels


def batcher(tokenizer, n_token_type=3, is_train=True):
    def batcher_dev(batch):
        raw_input_ids, raw_input_types, raw_edge_indexs, raw_visit_positions, raw_labels = [feat[0] for feat in batch], [feat[1] for feat in batch], [feat[2] for feat in batch], [feat[3] for feat in batch], [feat[4] for feat in batch]

        pad_id = tokenizer.vocab.word2idx["[PAD]"]
        max_n_tokens = max([x.size(1) for x in raw_input_ids])
        input_ids = torch.cat([F.pad(raw_input_id, (0, max_n_tokens - raw_input_id.size(1)), "constant", pad_id) for raw_input_id in raw_input_ids], dim=0)

        max_n_token_types = max([x.size(1) for x in raw_input_types])
        input_types = torch.cat([F.pad(raw_input_type, (0, max_n_token_types - raw_input_type.size(1)), "constant", 0) for raw_input_type in raw_input_types], dim=0)

        n_cumsum_nodes = [0] + np.cumsum([input_id.size(0) for input_id in raw_input_ids]).tolist()
        edge_index = []
        for i, raw_edge_index in enumerate(raw_edge_indexs):
            if raw_edge_index.shape[0] > 0:
                edge_index.append(raw_edge_index + n_cumsum_nodes[i])
        edge_index = torch.cat(edge_index, dim=1) if len(edge_index) > 0 else None

        visit_positions = torch.cat(raw_visit_positions, dim=0)

        if is_train:
            labels = []  # [n_token_type, B, n_tokens], each element is a multi-hop label tensor
            for i in range(n_token_type):
                labels.append(torch.cat([x[i] for x in raw_labels]))
            
            raw_anomaly_labels = [feat[5] for feat in batch]
            if raw_anomaly_labels[0] is not None:
                max_n_anomaly_labels = max([x.size(1) for x in raw_anomaly_labels])
                anomaly_labels = torch.cat([F.pad(raw_anomaly_label, (0, max_n_anomaly_labels - raw_anomaly_label.size(1)), "constant", 0) for raw_anomaly_label in raw_anomaly_labels], dim=0)
            else:
                anomaly_labels = None
            return input_ids, input_types, edge_index, visit_positions, labels, anomaly_labels
        else:
            labels = torch.stack(raw_labels, dim=0)
            labeled_batch_idx = [n - 1 for n in n_cumsum_nodes[1:]]  # indicate the index of the to-be-predicted admission
            return input_ids, input_types, edge_index, visit_positions, labeled_batch_idx, labels
    
    return batcher_dev

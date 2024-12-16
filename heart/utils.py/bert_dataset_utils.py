import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
from utils.token_utils import EHRTokenizer


class PretrainEHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, token_type=['diag', 'med', 'pro', 'lab']):
        self.tokenizer = tokenizer

        def transform_data(data):
            records, ages = {}, {}
            genders = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                genders[subject_id] = [item_df.head(1)["GENDER"].values[0]]

                patient, age = [], []
                for _, row in item_df.iterrows():
                    admission = []
                    if "diag" in token_type:
                        admission.append(list(row['ICD9_CODE']))
                    if "med" in token_type:
                        admission.append(list(row['NDC']))
                    if "pro" in token_type:
                        admission.append(list(row['PRO_CODE']))
                    if "lab" in token_type:
                        admission.append(list(row['LAB_TEST']))
                    patient.append(admission)
                    age.append(row['AGE'])
                records[subject_id] = list(patient)
                ages[subject_id] = age
            return records, ages, genders

        self.records, self.ages, self.genders = transform_data(data_pd)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        subject_id = list(self.records.keys())[item]

        input_tokens = ['[CLS]']
        visit_positions = [0]
        age_tokens = [self.ages[subject_id][0]]
        for idx, adm in enumerate(self.records[subject_id]):  # each subject have multiple admissions, idx:visit id
            cur_input_tokens = []
            for i in range(len(adm)):  # one admission may have many kinds of entities, [[diag], [med], ...]
                cur_input_tokens.extend(list(adm[i]))
            if idx != len(self.records[subject_id]) - 1:  # add [SEP] token between visits
                cur_input_tokens.append('[SEP]')
            input_tokens.extend(cur_input_tokens)
            visit_positions.extend([idx] * len(cur_input_tokens))
            age_tokens.extend([self.ages[subject_id][idx]] * len(cur_input_tokens))

        visit_positions = torch.tensor(visit_positions)
        age_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(age_tokens))

        # masked token prediction
        non_special_tokens_idx = [idx for idx, x in enumerate(input_tokens) if x != '[CLS]' and x != '[SEP]']
        masked_tokens_idx = np.random.choice(non_special_tokens_idx, max(1, int(len(non_special_tokens_idx) * 0.15)))
        masked_tokens = [input_tokens[idx] for idx in masked_tokens_idx]
        masked_input_tokens = input_tokens.copy()
        for idx in masked_tokens_idx:
            if np.random.random() < 0.8:
                masked_input_tokens[idx] = '[MASK]'
            elif np.random.random() < 0.5:
                # TODO: sample based on the entity type
                masked_input_tokens[idx] = np.random.choice(list(self.tokenizer.diag_voc.word2idx.keys()))
        masked_input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(masked_input_tokens))
        masked_tokens_idx = torch.tensor(masked_tokens_idx)
        # TODO: consider all kinds of entities
        masked_lm_labels = torch.tensor(self.tokenizer.convert_tokens_to_ids(masked_tokens, voc_type='diag'))
        return masked_input_ids, visit_positions, age_tokens, masked_lm_labels, masked_tokens_idx


class FinetuneEHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, token_type=['diag', 'med', 'pro', 'lab'], task='death'):
        self.tokenizer = tokenizer
        self.task = task

        def transform_data(data, task):
            age_records = {}
            hadm_records = {}  # including current admission and previous admissions
            genders = {}
            labels = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                patient, ages = [], []
                
                for _, row in item_df.iterrows():
                    admission = []
                    hadm_id = row['HADM_ID']
                    if "diag" in token_type:
                        admission.append(list(row['ICD9_CODE']))
                    if "med" in token_type:
                        admission.append(list(row['NDC']))
                    if "pro" in token_type:
                        admission.append(list(row['PRO_CODE']))
                    if "lab" in token_type:
                        admission.append(list(row['LAB_TEST']))
                    patient.append(admission)
                    ages.append(row['AGE'])
                    if task in ["death", "stay", "readmission"]:  # binary prediction
                        hadm_records[hadm_id] = list(patient)
                        age_records[hadm_id] = ages
                        genders[hadm_id] = [item_df.head(1)["GENDER"].values[0]]
                        if "READMISSION" in row:
                            labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"], row["READMISSION"]]
                        else:
                            labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"]]
                    else:  # next diagnosis prediction
                        label = row["NEXT_DIAG_6M"] if task == "next_diag_6m" else row["NEXT_DIAG_12M"]
                        if str(label) != "nan":  # only include the admission with next diagnosis
                            hadm_records[hadm_id] = list(patient)
                            age_records[hadm_id] = ages
                            genders[hadm_id] = [item_df.head(1)["GENDER"].values[0]]
                            labels[hadm_id] = list(label)

            return hadm_records, age_records, genders, labels
        self.records, self.ages, self.genders, self.labels = transform_data(data_pd, task)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        hadm_id = list(self.records.keys())[item]

        input_tokens = ['[CLS]']
        visit_positions = [0]
        age_tokens = [self.ages[hadm_id][0]]
        for idx, adm in enumerate(self.records[hadm_id]):  # each subject have multiple admissions, idx:visit id
            cur_input_tokens = []
            for i in range(len(adm)):  # one admission may have many kinds of entities, [[diag], [med], ...]
                cur_input_tokens.extend(list(adm[i]))
            if idx != len(self.records[hadm_id]) - 1:  # add [SEP] token between visits
                cur_input_tokens.append('[SEP]')
            input_tokens.extend(cur_input_tokens)
            visit_positions.extend([idx] * len(cur_input_tokens))
            age_tokens.extend([self.ages[hadm_id][idx]] * len(cur_input_tokens))

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
            input_tokens.extend(['[SEP]', '[MASK]'])
            visit_positions.extend([visit_positions[-1]+1, visit_positions[-1]+1])
            age_tokens.extend([age_tokens[-1], age_tokens[-1]])
            label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.labels[hadm_id], voc_type='diag'))
            labels = torch.zeros(self.tokenizer.token_number(voc_type='diag')).long()
            labels[label_ids] = 1  # multi-hop vector

        visit_positions = torch.tensor(visit_positions)
        age_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(age_tokens))
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(input_tokens))

        return input_ids, visit_positions, age_tokens, labels


def batcher(tokenizer):
    def batcher_dev(batch):
        input_ids, visit_positions, age_tokens = [feat[0] for feat in batch], [feat[1] for feat in batch], [feat[2] for feat in batch]
        if len(batch[0]) > 3:
            batch_data = []
            for d in range(3, len(batch[0])):
                data = [feat[d] for feat in batch]
                max_len = max([len(x) for x in data])
                data = [F.pad(x, (0, max_len - len(x)), "constant", 0) for x in data]
                data = torch.stack(data)
                batch_data.append(data)

        # padding
        pad_id = tokenizer.vocab.word2idx['[PAD]']
        max_len = max([len(x) for x in input_ids])
        input_ids = [F.pad(x, (0, max_len - len(x)), "constant", pad_id) for x in input_ids]
        visit_positions = [F.pad(x, (0, max_len - len(x)), "constant", pad_id) for x in visit_positions]
        age_tokens = [F.pad(x, (0, max_len - len(x)), "constant", pad_id) for x in age_tokens]
        
        input_ids = torch.stack(input_ids)
        visit_positions = torch.stack(visit_positions)
        age_tokens = torch.stack(age_tokens)

        if len(batch[0]) > 3:
            return input_ids, age_tokens, visit_positions, *batch_data
        else:
            return input_ids, age_tokens, visit_positions
    
    return batcher_dev

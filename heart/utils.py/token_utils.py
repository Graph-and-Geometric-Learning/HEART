import torch
import numpy as np
from utils.tree_utils import build_atc_tree, build_icd9_tree


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


class EHRTokenizer(object):
    def __init__(self, diag_sentences, med_sentences, lab_sentences, pro_sentences, gender_set, age_set, age_gender_set=None, special_tokens=("[PAD]", "[CLS]", "[SEP]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)
        self.n_special_tokens = len(special_tokens)
        self.age_voc = self.add_vocab(age_set)
        self.diag_voc = self.add_vocab(diag_sentences)
        self.med_voc = self.add_vocab(med_sentences)
        self.lab_voc = self.add_vocab(lab_sentences)
        if pro_sentences is not None:
            self.pro_voc = self.add_vocab(pro_sentences)
        else:
            self.pro_voc = Voc()
        self.gender_voc = self.add_vocab(gender_set)
        if age_gender_set is not None:
            self.age_gender_voc = self.add_vocab(age_gender_set)
        else:
            self.age_gender_voc = Voc()

        assert len(special_tokens) + len(self.age_voc.idx2word) + len(self.diag_voc.idx2word) + len(self.med_voc.idx2word) + \
                len(self.lab_voc.idx2word) + len(self.pro_voc.idx2word) + len(self.gender_voc.idx2word) + len(self.age_gender_voc.idx2word) == len(self.vocab.idx2word)

    def build_tree(self):
        # create tree for diagnosis and medication
        diag2tree, self.diag_tree_voc = build_icd9_tree(Voc(), list(self.diag_voc.idx2word.values()))
        med2tree, self.med_tree_voc = build_atc_tree(Voc(), list(self.med_voc.idx2word.values()))
        
        diag_tree_table = []
        for diag_id in range(len(self.diag_voc.idx2word)):
            diag_tree = diag2tree[self.diag_voc.idx2word[diag_id]]  # [code1, code2, ...]
            diag_tree_table.append([self.diag_tree_voc.word2idx[code] for code in diag_tree])
        
        med_tree_table = []
        for med_id in range(len(self.med_voc.idx2word)):
            med_tree = med2tree[self.med_voc.idx2word[med_id]]  # [code1, code2, ...]
            med_tree_table.append([self.med_tree_voc.word2idx[code] for code in med_tree])
        
        # [n_diag/med, n_level]
        self.diag_tree_table, self.med_tree_table = torch.tensor(diag_tree_table), torch.tensor(med_tree_table)

    def add_vocab(self, sentences):
        voc = self.vocab
        specific_voc = Voc()
        for sentence in sentences:
            voc.add_sentence(sentence)
            specific_voc.add_sentence(sentence)
        return specific_voc

    def convert_tokens_to_ids(self, tokens, voc_type="all"):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if voc_type == "all":
                ids.append(self.vocab.word2idx[token])
            elif voc_type == "diag":
                ids.append(self.diag_voc.word2idx[token])
            elif voc_type == "med":
                ids.append(self.med_voc.word2idx[token])
            elif voc_type == "lab":
                ids.append(self.lab_voc.word2idx[token])
            elif voc_type == "pro":
                ids.append(self.pro_voc.word2idx[token])
        return ids

    def convert_ids_to_tokens(self, ids, voc_type="all"):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            if voc_type == "all":
                tokens.append(self.vocab.idx2word[i])
            elif voc_type == "diag":
                tokens.append(self.diag_voc.idx2word[i])
            elif voc_type == "med":
                tokens.append(self.med_voc.idx2word[i])
            elif voc_type == "lab":
                tokens.append(self.lab_voc.idx2word[i])
            elif voc_type == "pro":
                tokens.append(self.pro_voc.idx2word[i])
        return tokens
    
    def token_id_range(self, voc_type="diag"):
        age_size = len(self.age_voc.idx2word)
        diag_size = len(self.diag_voc.idx2word)
        med_size = len(self.med_voc.idx2word)
        lab_size = len(self.lab_voc.idx2word)

        if voc_type == "diag":
            return [self.n_special_tokens + age_size, self.n_special_tokens + age_size + diag_size]
        elif voc_type == "med":
            return [self.n_special_tokens + age_size + diag_size, self.n_special_tokens + age_size + diag_size + med_size]
        elif voc_type == "lab":
            return [self.n_special_tokens + age_size + diag_size + med_size, self.n_special_tokens + age_size + diag_size + med_size + lab_size]
        elif voc_type == "pro":
            return [self.n_special_tokens + age_size + diag_size + med_size + lab_size, len(self.vocab.idx2word)]
    
    def token_number(self, voc_type="diag"):
        if voc_type == "diag":
            return len(self.diag_voc.idx2word)
        elif voc_type == "med":
            return len(self.med_voc.idx2word)
        elif voc_type == "lab":
            return len(self.lab_voc.idx2word)
        elif voc_type == "pro":
            return len(self.pro_voc.idx2word)
    
    def random_token(self, voc_type="diag"):
        # randomly sample a token from the vocabulary
        if voc_type == "diag":
            return self.diag_voc.idx2word[np.random.randint(len(self.diag_voc.idx2word))]
        elif voc_type == "med":
            return self.med_voc.idx2word[np.random.randint(len(self.med_voc.idx2word))]
        elif voc_type == "lab":
            return self.lab_voc.idx2word[np.random.randint(len(self.lab_voc.idx2word))]
        elif voc_type == "pro":
            return self.pro_voc.idx2word[np.random.randint(len(self.pro_voc.idx2word))]


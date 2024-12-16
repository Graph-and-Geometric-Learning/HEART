import torch
import torch.nn as nn
import torch.nn.functional as F

from heart.models.gnn import DotAttnConv
from heart.models.transformer import TransformerBlock
from heart.models.transformer_edge import EdgeTransformerBlock, EdgeModule

from time import time


class TreeEmbeddings(nn.Module):
    def __init__(self, config, diag_tree_table, med_tree_table, n_diag_tokens, n_med_tokens, diag_range, med_range):
        super(TreeEmbeddings, self).__init__()
        # tree_table: [n_diag/n_med, n_level]
        self.n_dim = config.hidden_size
        self.diag_range, self.med_range = diag_range, med_range
        self.diag_tree_table, self.med_tree_table = diag_tree_table, med_tree_table

        self.diag_tokens = nn.Embedding(n_diag_tokens, config.hidden_size // diag_tree_table.shape[1])
        self.med_tokens = nn.Embedding(n_med_tokens, config.hidden_size // med_tree_table.shape[1])
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)  # [PAD] token
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_types):
        B, N = input_ids.shape[0], input_ids.shape[1]

        # concat the embedding at each layer
        diag_tree_tokens = self.diag_tokens(self.diag_tree_table.to(input_ids.device)).reshape(-1, self.n_dim)
        med_tree_tokens = self.med_tokens(self.med_tree_table.to(input_ids.device)).reshape(-1, self.n_dim)

        input_ids = input_ids.reshape(-1)
        diag_mask = (input_ids >= self.diag_range[0]) * (input_ids < self.diag_range[1])
        med_mask = (input_ids >= self.med_range[0]) * (input_ids < self.med_range[1])

        words_embeddings = self.word_embeddings(input_ids)
        diag_embeddings = diag_tree_tokens[input_ids[diag_mask] - self.diag_range[0]]
        med_embeddings = med_tree_tokens[input_ids[med_mask] - self.med_range[0]]

        # replace the diagnosis and medication embeddings with tree embeddings
        words_embeddings[diag_mask] = diag_embeddings
        words_embeddings[med_mask] = med_embeddings
        words_embeddings = words_embeddings.reshape(B, N, -1)

        # words_embeddings = words_embeddings + self.type_embedding(token_types)

        return self.emb_dropout(words_embeddings)


class HBERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(HBERTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)  # [PAD] token
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_types):
        # embedding the indexed sequence to sequence of vectors
        words_embeddings = self.word_embeddings(input_ids)
        return self.emb_dropout(words_embeddings)


# Hierarchical Transformer
class HiTransformer(nn.Module):
    def __init__(self, config):
        super(HiTransformer, self).__init__()
        
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        # multi-layers transformer blocks, deep network
        if config.gat == "dotattn":
            self.cross_attentions = nn.ModuleList(
                [DotAttnConv(config.hidden_size, config.hidden_size, config.gnn_n_heads, config.max_visit_size, config.gnn_temp) for _ in range(config.num_hidden_layers)])
        elif config.gat == "None":
            self.cross_attentions = None

    def forward(self, x, edge_index, mask, visit_positions):
        # running over multiple transformer blocks
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x, mask)  # [B, L, D]
            if edge_index is not None and self.cross_attentions is not None:
                x = torch.cat([self.cross_attentions[i](x[:, 0], edge_index, visit_positions).unsqueeze(dim=1), 
                                x[:, 1:]], dim=1)  # communicate between visits
        return x


# Hierarchical Transformer with Edge Representation
class HiEdgeTransformer(nn.Module):
    def __init__(self, config):
        super(HiEdgeTransformer, self).__init__()

        self.edge_module = EdgeModule(config)
        
        self.transformer_blocks = nn.ModuleList(
            [EdgeTransformerBlock(config) for _ in range(config.num_hidden_layers)])
        
        if config.gat == "dotattn":
            self.cross_attentions = nn.ModuleList(
                [DotAttnConv(config.hidden_size, config.hidden_size, config.gnn_n_heads, config.max_visit_size, config.gnn_temp) for _ in range(config.num_hidden_layers)])
        elif config.gat == "None":
            self.cross_attentions = None

    def forward(self, x, x_types, edge_index, mask, visit_positions):
        edge_embs = self.edge_module(x, x_types)  # [B, L, L, D]
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x, edge_embs, mask)  # [B, L, D]
            if edge_index is not None and self.cross_attentions is not None:
                x = torch.cat([self.cross_attentions[i](x[:, 0], edge_index, visit_positions).unsqueeze(dim=1), 
                                x[:, 1:]], dim=1)  # communicate between visits
        return x


class MaskedPredictionHead(nn.Module):
    def __init__(self, config, voc_size):
        super(MaskedPredictionHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size), 
                nn.ReLU(), 
                nn.Linear(config.hidden_size, voc_size)
            )

    def forward(self, input):
        return self.cls(input)


# binary classification task
class BinaryPredictionHead(nn.Module):
    def __init__(self, config):
        super(BinaryPredictionHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size), 
                nn.ReLU(), 
                nn.Linear(config.hidden_size, 1)
            )

    def forward(self, input):
        return self.cls(input)


class HBERT_Pretrain(nn.Module):
    def __init__(self, config, tokenizer):
        super(HBERT_Pretrain, self).__init__()

        if config.diag_med_emb == "simple":
            self.embeddings = HBERTEmbeddings(config)
        elif config.diag_med_emb == "tree":
            diag_tree_table, med_tree_table = tokenizer.diag_tree_table, tokenizer.med_tree_table
            n_diag_tokens, n_med_tokens = len(tokenizer.diag_tree_voc.idx2word), len(tokenizer.med_tree_voc.idx2word)
            diag_range, med_range = tokenizer.token_id_range("diag"), tokenizer.token_id_range("med")
            self.embeddings = TreeEmbeddings(config, diag_tree_table, med_tree_table, 
                                            n_diag_tokens, n_med_tokens, diag_range, med_range)

        # self.loss_fn = torch.nn.BCEWithLogitsLoss
        self.loss_fn = F.binary_cross_entropy_with_logits
        self.pos_weight = torch.tensor(config.pos_weight)
        self.encoder = config.encoder

        if config.encoder == "hi":
            self.transformer = HiTransformer(config)
        elif config.encoder == "hi_edge":
            self.transformer = HiEdgeTransformer(config)
        else:
            raise NotImplementedError

        self.mask_token_id = config.mask_token_id  # {token_type: masked_id}
        predicted_token_type = config.predicted_token_type  # ["diag", "med", "pro", "lab"]
        label_vocab_size = config.label_vocab_size  # {token_type: vocab_size}
        for token_type in predicted_token_type:
            self.add_module(f"{token_type}_cls", MaskedPredictionHead(config, label_vocab_size[token_type]))
        if config.anomaly_rate > 0:
            self.anomaly_loss_weight = config.anomaly_loss_weight
            self.anomaly_detection_head = BinaryPredictionHead(config)

    def forward(self, input_ids, token_types, edge_index, visit_positions, masked_labels, anomaly_labels):
        device = input_ids.device
        pad_mask = (input_ids > 0)
        pair_pad_mask = pad_mask.unsqueeze(1).repeat(1, input_ids.size(1), 1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embeddings(input_ids, token_types)

        if self.encoder == "hi":
            x = self.transformer(x, edge_index, ~pair_pad_mask, visit_positions)
        elif self.encoder == "hi_edge":
            x = self.transformer(x, token_types, edge_index, ~pair_pad_mask, visit_positions)

        ave_loss, loss_dict = 0, {}
        for i, (token_type, mask_id) in enumerate(self.mask_token_id.items()):
            masked_token_emb = x[input_ids == mask_id]
            prediction = self._modules[f"{token_type}_cls"](masked_token_emb)
            loss = self.loss_fn(prediction, masked_labels[i].to(input_ids.device))
            # loss = self.loss_fn(prediction.view(-1), masked_labels[i].view(-1).to(device), pos_weight=self.pos_weight.to(device))
            ave_loss += loss
            loss_dict[token_type] = loss.cpu().item()
        
        if anomaly_labels is not None:
            anomaly_prediction = self.anomaly_detection_head(x)
            # anomaly_loss = self.loss_fn(reduction='none')(anomaly_prediction.view(-1), anomaly_labels.view(-1))
            anomaly_loss = self.loss_fn(anomaly_prediction.view(-1), anomaly_labels.view(-1), reduction='none')
            anomaly_loss = (anomaly_loss * pad_mask.view(-1)).sum() / pad_mask.sum()
            ave_loss += self.anomaly_loss_weight * anomaly_loss
            loss_dict["anomaly"] = anomaly_loss.cpu().item()
        else:
            loss_dict["anomaly"] = 0.

        return ave_loss / len(loss_dict), loss_dict


class HBERT_Finetune(nn.Module):
    def __init__(self, config, tokenizer):
        super(HBERT_Finetune, self).__init__()

        if config.diag_med_emb == "simple":
            self.embeddings = HBERTEmbeddings(config)
        elif config.diag_med_emb == "tree":
            diag_tree_table, med_tree_table = tokenizer.diag_tree_table, tokenizer.med_tree_table
            n_diag_tokens, n_med_tokens = len(tokenizer.diag_tree_voc.idx2word), len(tokenizer.med_tree_voc.idx2word)
            diag_range, med_range = tokenizer.token_id_range("diag"), tokenizer.token_id_range("med")
            self.embeddings = TreeEmbeddings(config, diag_tree_table, med_tree_table, 
                                            n_diag_tokens, n_med_tokens, diag_range, med_range)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.encoder = config.encoder
        self.diag_mask_id = 3  # the idx of [MASK0] token
        self.task = config.task

        if config.encoder == "hi":
            self.transformer = HiTransformer(config)
        elif config.encoder == "hi_edge":
            self.transformer = HiEdgeTransformer(config)
        else:
            raise NotImplementedError

        if config.task in ["death", "stay", "readmission"]:
            self.downstream_cls = BinaryPredictionHead(config)
        else:
            self.downstream_cls = MaskedPredictionHead(config, config.label_vocab_size)

    def load_weight(self, checkpoint_dict):
        param_dict = dict(self.named_parameters())
        for key in checkpoint_dict.keys():
            if key in param_dict:
                param_dict[key].data.copy_(checkpoint_dict[key])
    
    def forward(self, input_ids, token_types, edge_index, visit_positions, labeled_ids):
        pad_mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embeddings(input_ids, token_types)

        if self.encoder == "hi":
            x = self.transformer(x, edge_index, ~pad_mask, visit_positions)
        elif self.encoder == "hi_edge":
            x = self.transformer(x, token_types, edge_index, ~pad_mask, visit_positions)

        if self.task in ["death", "stay", "readmission"]:
            prediction = self.downstream_cls(x[labeled_ids][:, 0])
        else:
            labeled_ids, labeled_x = input_ids[labeled_ids], x[labeled_ids]
            masked_pos_embs = labeled_x[labeled_ids == self.diag_mask_id]
            prediction = self.downstream_cls(masked_pos_embs)
        return prediction


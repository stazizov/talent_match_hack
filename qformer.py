import torch
from torch import nn
from torch.nn import MultiheadAttention
from transformers import AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch.nn.functional as F

from modeling_rope_bert import RoPEBertEncoder, RoPEBertConfig


def max_sim(docs, queries, normalize: bool = False):
    x = torch.einsum('abz,ijz->abij', queries, docs)
    x = x.max(dim=-1).values.sum(dim=1)
    if normalize:
        x = x / queries.shape[1]
    return x


class QFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
        for param in self.base_model.embeddings.parameters():
            param.requires_grad = False
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
        hs = self.base_model.config.hidden_size
        self.trans_proj = nn.Sequential(nn.Linear(hs, hs), nn.GELU(), nn.Linear(hs, 128))
        self.pos_ids = RoPEBertEncoder(
            RoPEBertConfig(
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=1,
                intermediate_size=256
            )
        )
        self.seq_id = nn.Embedding(4, self.base_model.config.hidden_size)
        self.proj_out = nn.Linear(128, 1)

    def forward(self, data, out_logits: bool = False):
        vac_vecs = self.base_model(input_ids=data['vacancy']['input_ids'],
                                   attention_mask=data['vacancy']['attention_mask']).last_hidden_state
        cand_vecs = self.base_model(input_ids=data['candidate']['input_ids'],
                                    attention_mask=data['candidate']['attention_mask']).last_hidden_state
        exp_vecs = [
            self.base_model(input_ids=x['input_ids'], attention_mask=x['attention_mask']).last_hidden_state
            for
            x
            in data['experiences']
        ]
        all_exp_vecs = torch.cat(exp_vecs, dim=1)

        mask = []
        mask.extend([0] * vac_vecs.shape[1])
        mask.extend([1] * cand_vecs.shape[1])
        mask.extend([2] * all_exp_vecs.shape[1])
        mask = torch.tensor(mask, dtype=torch.long, device=vac_vecs.device)
        next_in = self.seq_id(mask).unsqueeze(0) + torch.cat([vac_vecs, cand_vecs, all_exp_vecs], dim=1)
        next_in = self.trans_proj(next_in)
        next_in = self.pos_ids(next_in).last_hidden_state[:, 0]

        out = self.proj_out(next_in).squeeze(1)
        if out_logits:
            return out, next_in
        else:
            return out
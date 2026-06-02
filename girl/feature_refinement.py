from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from girl.components import Conv1D, WeightedPool


class FeatureRefinement(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        sim_w = torch.empty(d_model, 1)
        nn.init.xavier_uniform_(sim_w)
        self.sim_w = nn.Parameter(sim_w, requires_grad=True)

        cor_v_w = torch.empty(1, d_model)
        nn.init.xavier_uniform_(cor_v_w)
        self.cor_v_w = nn.Parameter(cor_v_w, requires_grad=True)

        cor_q_w = torch.empty(1, 1)
        nn.init.xavier_uniform_(cor_q_w)
        self.cor_q_w = nn.Parameter(cor_q_w, requires_grad=True)

        # self.sentence_feature_extractor = GlobalFeatureExtractor(d_model, d_model)
        self.word_to_sentence_pool = WeightedPool(dim=d_model)
        # self.mixer = Conv1D(in_dim=3 * d_model, out_dim=d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.mixer = Conv1D(in_dim=4 * d_model, out_dim=d_model, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, video_features, query_features,
                video_mask: Optional[Tensor] = None, query_mask: Optional[Tensor] = None, ):
        bs, vl, dim = video_features.shape
        _, ql, _ = query_features.shape

        query_expanded = query_features.unsqueeze(2)  # [bs, num_words, 1, hidden_size]
        video_expanded = video_features.unsqueeze(1)  # [bs, 1, num_clips, hidden_size]

        correlation = (query_expanded * video_expanded) # [bs, num_words, num_clips, hidden_size]
        correlation_scores = nn.Softmax(dim=1)(correlation.sum(dim=-1))  # [bs, num_words, num_clips]

        cor_v_w = self.cor_v_w.repeat(bs, 1, 1)
        cor_q_w = self.cor_q_w.repeat(bs, ql, 1)

        cor_w = cor_v_w * cor_q_w

        corr_matrix = self.dropout(torch.matmul(correlation_scores.transpose(1,2), cor_w))

        # word-level -> sentence-level
        sentence_feature = self.word_to_sentence_pool(query_features, query_mask).unsqueeze(1)
        sim = F.cosine_similarity(video_features, sentence_feature, dim=-1) + (video_mask + 1e-45).log()

        sim_features = self.dropout(torch.matmul(self.sim_w.transpose(1, 0).expand(bs, 1, dim)
                                                 .transpose(1, 2), sim.unsqueeze(1))).transpose(1, 2)

        # pooled_query = self.weighted_pool(query_features, query_mask)
        pooled_query = self.dropout(sentence_feature.repeat(1, vl, 1))

        features = torch.cat([self.dropout(video_features), sim_features, pooled_query, corr_matrix], dim=2)
        # features = torch.cat([self.dropout(video_features), sim_features, pooled_query], dim=2)

        # output = self.conv1d(output)
        out_features = self.mixer(features)
        return self.dropout(F.relu(out_features))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

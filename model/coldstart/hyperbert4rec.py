import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------
# Core Transformer Components
# -----------------------------
class Attention(nn.Module):
    """Scaled Dot-Product Attention"""
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class GELU(nn.Module):
    """Gaussian Error Linear Units"""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x.pow(3))))

class PositionwiseFeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = GELU()
    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))

class SublayerConnection(nn.Module):
    """Residual + LayerNorm"""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadedAttention(nn.Module):
    """Multi-head attention"""
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out_linear = nn.Linear(d_model, d_model)
        self.attn = Attention()
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask=None):
        B, _, _ = query.size()
        # project
        query, key, value = [l(x).view(B, -1, self.h, self.d_k).transpose(1,2)
                             for l, x in zip(self.linears, (query, key, value))]
        # attend
        x, attn = self.attn(query, key, value, mask=mask, dropout=self.dropout)
        # concat
        x = x.transpose(1,2).contiguous().view(B, -1, self.h*self.d_k)
        return self.out_linear(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    def forward(self, x):
        B = x.size(0)
        return self.pe.weight.unsqueeze(0).expand(B, -1, -1)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=0)

# -----------------------------
# Target Feature Embedding
# -----------------------------
class TargetFeatureEmbedding(nn.Module):
    """
    Encode 5 item-side features into one vector:
    cat₁ | cat₂ | click | like | comment
    """
    def __init__(self, feature_dims: dict, embed_dim: int,
                 counts: dict[str, torch.Tensor]):
        """
        counts: log-count statistics for all target-domain items
                {'click':   FloatTensor[V],
                 'like':    FloatTensor[V],
                 'comment': FloatTensor[V]}
        """
        super().__init__()

        # categorical features
        self.cat1 = nn.Embedding(feature_dims['category_first'],  embed_dim, padding_idx=0)
        self.cat2 = nn.Embedding(feature_dims['category_second'], embed_dim, padding_idx=0)

        # numeric features
        self.click_mlp   = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.like_mlp    = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.comment_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))

        # mean / std of log1p(count) -- computed once
        with torch.no_grad():
            def stat(t):
                m, s = t.mean(), t.std()
                if s == 0: s = torch.tensor(1.0, device=t.device)
                return m, s
            m_c, s_c = stat(torch.log1p(counts['click']))
            m_l, s_l = stat(torch.log1p(counts['like']))
            m_cm, s_cm = stat(torch.log1p(counts['comment']))

        for name, val in [('m_click', m_c),   ('s_click', s_c),
                          ('m_like',  m_l),   ('s_like',  s_l),
                          ('m_comment', m_cm), ('s_comment', s_cm)]:
            self.register_buffer(name, val)

    @staticmethod
    def _norm(x, mean, std):
        return (torch.log1p(x) - mean) / std

    def forward(self, feats: dict) -> torch.Tensor:
        cat1 = self.cat1(feats['category_first'])
        cat2 = self.cat2(feats['category_second'])

        click   = self.click_mlp(  self._norm(feats['click_count'],   self.m_click,   self.s_click  ).unsqueeze(-1))
        like    = self.like_mlp(   self._norm(feats['like_count'],    self.m_like,    self.s_like   ).unsqueeze(-1))
        comment = self.comment_mlp(self._norm(feats['comment_count'], self.m_comment, self.s_comment).unsqueeze(-1))

        return torch.cat([cat1, cat2, click, like, comment], dim=-1)

# -----------------------------
# BERT Backbone
# -----------------------------
class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_size)
        self.pos   = PositionalEmbedding(max_len, embed_size)
        self.drop  = nn.Dropout(dropout)
    def forward(self, seq):
        return self.drop(self.token(seq) + self.pos(seq))

class TransformerBlock(nn.Module):
    def __init__(self, hidden, heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(heads, hidden, dropout)
        self.ff   = PositionwiseFeedForward(hidden, d_ff, dropout)
        self.s1   = SublayerConnection(hidden, dropout)
        self.s2   = SublayerConnection(hidden, dropout)
    def forward(self, x, mask):
        x = self.s1(x, lambda y: self.attn(y, y, y, mask))
        return self.s2(x, self.ff)

class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden = args.hidden_size
        self.embed  = BERTEmbedding(args.num_embedding+1, self.hidden, args.max_len, args.dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(self.hidden, args.num_heads, 4*self.hidden, args.dropout)
            for _ in range(args.block_num)
        ])
    def forward(self, seq):
        mask = (seq > 0).unsqueeze(1).unsqueeze(2)
        x = self.embed(seq)
        for layer in self.layers:
            x = layer(x, mask)
        return x  # [B, T, H]

# -----------------------------
# Hypernetwork-enhanced Cold-start Model
# -----------------------------
class Hyper_BERT_ColdstartModel(nn.Module):
    def __init__(self, args, item_feat_dict):
        super().__init__()
        # ---------------- base encoder ----------------
        self.bert = BERT(args)
        H = self.bert.hidden
        V = args.num_items + 1

        cf = torch.LongTensor([item_feat_dict['category_first'].get(i, 0) for i in range(V)])
        cs = torch.LongTensor([item_feat_dict['category_second'].get(i, 0) for i in range(V)])
        cc = torch.FloatTensor([item_feat_dict['click_count']  .get(i, 0.) for i in range(V)])
        lc = torch.FloatTensor([item_feat_dict['like_count']   .get(i, 0.) for i in range(V)])
        cm = torch.FloatTensor([item_feat_dict['comment_count'].get(i, 0.) for i in range(V)])

        self.register_buffer('category_first',  cf)
        self.register_buffer('category_second', cs)
        self.register_buffer('click_count',     cc)
        self.register_buffer('like_count',      lc)
        self.register_buffer('comment_count',   cm)

        counts = {
            'click'  : cc,
            'like'   : lc,
            'comment': cm
        }
        feat_dims = {
            'category_first' : args.num_category_first,
            'category_second': args.num_category_second
        }
        self.feat_embed = TargetFeatureEmbedding(
            feature_dims = feat_dims,
            embed_dim    = args.feature_dim,
            counts       = counts
        )

        self.hypernet = nn.Sequential(
            nn.Linear(5 * args.feature_dim, H),
            nn.ReLU(),
            nn.Linear(H, H)
        )

    def forward(self, seq):
        rep = self.bert(seq).mean(dim=1)                     # [B,H]
        feats = {                                             # buffers
            'category_first' : self.category_first,
            'category_second': self.category_second,
            'click_count'    : self.click_count,
            'like_count'     : self.like_count,
            'comment_count'  : self.comment_count
        }
        item_feat_vec = self.feat_embed(feats)               # [V,5F]
        W = self.hypernet(item_feat_vec)                     # [V,H]
        return rep @ W.t()                                   # [B,V]
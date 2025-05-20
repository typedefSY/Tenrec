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
        self.activation = GELU()
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

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
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # project
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # attend
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # concat
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)

# -----------------------------
# Embeddings
# -----------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
    """
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)

# -----------------------------
# BERT Model Components
# -----------------------------
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        max_len = args.max_len
        n_layers = args.block_num
        heads = args.num_heads
        vocab_size = args.num_embedding + 1
        hidden = args.hidden_size
        self.hidden = hidden
        dropout = args.dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, heads, hidden * 4, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x, mask)
            
        return x

# -----------------------------
# Attention Pooling
# -----------------------------
class AttentionPooling(nn.Module):
    """Replace mean pooling with attention-based pooling"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x, mask):
        # x: [batch_size, seq_len, hidden_size]
        attn_scores = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return pooled

# -----------------------------
# Domain Adapter
# -----------------------------
class DomainAdapter(nn.Module):
    """Adapts embeddings from source to target domain"""
    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Residual connection
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))
        
    def forward(self, x):
        adapted = self.adapter(x)
        # Mix original and adapted representations
        return self.alpha * x + (1 - self.alpha) * adapted

# -----------------------------
# Target Feature Embedding
# -----------------------------
class TargetFeatureEmbedding(nn.Module):
    """
    Encode item-side features into one vector
    """
    def __init__(self, feature_dims, embed_dim, counts):
        super().__init__()
        
        # categorical features
        self.cat1 = nn.Embedding(feature_dims['category_first'], embed_dim, padding_idx=0)
        self.cat2 = nn.Embedding(feature_dims['category_second'], embed_dim, padding_idx=0)

        # numeric features
        self.click_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.like_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
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

    def forward(self, feats):
        cat1 = self.cat1(feats['category_first'])
        cat2 = self.cat2(feats['category_second'])

        click = self.click_mlp(self._norm(feats['click_count'], self.m_click, self.s_click).unsqueeze(-1))
        like = self.like_mlp(self._norm(feats['like_count'], self.m_like, self.s_like).unsqueeze(-1))
        comment = self.comment_mlp(self._norm(feats['comment_count'], self.m_comment, self.s_comment).unsqueeze(-1))

        return torch.cat([cat1, cat2, click, like, comment], dim=-1)

# -----------------------------
# Main Model Definition
# -----------------------------
class Hyper_BERT_ColdstartModel(nn.Module):
    def __init__(self, args, item_feat_dict):
        super().__init__()
        # Base encoder
        self.bert = BERT(args)
        H = self.bert.hidden
        V = args.num_items + 1

        # Register item features
        cf = torch.LongTensor([item_feat_dict['category_first'].get(i, 0) for i in range(V)])
        cs = torch.LongTensor([item_feat_dict['category_second'].get(i, 0) for i in range(V)])
        cc = torch.FloatTensor([item_feat_dict['click_count'].get(i, 0.) for i in range(V)])
        lc = torch.FloatTensor([item_feat_dict['like_count'].get(i, 0.) for i in range(V)])
        cm = torch.FloatTensor([item_feat_dict['comment_count'].get(i, 0.) for i in range(V)])

        self.register_buffer('category_first', cf)
        self.register_buffer('category_second', cs)
        self.register_buffer('click_count', cc)
        self.register_buffer('like_count', lc)
        self.register_buffer('comment_count', cm)

        counts = {
            'click': cc,
            'like': lc,
            'comment': cm
        }
        
        feat_dims = {
            'category_first': args.num_category_first,
            'category_second': args.num_category_second
        }
        
        self.feat_embed = TargetFeatureEmbedding(
            feature_dims=feat_dims,
            embed_dim=args.feature_dim,
            counts=counts
        )

        # Improvement 1: Attention Pooling
        self.attention_pool = AttentionPooling(H)
        
        # Improvement 2: Domain Adapter
        self.domain_adapter = DomainAdapter(H, dropout=args.dropout)
        
        # Improvement 3: Improved Hypernetwork
        self.hypernet = nn.Sequential(
            nn.Linear(5 * args.feature_dim, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(H, H),
            nn.LayerNorm(H)
        )
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, seq):
        # Create mask for attention
        mask = (seq > 0)
        
        # Get BERT representations
        bert_output = self.bert(seq)
        
        # Improvement 1: Use attention pooling instead of mean
        rep = self.attention_pool(bert_output, mask)
        
        # Improvement 2: Apply domain adaptation
        rep = self.domain_adapter(rep)  # [batch_size, H]
        
        # Get item features
        feats = {
            'category_first': self.category_first,
            'category_second': self.category_second,
            'click_count': self.click_count,
            'like_count': self.like_count,
            'comment_count': self.comment_count
        }
        
        # Get feature embeddings
        item_feat_vec = self.feat_embed(feats)
        
        # Improvement 3: Use improved hypernetwork
        W = self.hypernet(item_feat_vec)   # [V, H]
        
        # Return scores with temperature scaling
        return (rep @ W.t()) / self.temperature # [batch_size, V], V is num of items in target domain + 1 padding

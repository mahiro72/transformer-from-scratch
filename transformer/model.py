import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257, #語彙のサイズ
    "context_length": 1024, #コンテキストの長さ
    "emb_dim": 768, #埋め込み次元
    "n_heads": 12, #ヘッドの数
    "n_layers": 12, #レイヤー数(transformerのブロック数)
    "drop_rate": 0.1, #ドロップアウト率
    "qkv_bias": False, #クエリ、キー、値の計算にバイアスを使うかどうか
}

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # 標準化時にゼロ除算を防ぐために分散に加算される小さな定数
        self.scale = nn.Parameter(torch.ones(emb_dim)) # 標準化されたデータを拡大/縮小
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 上記を平行移動

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0, "d_out must be divisible by num_heads")

        self.d_out = d_out # 全ヘッドの合計次元
        self.num_heads = num_heads # ヘッドの数
        self.head_dim = d_out // num_heads # 各ヘッドの次元

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # xは[バッチサイズ2, トークン数4, 埋め込み次元768]のテンソル
        b, num_tokens, d_in = x.shape

        # 入力xに重みをかけて、それぞれのkeys, queries, valuesを計算する
        # この時点では、それぞれ複数ヘッドのベクトルがまとめて計算されている
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # ここで[バッチサイズ2, トークン数4, ヘッド数12, ヘッド次元64]のテンソルに変換する。
        # これによりヘッドを独立させて注意計算を行う。独立させなかった場合、注意スコア算出時に全てのヘッド分(ヘッド数*各ヘッドの次元)のスコアを合算してしまうため、分割が必要。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 変換前: [バッチサイズ2, トークン数4, ヘッド数12, ヘッド次元64]
        # 変換後: [バッチサイズ2, ヘッド数12, トークン数4, ヘッド次元64] -> トークン数とヘッド数を入れ替える
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # keys: 変換後は、[バッチサイズ2, ヘッド数12, ヘッド次元64, トークン数4] ここでは注意スコアを計算している
        # アテンションスコア計算: 各64次元のクエリとキーで内積を計算
        attn_scores = queries @ keys.transpose(2, 3) # 内積の計算
        # 結果: (バッチサイズ2, ヘッド数12, クエリトークン数4, キートークン数4)
        # それぞれのトークンに対して、ヘッド数分のqueryとkeyの内積を計算している
        # (head1: k1*q1, k2*q2, ... head2: k1*q1, k2*q2, ...)

        # 未来のトークンを見えなくするマスク適用
        # mask: 上三角行列で未来の情報を-infでマスク
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # スケーリング（√64(ヘッド次元) = 8で割る）とソフトマックスで確率分布に変換
        # アテンション重み: (バッチサイズ2, ヘッド数12, クエリトークン数4, キートークン数4)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # アテンション重みとvaluesの行列乗算、その後元の次元順序に戻す
        # (バッチサイズ2, ヘッド数12, クエリトークン数4, キートークン数4) @ (バッチサイズ2, ヘッド数12, トークン数4, ヘッド次元64)
        # = (バッチサイズ2, ヘッド数12, トークン数4, ヘッド次元64) → transpose → (バッチサイズ2, トークン数4, ヘッド数12, ヘッド次元64)
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        # 12個のヘッドを連結して元の768次元に戻す
        # (バッチサイズ2, トークン数4, ヘッド数12, ヘッド次元64) から view で (バッチサイズ2, トークン数4, 埋め込み次元768)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 最終的な線形変換
        # (バッチサイズ2, トークン数4, 埋め込み次元768)
        context_vec = self.out_proj(context_vec)
        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # トークン埋め込み層
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # 位置埋め込み層
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits

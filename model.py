import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SimpleMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        emb = emb.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embed_dim)
        out = self.transformer(emb)  # same shape
        out = out.permute(1, 0, 2)  # back to (batch_size, seq_len, embed_dim)
        logits = self.fc(out)  # (batch_size, seq_len, vocab_size)
        return logits

def mask_tokens(inputs, mask_token_id, vocab_size, mlm_probability=0.15):
    """
    入力の一部をマスクする関数
    inputs: Tensor (batch_size, seq_len)
    """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # マスクしてない部分は損失計算しないように-100に

    # マスクトークンに置き換える確率80%
    indices_replaced = (torch.bernoulli(torch.full(labels.shape, 0.8)).bool()) & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10%はランダム単語に置き換え
    indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool()) & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 残り10%は元の単語のまま
    return inputs, labels

# 使い方例
if __name__ == "__main__":
    vocab_size = 1000
    mask_token_id = vocab_size - 1  # 最後のIDをマスクトークンに設定とか
    batch_size = 2
    seq_len = 10

    model = SimpleMLM(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ダミーデータ（単語IDのランダム列）
    inputs = torch.randint(0, vocab_size-1, (batch_size, seq_len))
    inputs, labels = mask_tokens(inputs, mask_token_id, vocab_size)

    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)  # (batch_size, seq_len, vocab_size)

    loss = F.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())

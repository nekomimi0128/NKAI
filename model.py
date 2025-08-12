import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
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

    # 80%マスクトークンに置き換え
    indices_replaced = (torch.bernoulli(torch.full(labels.shape, 0.8)).bool()) & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10%ランダム単語に置き換え
    indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool()) & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 残り10%は元の単語のまま
    return inputs, labels


def generate_response(model, input_ids, max_len=20, eos_token_id=None, device='cpu'):
    """
    シンプルな文章生成（次のトークンを貪欲に選択）
    input_ids: Tensor (batch_size, seq_len)
    max_len: 生成する最大長
    eos_token_id: 終了トークンID（あれば）
    """
    model.eval()
    generated = input_ids.to(device)
    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(generated)  # (batch_size, seq_len, vocab_size)
            next_token_logits = outputs[:, -1, :]  # 最後のトークンの予測
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # 貪欲に選択
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
    return generated


if __name__ == "__main__":
    # 設定
    vocab_size = 1000
    mask_token_id = vocab_size - 1  # マスクトークンID
    eos_token_id = 999  # 終了トークンID（適宜設定）
    batch_size = 2
    seq_len = 10
    device = 'cpu'

    model = SimpleMLM(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ダミーデータ（単語IDのランダム列）
    inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len)).to(device)
    inputs, labels = mask_tokens(inputs, mask_token_id, vocab_size)

    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)  # (batch_size, seq_len, vocab_size)

    loss = F.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())

    # 生成テスト
    test_input = torch.randint(0, vocab_size - 1, (1, 5)).to(device)  # 1サンプル、長さ5の入力
    generated_seq = generate_response(model, test_input, max_len=10, eos_token_id=eos_token_id, device=device)
    print("Generated sequence:", generated_seq)

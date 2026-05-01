import torch
import torch.nn as nn


class PoetryLSTM(nn.Module):
    """
    字符级古诗生成模型
    输入: [batch_size, seq_len]
    输出: [batch_size, seq_len, vocab_size]
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ):
        super(PoetryLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: [B, T]
        hidden: (h0, c0)
        """
        emb = self.embedding(x)              # [B, T, embed_dim]
        out, hidden = self.lstm(emb, hidden) # [B, T, hidden_dim]
        logits = self.fc(out)                # [B, T, vocab_size]
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 2823
    model = PoetryLSTM(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ).to(device)

    print(model)

    x = torch.randint(0, vocab_size, (4, 27)).to(device)
    logits, hidden = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
    print("Hidden h shape:", hidden[0].shape)
    print("Hidden c shape:", hidden[1].shape)
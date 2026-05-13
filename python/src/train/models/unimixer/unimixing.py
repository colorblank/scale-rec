import torch, torch.nn as nn


def sinkhorn_knopp(mat, n_iters=3):
    m = mat.exp()
    eps = 1e-8
    for _ in range(n_iters):
        m = m / (m.sum(dim=m.ndim - 1, keepdim=True) + eps)
        m = m / (m.sum(dim=m.ndim - 2, keepdim=True) + eps)
    return m


class UniMixing(nn.Module):
    def __init__(self, embed_dim, block_size):
        super().__init__()
        assert embed_dim % block_size == 0
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_blocks = embed_dim // block_size
        n, b = self.num_blocks, block_size
        self.global_weights_logits = nn.Parameter(torch.randn(n, n))
        self.local_weights_logits = nn.Parameter(torch.randn(n, b, b))

    def forward(self, x, temperature):
        bs = x.shape[0]
        n, b = self.num_blocks, self.block_size
        x_blocks = x.view(bs, n, b)
        w_b = sinkhorn_knopp(self.local_weights_logits / temperature)
        w_b = 0.5 * (w_b + w_b.transpose(1, 2))
        h = torch.matmul(x_blocks.unsqueeze(2), w_b.unsqueeze(0)).squeeze(2)
        w_g = sinkhorn_knopp(self.global_weights_logits / temperature)
        w_g = 0.5 * (w_g + w_g.transpose(0, 1))
        out = torch.matmul(w_g.unsqueeze(0).expand(bs, n, n), h)
        return out.reshape(bs, self.embed_dim)


class UniMixingLite(nn.Module):
    def __init__(self, embed_dim, block_size, num_basis, rank):
        super().__init__()
        assert embed_dim % block_size == 0
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_blocks = embed_dim // block_size
        n, b = self.num_blocks, block_size
        self.a_g = nn.Parameter(torch.randn(n, rank))
        self.b_g = nn.Parameter(torch.randn(rank, n))
        self.z = nn.Parameter(torch.randn(num_basis, b, b))
        self.omega = nn.Parameter(torch.randn(n, num_basis))

    def forward(self, x, temperature):
        bs = x.shape[0]
        n, b, l = self.num_blocks, self.block_size, self.embed_dim
        x_blocks = x.view(bs, n, b)
        w_b = (self.omega.unsqueeze(2).unsqueeze(3) * self.z.unsqueeze(0)).sum(dim=1)
        w_b = 0.5 * (w_b + w_b.transpose(1, 2))
        w_b = sinkhorn_knopp(w_b / temperature)
        h = torch.matmul(x_blocks.unsqueeze(2), w_b.unsqueeze(0)).squeeze(2)
        w_g = torch.matmul(self.a_g, self.b_g)
        w_g = 0.5 * (w_g + w_g.transpose(0, 1))
        w_g = sinkhorn_knopp(w_g / temperature)
        out = torch.matmul(w_g.unsqueeze(0).expand(bs, n, n), h)
        return out.reshape(bs, l)

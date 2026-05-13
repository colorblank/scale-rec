import torch, torch.nn as nn
class FeatureTokenizer(nn.Module):
    def __init__(self, features, token_dim, num_tokens):
        super().__init__(); self.ordered_names=[]; self.feature_to_idx={}; total=0
        for i,(name,vocab_size,embed_dim) in enumerate(features):
            self.feature_to_idx[name]=i; self.ordered_names.append(name)
            setattr(self,f"emb_{name}",nn.Embedding(vocab_size,embed_dim)); total+=embed_dim
        if total % num_tokens != 0: raise ValueError(f"total_embed_dim ({total}) not divisible by num_tokens ({num_tokens})")
        self.token_dim=token_dim; self.num_tokens=num_tokens; self.total_embed_dim=total
        self.token_projections=nn.Conv1d(total,num_tokens*token_dim,kernel_size=1,groups=num_tokens)
    def forward(self, x_inputs):
        embeds=[getattr(self,f"emb_{n}")(x_inputs[n]) for n in self.ordered_names]
        concat=torch.cat(embeds,dim=1); conv_in=concat.unsqueeze(2)
        conv_out=self.token_projections(conv_in); squeezed=conv_out.squeeze(2)
        return squeezed.view(squeezed.shape[0],self.num_tokens,self.token_dim)

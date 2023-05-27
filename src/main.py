from math import sqrt

import torch
from torch import bmm, concat, ones, softmax, transpose, triu
from torch.nn import Embedding, Linear, Module, ModuleDict, ModuleList, ReLU, Sequential


class Attention(Module):
    def __init__(self, d_model, d_q, d_k, d_v):
        super().__init__()

        self.d_k = d_k
        self.weights = ModuleDict({
            'query_weights': Linear(d_model, d_q, bias=False),
            'key_weights': Linear(d_model, d_k, bias=False),
            'value_weights': Linear(d_model, d_v, bias=False)
        })    

    def forward(self, x): # batch_size x sequence_length x d_k
        queries = self.weights['query_weights'](x) # batch_size x sequence_length x d_q
        keys = self.weights['key_weights'](x)      # batch_size x sequence_length x d_k
        values = self.weights['value_weights'](x)  # batch_size x sequence_length x d_v

        # transpose the last two dimensions so that the inner dimension is correct
        keys_transpose = transpose(keys, -2, -1) # batch_size x d_k x sequence_length

        # batch matrix multiplication to perform Q*K
        similarities = bmm(queries, keys_transpose) # batch_size x sequence_length x sequence_length

        # normalize similarity score by the sqrt of the dimensionality of the keys
        normalized_similarities = similarities / sqrt(self.d_k) # batch_size x sequence_length x sequence_length

        # replace the upper trianglar portion of the matrix with -inf to make sure that past tokens don't attend to future tokens
        mask = triu(ones(normalized_similarities.size()), diagonal=1).bool() # batch_size x sequence_length x sequence_length
        causal_similarities = normalized_similarities.masked_fill(mask, float("-inf")) # batch_size x sequence_length x sequence_length

        # perform softmax to obtain probability that each key attends to a query
        probabilities = softmax(causal_similarities, dim=-1) # batch_size x sequence_length x sequence_length

        # weight the values with the probabilities
        weighted_values = bmm(probabilities, values)

        return weighted_values


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        d_proj = d_model // n_heads
        self.heads = ModuleList([Attention(d_model=d_model, d_q=d_proj, d_k=d_proj, d_v=d_proj) for _ in range(n_heads)])

    def forward(self, x): # batch_size x sequence_length x d_model
        # run embeddings through all heads
        results = [head(x) for head in self.heads] # list of batch_size x sequence_length x d_proj
        # stack the attentions
        return concat(results, dim=-1) # batch_size x sequence_length x d_model

class MLP(Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.model = Sequential(
            Linear(in_features, hidden_features, bias=False), # fan out
            ReLU(),
            Linear(hidden_features, out_features, bias=False), # fan in
            ReLU(),
        )
    
    def forward(self, x): # batch_size x sequence_length x d_model
        return self.model(x) # batch_size x sequence_length x d_model
    
class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, hidden_features):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.mlp = MLP(in_features=d_model, hidden_features=hidden_features, out_features=d_model)
    
    def forward(self, x):
        a = self.multi_head_attention(x) # batch_size x sequence_length x d_model
        # perform key-value lookup for facts (fan out -> fan in)
        o = self.mlp(a) # batch_size x sequence_length x d_model
        # add residual connection
        return o + x # batch_size x sequence_length x d_model

class Transformer(Module):
    def __init__(self, d_model, vocab_size, sequence_length, n_heads, hidden_features, n_layers):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model)
        self.positional_embedding = Embedding(sequence_length, d_model)

        self.blocks = Sequential(*[TransformerBlock(d_model=d_model, n_heads=n_heads, hidden_features=hidden_features) for _ in range(n_layers)])

        # transforms from d_model to vocab_size
        self.vocab_transform = Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens): # batch_size x sequence_length
        embeddings = self.embedding(tokens) # batch_size x sequence_length x d_model

        # get position numbers for each token and make a copy for each batch
        positions = torch.arange(tokens.shape[1]).unsqueeze(0).repeat(tokens.shape[0], 1) # batch_sized x sequence_length
        positional_embeddings = self.positional_embedding(positions) # batch_size x sequence_length x d_model

        combined = embeddings + positional_embeddings # batch_size x sequence_length x d_model

        output = self.blocks(combined) # batch_size x sequence_length x d_model

        # get probability across all tokens
        vocab_probs = softmax(self.vocab_transform(output), dim=-1) # batch_size x sequence_length x vocab_size

        return vocab_probs # batch_size x sequence_length x vocab_size
    
if __name__ == "__main__":
    d_model = 10
    n_heads = 2 # n_heads has to divide d_model evenly!
    batch_size = 1
    sequence_length = 5
    vocab_size = 15
    n_layers = 2

    transformer = Transformer(d_model=d_model, vocab_size=vocab_size, sequence_length=sequence_length, n_heads=n_heads, hidden_features=d_model*2, n_layers=n_layers)

    print(transformer)

    tokens = torch.randint(0, vocab_size-1, (batch_size, sequence_length)) # batch_size x sequence_length

    output = transformer(tokens) # batch_size x sequence_length x d_model

    print(output, output.shape)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    sns.heatmap(output[0].detach().numpy(), cmap="YlGnBu")
    plt.show()
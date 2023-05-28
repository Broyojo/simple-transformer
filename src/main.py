from math import sqrt

import matplotlib.pyplot as plt
import torch
from torch import bmm, concat, ones, softmax, transpose, triu
from torch.nn import (
    CrossEntropyLoss,
    Embedding,
    Linear,
    Module,
    ModuleDict,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim import AdamW
from tqdm import tqdm


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

        # get logits across all possible tokens
        vocab_logits = self.vocab_transform(output) # batch_size x sequence_length x vocab_size

        return vocab_logits # batch_size x sequence_length x vocab_size

# simple character-level tokenizer
class Tokenizer:
    def __init__(self, dataset):
        self.vocab = sorted(list(set(dataset)))
        self.ctoi = {c: i for i, c in enumerate(self.vocab)}
        self.itoc = {i: c for c, i in self.ctoi.items()}

    def __call__(self, text):
        return [self.ctoi[c] for c in text]

    def decode(self, token_ids):
        return "".join([self.itoc[id] for id in token_ids])

if __name__ == "__main__":
    with open("shakespeare.txt", "r") as f:
        dataset = f.read()
    tokenizer = Tokenizer(dataset=dataset)

    # special token for padding
    tokenizer.vocab.append("~")
    tokenizer.ctoi["~"] = len(tokenizer.vocab)-1
    tokenizer.itoc[len(tokenizer.vocab)-1] = "~"

    # print(tokenizer.vocab)
    # print(tokenizer.ctoi)
    # print(tokenizer.itoc)

    # Set device that should be used
    torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    sequence_length = 512

    model = Transformer(
        d_model=384,
        vocab_size=len(tokenizer.vocab),
        sequence_length=sequence_length,
        n_heads=8,
        hidden_features=384*2, # d_model*2 for now
        n_layers=8,
    )

    # Parameters
    epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Define loss function and optimizer
    criterion = CrossEntropyLoss(ignore_index=tokenizer.ctoi["~"])
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Prepare data
    data = tokenizer(dataset)
    data = torch.tensor(data, dtype=torch.long)

    # Split data into batches
    num_batches = len(data) // (batch_size * sequence_length)
    data = data[:num_batches * batch_size * sequence_length]
    data = data.view(batch_size, -1)

    losses = []

    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")

        epoch_losses = []

        # Prepare input and target
        for i in tqdm(range(0, data.size(1)-sequence_length, sequence_length)):
            inputs = data[:, i:i+sequence_length]
            targets = data[:, i+1:i+sequence_length+1]

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.view(-1, len(tokenizer.vocab)), targets.reshape(-1))

            if i % 100 == 0:
                print(loss.item())
                epoch_losses.append(loss.item())
                losses.append(loss.item())

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("average loss:", sum(epoch_losses) / len(epoch_losses))

    plt.plot(losses)
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'transformer_model_batched.pt')

    # Use the model for text generation
    def generate_text(model, tokenizer, start_text, num_chars = 100, top_k = 5):
        model = model.eval()

        start_tokens = torch.tensor(tokenizer(start_text), dtype=torch.long).unsqueeze(0)

        generated_text = start_text

        with torch.no_grad():
            for _ in range(num_chars):
                outputs = model(start_tokens)

                probabilities = torch.nn.functional.softmax(outputs, dim=-1)

                top_probs, top_indices = torch.topk(probabilities[0, -1, :], top_k)

                sampled_index = torch.multinomial(top_probs, 1).item()

                generated_text += tokenizer.itoc[top_indices[sampled_index].item()]  # Convert tensor to integer

                start_tokens = torch.cat((start_tokens, top_indices[sampled_index].unsqueeze(0).unsqueeze(0)), dim=1)

        return generated_text

    output = generate_text(model=model, tokenizer=tokenizer, start_text="To be or not to be:", num_chars=400)

    print(output)

    # print(model)

    # input_sequences = []

    # for i in range(0, len(dataset)-512, 512):
    #     chunk = dataset[i : i + 512]
    #     token_ids = tokenizer(chunk)
    #     input_sequences.append(token_ids)

    # num_epochs = 1

    # loss_fn = CrossEntropyLoss()
    # optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    # for e in range(num_epochs):
    #     for size in tqdm(range(1, 511)):
    #         input_batch = torch.LongTensor()
    #         output_batch = torch.LongTensor()
    #         for sequence in input_sequences:
    #             X = torch.as_tensor(sequence[:size])
    #             input_batch = input
    #             token_to_predict = sequence[size]
    #             y = torch.zeros(1, len(tokenizer.vocab))
    #             y[0][token_to_predict] = 1.0
    #             output_batch.append(y)

    #         input_batch = torch.as_tensor(input_batch)
    #         output_batch = torch.as_tensor(output_batch)

    #         print(input_batch.shape, output_batch.shape)

    #         optimizer.zero_grad()

    #         logits = model(input_batch)
    #         loss = loss_fn(logits, output_batch)
    #         loss.backward()

    #         optimizer.step()





            # total_loss = 0
            # # create a batch from one sequence of all lengths
            # for i in range(2, len(sequence)):
            #     X = torch.as_tensor([sequence[:i]])
                
            #     token_to_predict = sequence[i]
            #     y = torch.zeros(1, len(tokenizer.vocab))
            #     y = y.type(torch.LongTensor)
            #     y[0][token_to_predict] = 1.0 # make the next token have 100% probability

            #     # print(X.shape, y.shape)

            #     optimizer.zero_grad()

            #     logits = model(X)
            #     loss = loss_fn(input=logits, target=y)
            #     loss.backward()

            #     #print(loss)

            #     optimizer.step()

            # print(total_loss / 511)

    # torch.save(model.state_dict(), "model.pt")

    # prompt = "From fairest"

    # while True:
    #     token_ids = torch.as_tensor([tokenizer(prompt)])

    #     with torch.no_grad():
    #         logits = model(token_ids)[0][-1]

    #     probs = softmax(logits, -1)

    #     top_k_tokens = torch.topk(probs, 1)

    #     sample_index = torch.multinomial(top_k_tokens.values, num_samples=1)

    #     sample_token_id = top_k_tokens.indices[sample_index].item()

    #     sample_token = tokenizer.itoc[sample_token_id]

    #     print(sample_token, end="", flush=False)

    #     prompt += sample_token

    # d_model = 10
    # n_heads = 2 # n_heads has to divide d_model evenly!
    # batch_size = 1
    # sequence_length = 5
    # vocab_size = 15
    # n_layers = 2

    # transformer = Transformer(d_model=d_model, vocab_size=vocab_size, sequence_length=sequence_length, n_heads=n_heads, hidden_features=d_model*2, n_layers=n_layers)

    # print(transformer)

    # # create a random sequence of tokens for testing
    # tokens = torch.randint(0, vocab_size-1, (batch_size, sequence_length)) # batch_size x sequence_length

    # output = transformer(tokens) # batch_size x sequence_length x d_model

    # print(output, output.shape)

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # plt.figure(figsize=(10, 10))
    # sns.heatmap(probs.detach().numpy(), cmap="YlGnBu")
    # plt.show()
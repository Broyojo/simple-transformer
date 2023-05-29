import torch
from torch import Tensor

from main import Tokenizer, Transformer


def generate_text(model, tokenizer, start_text, max_seq_len=512, num_chars = 100, top_k = 5, print_output=True):
    model = model.eval()

    if print_output:
        print(start_text, end="", flush=True)

    start_tokens = torch.tensor(tokenizer(start_text), dtype=torch.long).unsqueeze(0)

    generated_text = start_text

    with torch.no_grad():
        for _ in range(num_chars):
            outputs = model(start_tokens)

            probabilities = torch.nn.functional.softmax(outputs, dim=-1)

            top_probs, top_indices = torch.topk(probabilities[0, -1, :], top_k)

            sampled_index = torch.multinomial(top_probs, 1).item()

            token = tokenizer.itoc[top_indices[sampled_index].item()] # Convert tensor to integer

            if print_output:
                print(token, end="", flush=True)

            generated_text += token

            # Create a new token tensor that includes the newly sampled token
            new_token = torch.tensor([top_indices[sampled_index].item()], dtype=torch.long).unsqueeze(0)
            start_tokens = torch.cat((start_tokens, new_token), dim=1)

            # If the sequence is too long, keep only the last 'max_seq_len' tokens
            if start_tokens.shape[1] > max_seq_len:
                start_tokens = start_tokens[:, -max_seq_len:]

    return generated_text

if __name__ == "__main__":
    # Load tokenizer
    with open("shakespeare.txt", "r") as f:
        dataset = f.read()
    tokenizer = Tokenizer(dataset=dataset)
    tokenizer.vocab.append("~")
    tokenizer.ctoi["~"] = len(tokenizer.vocab)-1
    tokenizer.itoc[len(tokenizer.vocab)-1] = "~"

    # Load model
    sequence_length = 512

    model = Transformer(
        d_model=384,
        vocab_size=len(tokenizer.vocab),
        sequence_length=sequence_length,
        n_heads=8,
        hidden_features=384*2, # d_model*2 for now
        n_layers=8,
    )

    model.load_state_dict(torch.load('transformer_model.pt'))

    # Generate text
    start_text = "To be or not to be:"
    output = generate_text(model, tokenizer, start_text, num_chars=1024, top_k=5)

    print(output)
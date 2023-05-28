import torch
from torch import Tensor

from main import Tokenizer, Transformer


def generate_text(model: Transformer, tokenizer: Tokenizer, start_text: str, num_chars: int = 100, top_k: int = 5, device: str = 'cpu') -> str:
    model = model.eval().to(device)

    start_tokens = torch.tensor(tokenizer(start_text), dtype=torch.long).unsqueeze(0).to(device)

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

if __name__ == "__main__":
    # Load tokenizer
    with open("shakespeare.txt", "r") as f:
        dataset = f.read()
    tokenizer = Tokenizer(dataset=dataset)
    tokenizer.vocab.append("~")
    tokenizer.ctoi["~"] = len(tokenizer.vocab)-1
    tokenizer.itoc[len(tokenizer.vocab)-1] = "~"

    # Load model
    model = Transformer(
        d_model=16,
        vocab_size=len(tokenizer.vocab),
        sequence_length=512,
        n_heads=2,
        hidden_features=32,  # d_model*2 for now
        n_layers=4,
    )
    model.load_state_dict(torch.load('transformer_model_batched.pt'))

    # Generate text
    start_text = "the the the "
    print(generate_text(model, tokenizer, start_text, num_chars=100, top_k=1))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def fractal_attention(query, key, value, fractal_params, attention_mask=None):
    """
    Calculate attention scores using fractal properties.

    Args:
    query (torch.Tensor): The query tensor.
    key (torch.Tensor): The key tensor.
    value (torch.Tensor): The value tensor.
    fractal_params (torch.Tensor): The fractal parameters tensor.
    attention_mask (torch.Tensor, optional): A mask to ignore certain positions in the input.

    Returns:
    torch.Tensor: The attention output.
    torch.Tensor: The attention weights.
    """
    
    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(query.size(-1))

    # Adjust scores with fractal parameters
    fractal_params = fractal_params.unsqueeze(-1).expand_as(scores)
    scores += fractal_params

    if attention_mask is not None:
        # Expand attention_mask to match the shape of scores
        attention_mask = attention_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention weights to values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

# Example inputs
query = torch.rand(2, 5, 64)  # Batch size of 2, sequence length of 5, embedding size of 64
key = torch.rand(2, 5, 64)
value = torch.rand(2, 5, 64)
fractal_params = torch.rand(2, 5)  # Shape is correct now

# Optional attention mask (for example, ignoring padding)
attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])  # Shape should match query/key/value

# Call the fractal_attention function
output, attention_weights = fractal_attention(query, key, value, fractal_params, attention_mask)

# Visualization
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Visualize attention weights for both batches
for i in range(2):
    ax = axs[i, 0]
    im = ax.imshow(attention_weights[i].detach().numpy(), cmap='viridis')
    ax.set_title(f'Attention Weights (Batch {i+1})')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    fig.colorbar(im, ax=ax)

# Visualize output for both batches
for i in range(2):
    ax = axs[i, 1]
    im = ax.imshow(output[i].detach().numpy(), cmap='viridis')
    ax.set_title(f'Output (Batch {i+1})')
    ax.set_xlabel('Embedding')
    ax.set_ylabel('Sequence')
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# Print shapes for reference
print("Attention Weights Shape:", attention_weights.shape)
print("Output Shape:", output.shape)

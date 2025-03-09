import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import os
import math
import torch.nn.functional as F
from pathlib import Path

def extract_activations(model, input_text, dataset, device, layer_idx=None):
    """Extract activations from a specific layer or all layers."""
    model.eval()
    
    # Convert input text to model input format
    chars = [ch for ch in input_text]
    indices = [dataset.char_to_idx.get(ch, dataset.char_to_idx.get(' ', 0)) for ch in chars]
    
    # Handle sequence length
    if len(indices) > model.config.seq_length:
        context_indices = indices[-model.config.seq_length:]
    else:
        padding = model.config.seq_length - len(indices)
        context_indices = [0] * padding + indices
    
    # Convert to tensor
    context = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)
    mask = model._generate_square_subsequent_mask(context.size(1)).to(device)
    
    # We'll store activations for each layer
    activations = {}
    
    # Register hooks to capture activations
    hooks = []
    
    # For decoder-only model
    if model.model_type == "decoder_only":
        for i, block in enumerate(model.decoder):
            if layer_idx is not None and i != layer_idx:
                continue
                
            # Get attention activations
            def get_attn_fn(layer_id):
                def hook_fn(module, input, output):
                    activations[f'layer_{layer_id}_attn'] = output.detach().cpu()
                return hook_fn
            
            # Get feed-forward activations
            def get_ff_fn(layer_id):
                def hook_fn(module, input, output):
                    activations[f'layer_{layer_id}_ff'] = output.detach().cpu()
                return hook_fn
            
            # Register hooks
            hooks.append(block.attn.register_forward_hook(get_attn_fn(i)))
            hooks.append(block.feed_forward.register_forward_hook(get_ff_fn(i)))
    
    # Run forward pass to get activations
    with torch.no_grad():
        _ = model(context, mask)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations, chars

def plot_attention_patterns(model, input_text, dataset, device, save_dir="./visualizations"):
    """Visualize attention patterns for each head and layer."""
    model.eval()
    
    # Convert input text to model input format
    chars = [ch for ch in input_text]
    indices = [dataset.char_to_idx.get(ch, dataset.char_to_idx.get(' ', 0)) for ch in chars]
    
    # Handle sequence length
    if len(indices) > model.config.seq_length:
        indices = indices[-model.config.seq_length:]
    else:
        padding = model.config.seq_length - len(indices)
        indices = [0] * padding + indices
    
    # Convert to tensor
    context = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    mask = model._generate_square_subsequent_mask(context.size(1)).to(device)
    
    # Create directory to save plots
    os.makedirs(save_dir, exist_ok=True)
    
    # We'll store attention weights for each layer and head
    attention_weights = {}
    
    # Register hooks to capture attention weights
    hooks = []
    
    def attention_hook(module, input, output):
        # This assumes scores is calculated inside the attention mechanism
        # and we're interested in the attention weights after softmax
        q = input[0]
        k = input[1]
        v = input[2]
        
        # Calculate attention scores similar to how it's done in your model
        batch_size, seq_length, _ = q.shape
        head_dim = module.head_dim
        num_heads = module.num_heads
        
        q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
        
        # Calculate scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 1, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights
    
    # Extract attention patterns and visualize
    if model.model_type == "decoder_only":
        for layer_idx, block in enumerate(model.decoder):
            # Register hooks for attention weights
            with torch.no_grad():
                # Embed and position encode
                x = model.embedding(context) * math.sqrt(model.config.d_model)
                x = model.pos_encoder(x)
                
                # Process through previous layers
                for prev_idx in range(layer_idx):
                    x = model.decoder[prev_idx](x, mask)
                
                # Extract attention weights for this layer
                def get_attention_weights(module, input, output):
                    # Extract q, k, v from input
                    residual = input[0]
                    normalized = model.decoder[layer_idx].norm1(residual)
                    
                    # Get q, k, v
                    q = model.decoder[layer_idx].attn.q_proj(normalized)
                    k = model.decoder[layer_idx].attn.k_proj(normalized)
                    v = model.decoder[layer_idx].attn.v_proj(normalized)
                    
                    # Reshape for multi-head attention
                    batch_size = q.shape[0]
                    seq_length = q.shape[1]
                    head_dim = model.decoder[layer_idx].attn.head_dim
                    num_heads = model.decoder[layer_idx].attn.num_heads
                    
                    q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
                    k = k.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
                    
                    # Calculate attention scores
                    scores = torch.matmul(q, k.transpose(-1, -2)) * model.decoder[layer_idx].attn.scale
                    
                    # Apply mask - using mask from the outer scope
                    current_mask = mask  # Reference the mask from outer scope
                    if current_mask is not None:
                        if current_mask.dim() == 2:
                            current_mask = current_mask.unsqueeze(0).unsqueeze(0)
                        scores = scores.masked_fill(current_mask == 1, -1e4)
                    
                    # Attention weights
                    attn_weights = F.softmax(scores, dim=-1)
                    attention_weights[f'layer_{layer_idx}'] = attn_weights.detach().cpu().numpy()
                
                # Register hook
                hook = model.decoder[layer_idx].register_forward_hook(get_attention_weights)
                
                # Forward pass
                x = model.decoder[layer_idx](x, mask)
                
                # Remove hook
                hook.remove()
                
                # Plot attention patterns for this layer
                attn_weights = attention_weights[f'layer_{layer_idx}']
                plot_attention_heatmaps(attn_weights[0], layer_idx, chars, save_dir)

def plot_attention_heatmaps(attention_weights, layer_idx, tokens, save_dir):
    """Plot attention heatmaps for each attention head."""
    num_heads = attention_weights.shape[0]
    
    # Create a figure with subplots for each head
    fig_size = min(20, max(12, num_heads * 3))  # Dynamic figure size
    fig, axes = plt.subplots(nrows=1, ncols=num_heads, figsize=(fig_size, 10))
    
    if num_heads == 1:
        axes = [axes]
    
    # Define a title for the figure
    fig.suptitle(f'Attention Patterns - Layer {layer_idx}', fontsize=16)
    
    # Display meaningful tokens for x and y axes (remove padding tokens)
    display_tokens = tokens
    seq_len = attention_weights.shape[1]
    if len(tokens) < seq_len:
        # There was padding
        pad_length = seq_len - len(tokens)
        display_tokens = ['[PAD]'] * pad_length + tokens
    
    # Plot each attention head
    for h in range(num_heads):
        ax = axes[h]
        
        # Plot the attention weights
        im = ax.imshow(attention_weights[h], cmap='viridis')
        
        # Add head title
        ax.set_title(f'Head {h}')
        
        # Set tick labels for x and y axes
        if len(display_tokens) <= 20:
            # Show all tokens if there aren't too many
            ax.set_xticks(np.arange(len(display_tokens)))
            ax.set_yticks(np.arange(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(display_tokens, fontsize=8)
        else:
            # Show selected tokens for readability
            stride = max(1, len(display_tokens) // 10)
            indices = list(range(0, len(display_tokens), stride))
            ax.set_xticks(indices)
            ax.set_yticks(indices)
            ax.set_xticklabels([display_tokens[i] for i in indices], rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels([display_tokens[i] for i in indices], fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'attention_layer_{layer_idx}.png'), dpi=150)
    plt.close(fig)

def visualize_neuron_activation(model, input_texts, dataset, device, save_dir="./visualizations"):
    """Visualize neuron activations across different inputs."""
    os.makedirs(save_dir, exist_ok=True)
    
    # For each layer, get FF activations for each input
    layer_activations = []
    
    # Create a figure for each layer
    for layer_idx in range(model.config.n_layers):
        layer_data = []
        
        for i, text in enumerate(input_texts):
            # Get activations for this input
            activations, chars = extract_activations(model, text, dataset, device, layer_idx)
            
            # Get feed-forward activations (which represent neuron firings)
            ff_key = f'layer_{layer_idx}_ff'
            if ff_key in activations:
                # Get the last token's activation (prediction for next token)
                ff_activations = activations[ff_key][0, -1].numpy()
                layer_data.append(ff_activations)
        
        if layer_data:
            # Convert to array for easier plotting
            layer_data = np.array(layer_data)
            layer_activations.append(layer_data)
            
            # Plot heatmap of neuron activations
            plt.figure(figsize=(12, 8))
            sns.heatmap(layer_data, cmap='viridis')
            plt.title(f'Layer {layer_idx} - Neuron Activations')
            plt.xlabel('Neuron Index')
            plt.ylabel('Input Text Sample')
            plt.savefig(os.path.join(save_dir, f'neuron_activations_layer_{layer_idx}.png'))
            plt.close()
            
            # Plot histogram of activations
            plt.figure(figsize=(10, 6))
            plt.hist(layer_data.flatten(), bins=50, alpha=0.75)
            plt.title(f'Layer {layer_idx} - Distribution of Neuron Activations')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(save_dir, f'activation_histogram_layer_{layer_idx}.png'))
            plt.close()

def visualize_neurons(model_path, device, example_texts=None):
    """Main function to generate neuron visualizations."""
    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct config from saved data
    from memorizeit import Config, ConfigurableTransformer, TransformerDataset
    
    config = Config()
    for k, v in checkpoint['config'].items():
        setattr(config, k, v)
    
    # Create dataset
    vocab = checkpoint['vocab']
    dataset = TransformerDataset("A", config.seq_length, config.model_type)
    dataset.char_to_idx = vocab['char_to_idx']
    dataset.idx_to_char = vocab['idx_to_char']
    
    # Initialize model
    model = ConfigurableTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully - {config.model_type} with {config.n_layers} layers")
    
    # Set up example texts if none provided
    if example_texts is None:
        example_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "To be or not to be, that is the question.",
            "Machine learning models are fascinating.",
            "Neural networks can learn complex patterns.",
            "Transformers have revolutionized NLP tasks.",
            "Attention is all you need, as they say.",
            "Deep learning is a subset of machine learning."
        ]
    
    # Create visualization directory
    vis_dir = Path("./neuron_visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("Generating attention pattern visualizations...")
    for i, text in enumerate(example_texts[:3]):  # Use first 3 examples for attention
        subdir = vis_dir / f"attention_sample_{i}"
        subdir.mkdir(exist_ok=True)
        plot_attention_patterns(model, text, dataset, device, save_dir=str(subdir))
    
    print("Generating neuron activation visualizations...")
    visualize_neuron_activation(model, example_texts, dataset, device, save_dir=str(vis_dir))
    
    print(f"Visualizations saved to {vis_dir.absolute()}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize neurons in a decoder-only transformer model")
    parser.add_argument("--model", type=str, default="best_model.pt", help="Path to model checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_neurons(args.model, device)
import torch
import torch.nn.functional as F
import os

# Load the main script for model definitions - update to use ConfigurableTransformer
from memorizeit import ConfigurableTransformer, device

def load_model(checkpoint_path='best_model.pt'):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Make sure you've trained the model first.")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configuration and vocabulary
    config_dict = checkpoint['config']
    vocab = checkpoint['vocab']
    
    # Create a Config object with loaded parameters
    class Config:
        pass
    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # Set model_type if not in saved config (for backward compatibility)
    if not hasattr(config, 'model_type'):
        config.model_type = "decoder_only"
    
    # Initialize model with loaded config
    model = ConfigurableTransformer(config).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Print model statistics
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate neuron counts
    neurons_per_layer = config.d_model
    total_neurons = config.d_model * config.n_layers
    attention_neurons = config.n_heads * config.d_model // config.n_heads
    feedforward_neurons = config.d_model * 4  # Typical FFN size is 4x embedding dim
    
    # Calculate comprehensive neuron count
    embedding_neurons = config.vocab_size * config.d_model  # Embedding matrix
    positional_neurons = config.seq_length * config.d_model  # Positional encodings
    attention_neurons_total = attention_neurons * config.n_layers
    feedforward_neurons_total = feedforward_neurons * config.n_layers
    output_neurons = config.d_model * config.vocab_size  # Output projection
    
    # Grand total (all neural connections in the model)
    grand_total_neurons = (embedding_neurons + positional_neurons + 
                          attention_neurons_total + feedforward_neurons_total + 
                          output_neurons)
    
    print("\n" + "=" * 50)
    print("MODEL STATISTICS")
    print("=" * 50)
    print(f"Training epochs completed: {checkpoint['epoch'] + 1}")
    print(f"Final validation loss: {checkpoint['loss']:.6f}")
    
    # Update the architecture display based on model_type
    if config.model_type == "decoder_only":
        model_architecture = "Decoder-only Transformer (GPT-style)"
    elif config.model_type == "encoder_only":
        model_architecture = "Encoder-only Transformer (BERT-style)"
    else:
        model_architecture = "Encoder-Decoder Transformer (Original-style)"
        
    print(f"Model type: {config.model_type}")
    print(f"Model architecture: {model_architecture}")
    print(f"Number of parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")
    
    # Add comprehensive neuron information
    print("\nNeuron Structure:")
    print(f"Hidden dimension (neurons per layer): {config.d_model}")
    print(f"Total hidden neurons across all layers: {total_neurons}")
    print(f"Attention neurons per head: {config.d_model // config.n_heads}")
    print(f"Total attention neurons: {attention_neurons_total}")
    print(f"Feedforward neurons per layer: {feedforward_neurons}")
    print(f"Total feedforward neurons: {feedforward_neurons_total}")
    print(f"GRAND TOTAL NEURAL CONNECTIONS: {grand_total_neurons:,}")
    
    print("\nArchitecture Details:")
    print(f"Number of attention heads: {config.n_heads}")
    print(f"Number of layers: {config.n_layers}")
    print(f"Context length: {config.seq_length} characters")
    print(f"Vocabulary size: {config.vocab_size} characters")
    print(f"Dropout rate: {config.dropout}")
    
    print("\nMemory usage:")
    size_bytes = sum(param.nelement() * param.element_size() for param in model.parameters())
    print(f"Model size: {size_bytes / 1024**2:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    return model, vocab, config

def generate_text(model, vocab, config, prompt, max_length=500, temperature=0.7):
    """Generate text based on a prompt."""
    # Character to index mapping
    char_to_idx = vocab['char_to_idx']
    idx_to_char = vocab['idx_to_char']
    
    # Convert prompt to indices
    chars = [ch for ch in prompt]
    indices = [char_to_idx.get(ch, char_to_idx.get(' ', 0)) for ch in chars]
    
    # Generate text
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input sequence
            if len(indices) > config.seq_length:
                context_indices = indices[-config.seq_length:]
            else:
                context_indices = indices
            
            # Pad input if needed
            padding = config.seq_length - len(context_indices)
            if padding > 0:
                # Padding at the beginning
                context_indices = [0] * padding + context_indices
            
            # Convert to tensor
            context = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            # Create causal mask
            mask = model._generate_square_subsequent_mask(context.size(1)).to(device)
            
            # Get predictions based on model type
            if config.model_type == "decoder_only":
                # Standard approach for decoder-only models
                output = model(context, tgt_mask=mask)
            elif config.model_type == "encoder_only":
                # For encoder-only models, we use the context without a mask
                output = model(context)
            else:  # encoder_decoder
                # For encoder-decoder, use empty/starter sequence as decoder input
                # In inference we'd typically use teacher forcing with previously generated tokens
                output = model(context, context[:, -1:], src_mask=None, tgt_mask=None)
                
            next_token_logits = output[0, -1] / temperature
            
            # Top-k filtering
            top_k = 40
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            filtered_logits = next_token_logits.clone()
            filtered_logits[filtered_logits < top_k_logits[-1]] = -float('inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            indices.append(next_token)
            chars.append(idx_to_char[next_token])
    
    return ''.join(chars)

def main():
    # Load the model
    model_data = load_model()
    if not model_data:
        return
    
    model, vocab, config = model_data
    
    print("\n" + "=" * 50)
    print("TEXT GENERATION INTERFACE")
    print("=" * 50)
    print("Type your prompt and press Enter to generate text.")
    print("Type 'exit' to quit.")
    print("Type 'settings' to change temperature or max length.")
    print("Type 'stats' to view model statistics again.")
    
    max_length = 500
    temperature = 0.7
    
    while True:
        print("\n" + "-" * 50)
        prompt = input("Enter your prompt: ")
        
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        
        elif prompt.lower() == 'stats':
            # Print runtime statistics 
            num_params = sum(p.numel() for p in model.parameters())
            
            # Calculate neuron counts
            neurons_per_layer = config.d_model
            total_neurons = config.d_model * config.n_layers
            
            # Calculate comprehensive neuron count (simplified version for quick stats)
            embedding_neurons = config.vocab_size * config.d_model
            feedforward_neurons = config.d_model * 4 * config.n_layers
            output_neurons = config.d_model * config.vocab_size
            grand_total = embedding_neurons + total_neurons + feedforward_neurons + output_neurons
            
            print("\n" + "=" * 50)
            print("MODEL STATISTICS")
            print("=" * 50)
            print(f"Model architecture: Decoder-only Transformer (GPT-style)")
            print(f"Parameters: {num_params:,}")
            print(f"Total hidden neurons: {total_neurons:,}")
            print(f"GRAND TOTAL NEURAL CONNECTIONS: {grand_total:,}")
            print(f"Neurons per layer: {neurons_per_layer}")
            print(f"Context length: {config.seq_length} characters")
            print(f"Current temperature: {temperature}")
            
            if torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            continue
        
        elif prompt.lower() == 'settings':
            try:
                temp = float(input(f"Enter temperature (current: {temperature}): "))
                if 0.1 <= temp <= 2.0:
                    temperature = temp
                    print(f"Temperature set to {temperature}")
                else:
                    print("Temperature should be between 0.1 and 2.0")
            except ValueError:
                print("Using default temperature")
                
            try:
                length = int(input(f"Enter max length (current: {max_length}): "))
                if length > 0:
                    max_length = length
                    print(f"Max length set to {max_length}")
                else:
                    print("Length must be positive")
            except ValueError:
                print("Using default max length")
            continue
        
        elif not prompt:
            print("Prompt cannot be empty.")
            continue
        
        print("\nGenerating text...\n")
        
        # Generate text
        generated_text = generate_text(
            model, 
            vocab,
            config,
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        # Display the result
        print("\n" + "=" * 50)
        print("GENERATED TEXT:")
        print("=" * 50)
        print(generated_text)
        
        # Show some statistics
        print("\n" + "-" * 50)
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Generated text length: {len(generated_text)} characters")
        print(f"New content: {len(generated_text) - len(prompt)} characters")
    
if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import math
import time
from torch.amp import autocast, GradScaler
import copy  # Add this line
import argparse  # Add this line
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Move these print statements inside a function or the main function
def print_system_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration class with improved parameters
class Config:
    def __init__(self):
        self.vocab_size = 256  # ASCII characters
        self.seq_length = 128  # context length
        self.d_model = 256     # embedding dimension
        self.n_heads = 8       # Number of attention heads
        self.n_layers = 8      # number of layers
        self.dropout = 0.1     # Dropout rate
        self.learning_rate = 3e-4  # Learning rate
        self.batch_size = 32   # Reduced to handle larger context length
        self.epochs = 100      # Change from 100 to 200 for the second run, 300 for third run, etc.
        self.warmup_steps = 1000  # Learning rate warmup steps
        self.model_type = "decoder_only"  # Default model type

# Text dataset
class TransformerDataset(Dataset):
    def __init__(self, text, seq_length, model_type="decoder_only"):
        self.text = text
        self.seq_length = seq_length
        self.model_type = model_type
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
    def __len__(self):
        if self.model_type == "encoder_decoder":
            # For encoder-decoder we might want to create src-tgt pairs differently
            # This is just an example - you'd customize based on your task
            return max(1, len(self.text) // (2 * self.seq_length))
        else:
            return len(self.text) - self.seq_length - 1
    
    def __getitem__(self, idx):
        if self.model_type == "encoder_decoder":
            # Example: consecutive chunks as src and tgt
            src_start = idx * 2 * self.seq_length
            tgt_start = src_start + self.seq_length
            
            # Make sure we don't go out of bounds
            if tgt_start + self.seq_length > len(self.text):
                src_start = max(0, len(self.text) - 2 * self.seq_length)
                tgt_start = src_start + self.seq_length
            
            src_text = self.text[src_start:src_start + self.seq_length]
            tgt_text = self.text[tgt_start:tgt_start + self.seq_length]
            
            src = torch.tensor([self.char_to_idx[ch] for ch in src_text], dtype=torch.long)
            tgt = torch.tensor([self.char_to_idx[ch] for ch in tgt_text], dtype=torch.long)
        else:
            # Original implementation for encoder-only or decoder-only
            x = self.text[idx:idx + self.seq_length]
            y = self.text[idx + 1:idx + self.seq_length + 1]
            
            src = torch.tensor([self.char_to_idx[ch] for ch in x], dtype=torch.long)
            tgt = torch.tensor([self.char_to_idx[ch] for ch in y], dtype=torch.long)
        
        return src, tgt

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    # Update the MultiHeadAttention forward method to handle different mask types
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        # Linear projections and reshape
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 2:  # Square attention mask [seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            
            # Make sure mask is broadcastable and using proper value range
            scores = scores.masked_fill(mask == 1, -1e4)  # Use -1e4 instead of -1e9
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.o_proj(context)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern choice, you could use nn.ReLU() too
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention block with residual connection and normalization
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward block with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class ManualTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(ManualTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention (encoder-decoder attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # Self-attention block
        q1 = self.self_attn.q_proj(x)
        k1 = self.self_attn.k_proj(x)
        v1 = self.self_attn.v_proj(x)
        self_attn_output = self._attention_block(q1, k1, v1, tgt_mask, self.self_attn)
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)
        
        # Cross-attention block
        if memory is not None:
            q2 = self.cross_attn.q_proj(x)
            k2 = self.cross_attn.k_proj(memory)
            v2 = self.cross_attn.v_proj(memory)
            cross_attn_output = self._attention_block(q2, k2, v2, src_mask, self.cross_attn)
            x = x + self.dropout2(cross_attn_output)
            x = self.norm2(x)
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x

    def _attention_block(self, q, k, v, mask, attn_module):
        # Helper method for attention calculation
        batch_size = q.shape[0]
        q_seq_length = q.shape[1]
        k_seq_length = k.shape[1]
        head_dim = attn_module.head_dim
        num_heads = attn_module.num_heads
        
        # Reshape for multi-head attention
        q = q.view(batch_size, q_seq_length, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, k_seq_length, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, k_seq_length, num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 1, -1e4)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_module.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        
        # Reshape output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, q_seq_length, attn_module.d_model
        )
        output = attn_module.o_proj(context)
        
        return output


class ManualTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(ManualTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])
    
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(GPTBlock, self).__init__()
        # Self-attention with causal mask
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm architecture (often works better)
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout1(self.attn(x, mask))
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.feed_forward(x))
        
        return x


class ConfigurableTransformer(nn.Module):
    def __init__(self, config):
        super(ConfigurableTransformer, self).__init__()
        self.config = config
        self.model_type = config.model_type  # "encoder_only", "decoder_only", or "encoder_decoder"
        
        # Token embedding (shared across variants)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding (shared across variants)
        self.pos_encoder = PositionalEncoding(config.d_model, config.seq_length, config.dropout)
        
        # Encoder (used in encoder_only and encoder_decoder)
        if self.model_type in ["encoder_only", "encoder_decoder"]:
            encoder_layer = EncoderLayer(
                d_model=config.d_model,
                num_heads=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout
            )
            self.encoder = ManualTransformerEncoder(encoder_layer, config.n_layers)
        
        # Decoder (used in decoder_only and encoder_decoder)
        if self.model_type in ["decoder_only", "encoder_decoder"]:
            if self.model_type == "decoder_only":
                # For decoder-only (GPT-style), use the specialized GPT block
                self.decoder = nn.ModuleList([
                    GPTBlock(
                        d_model=config.d_model,
                        num_heads=config.n_heads,
                        dim_feedforward=config.d_model * 4,
                        dropout=config.dropout
                    ) for _ in range(config.n_layers)
                ])
            else:
                # For encoder-decoder (BERT+GPT style), use cross-attention decoder
                decoder_layer = DecoderLayer(
                    d_model=config.d_model,
                    num_heads=config.n_heads,
                    dim_feedforward=config.d_model * 4,
                    dropout=config.dropout
                )
                self.decoder = ManualTransformerDecoder(decoder_layer, config.n_layers)
        
        # Final normalization layer
        self.norm = nn.LayerNorm(config.d_model)
        
        # Output projection
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        """
        Unified forward method supporting all configurations
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_len]
            tgt: Optional target sequence tensor [batch_size, tgt_seq_len]
                 (Only used for encoder-decoder mode)
            src_mask: Optional source mask for encoder
            tgt_mask: Optional target mask for decoder
        """
        if self.model_type == "encoder_only":
            # BERT-style: just encode the source
            src = self.embedding(src) * math.sqrt(self.config.d_model)
            src = self.pos_encoder(src)
            encoder_output = self.encoder(src, src_mask)
            encoder_output = self.norm(encoder_output)
            return self.output_layer(encoder_output)
            
        elif self.model_type == "decoder_only":
            # GPT-style: process through decoder-only path
            # Generate causal mask if none provided
            if tgt_mask is None:
                tgt_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
            
            # Embedding + positional encoding
            x = self.embedding(src) * math.sqrt(self.config.d_model)
            x = self.pos_encoder(x)
            
            # Process through decoder blocks
            for block in self.decoder:
                x = block(x, tgt_mask)
                
            # Final normalization
            x = self.norm(x)
            return self.output_layer(x)
            
        elif self.model_type == "encoder_decoder":
            # Transformer with encoder-decoder (like original Transformer paper)
            if tgt is None:
                raise ValueError("Target sequence required for encoder-decoder mode")
                
            # Generate causal mask for decoder if none provided
            if tgt_mask is None:
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(src.device)
            
            # Embed and encode source
            src = self.embedding(src) * math.sqrt(self.config.d_model)
            src = self.pos_encoder(src)
            encoder_output = self.encoder(src, src_mask)
            
            # Embed target
            tgt = self.embedding(tgt) * math.sqrt(self.config.d_model)
            tgt = self.pos_encoder(tgt)
            
            # Decode
            decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
            decoder_output = self.norm(decoder_output)
            return self.output_layer(decoder_output)
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence to prevent attending to future positions."""
        # Create upper triangular mask (1s in upper triangle)
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask


# Move these function definitions to BEFORE the main() function:

def train(model, dataloader, optimizer, scheduler, criterion, config):
    model.train()
    total_loss = 0
    start_time = time.time()
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the base model from DDP if needed
    base_model = model.module if hasattr(model, 'module') else model
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Different handling based on model type
        if config.model_type == "encoder_decoder":
            # For encoder-decoder, use src as input and tgt as target
            # We need to shift tgt for teacher forcing (input is tgt[:-1], output is tgt[1:])
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Create masks
            src_mask = None  # Could be padding mask if needed
            tgt_mask = base_model._generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                output = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(output.reshape(-1, config.vocab_size), tgt_output.reshape(-1))
        
        else:  # encoder_only or decoder_only
            # Create mask appropriate for the model type
            if config.model_type == "decoder_only":
                mask = base_model._generate_square_subsequent_mask(src.size(1)).to(device)
            else:  # encoder_only
                mask = None  # No causal mask needed for encoder-only
            
            optimizer.zero_grad()
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                output = model(src, src_mask=mask)
                loss = criterion(output.reshape(-1, config.vocab_size), tgt.reshape(-1))
        
        # Rest of the training loop remains the same
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx+1}/{len(dataloader)} | Loss: {total_loss / (batch_idx+1):.4f} | Time: {elapsed:.2f}s")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0
    
    # Get the base model from DDP if needed
    base_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Different handling based on model type
            if config.model_type == "encoder_decoder":
                # For encoder-decoder, use src as input and tgt as target
                # We need to shift tgt for teacher forcing (input is tgt[:-1], output is tgt[1:])
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Create masks
                src_mask = None  # Could be padding mask if needed
                tgt_mask = base_model._generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                
                # Forward pass
                output = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(output.reshape(-1, config.vocab_size), tgt_output.reshape(-1))
            
            else:  # encoder_only or decoder_only
                # Create mask appropriate for the model type
                if config.model_type == "decoder_only":
                    mask = base_model._generate_square_subsequent_mask(src.size(1)).to(device)
                else:  # encoder_only
                    mask = None  # No causal mask needed for encoder-only
                
                # Forward pass
                output = model(src, src_mask=mask)
                loss = criterion(output.reshape(-1, config.vocab_size), tgt.reshape(-1))
                
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_text(model, dataset, start_text, max_len=500, temperature=0.6):
    model.eval()
    
    # Convert start text to indices
    chars = [ch for ch in start_text]
    indices = [dataset.char_to_idx.get(ch, dataset.char_to_idx.get(' ', 0)) for ch in chars]
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_len):
            # Prepare input sequence
            if len(indices) > model.config.seq_length:
                context_indices = indices[-model.config.seq_length:]
            else:
                context_indices = indices
            
            # Pad input if needed
            padding = model.config.seq_length - len(context_indices)
            if padding > 0:
                context_indices = [0] * padding + context_indices
            
            # Convert to tensor
            context = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            # Create causal mask
            mask = model._generate_square_subsequent_mask(context.size(1)).to(device)
            
            # Get predictions
            output = model(context, mask)
            next_token_logits = output[0, -1] / temperature
            
            # Sample from distribution
            # Optional: Top-k filtering
            top_k = 40
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            filtered_logits = next_token_logits.clone()
            filtered_logits[filtered_logits < top_k_logits[-1]] = -float('inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            indices.append(next_token)
            chars.append(dataset.idx_to_char[next_token])
    
    return ''.join(chars)


def setup(rank, world_size, backend="nccl"):
    """
    Setup distributed training environment for the given rank
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group with the specified backend
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a configurable transformer model")
    parser.add_argument("--model-type", type=str, default="decoder_only", 
                        choices=["encoder_only", "decoder_only", "encoder_decoder"],
                        help="Type of transformer architecture to use")
    parser.add_argument("--d-model", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--n-layers", type=int, default=8,
                        help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Total training batch size across all GPUs")
    parser.add_argument("--seq-length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--backend", type=str, default="nccl",
                        choices=["nccl", "gloo"],
                        help="Distributed backend (nccl or gloo)")
    # Add more arguments as needed
    
    return parser.parse_args()

def main_worker(gpu, ngpus_per_node, args):
    """
    Main worker function for distributed training
    """
    # Setup distributed environment with the specified backend
    rank = gpu
    setup(rank, ngpus_per_node, args.backend)
    
    # Print only from main process to avoid clutter
    is_main_process = (rank == 0)
    
    if is_main_process:
        print_system_info()
        print(f"Using device: cuda:{rank}")
        print(f"Using {args.backend} backend for distributed training")
    
    # Update config from arguments
    config = Config()
    config.model_type = args.model_type
    config.d_model = args.d_model
    config.n_layers = args.n_layers
    config.n_heads = args.n_heads
    
    # Scale learning rate based on batch size
    base_batch_size = 32  # Original design batch size
    total_batch_size = args.batch_size  # Total batch size across all GPUs
    lr_scale_factor = total_batch_size / base_batch_size
    config.learning_rate *= lr_scale_factor
    
    if is_main_process:
        print(f"Total batch size: {total_batch_size}")
        print(f"Learning rate scaled by {lr_scale_factor:.2f}x to {config.learning_rate:.6f}")
    
    # Set per-GPU batch size
    config.batch_size = args.batch_size // ngpus_per_node
    config.seq_length = args.seq_length
    
    # Load text data (all processes need this for dataset)
    data_file = "./data.txt"
    if not os.path.exists(data_file):
        if is_main_process:
            print(f"File not found: {data_file}")
        return
    
    # Read text data
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset
    dataset = TransformerDataset(text, config.seq_length, config.model_type)
    config.vocab_size = dataset.vocab_size
    
    # Use DistributedSampler
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=ngpus_per_node,
        rank=rank
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=False,  # Sampler handles shuffling
        pin_memory=True,
        num_workers=4,
        sampler=train_sampler
    )
    
    # Create model and move to correct device
    model = ConfigurableTransformer(config).to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    if is_main_process:
        print(f"Model type: {config.model_type}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize optimizer (after DDP wrapping)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Check if saved model exists and load if it does
    checkpoint_path = 'best_model.pt'
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        # Load checkpoint only on the main process and broadcast to others
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # For DDP models, need to handle state dict specially
        if 'module' not in list(checkpoint['model_state_dict'].keys())[0]:
            # Add 'module.' prefix for compatibility with DDP
            new_state_dict = {f'module.{k}': v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        
        if is_main_process:
            print(f"Resuming from epoch {start_epoch} with previous best loss: {best_loss:.4f}")
    else:
        if is_main_process:
            print("No checkpoint found, starting training from scratch")
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(dataloader),
        epochs=config.epochs,
        pct_start=0.1
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Store losses for plotting
    train_losses = []
    val_losses = []
    val_loss = best_loss
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        # Set epoch for sampler (important for proper shuffling)
        train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Modify train function to support DDP
        train_loss = train(model, dataloader, optimizer, scheduler, criterion, config)
        
        # Average losses across processes
        train_loss_tensor = torch.tensor([train_loss]).to(rank)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / ngpus_per_node
        
        if is_main_process:
            print(f"Train loss: {train_loss:.4f}")
            train_losses.append(train_loss)
        
        # Evaluate - only needed on one process to avoid duplicate work
        if is_main_process:
            val_loss = evaluate(model, dataloader, criterion, config)
            print(f"Validation loss: {val_loss:.4f}")
            val_losses.append(val_loss)
            
            # Add this inside the training loop after each epoch
            if torch.cuda.is_available():
                print(f"GPU memory usage: {torch.cuda.memory_allocated(rank) / 1024**2:.2f} MB")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                # Save model without 'module.' prefix for easier loading
                model_state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'config': {k: v for k, v in config.__dict__.items()},
                    'vocab': {
                        'char_to_idx': dataset.char_to_idx,
                        'idx_to_char': dataset.idx_to_char
                    }
                }, checkpoint_path)
                print(f"Saved new best model with loss: {val_loss:.4f}")
            
            # Generate text occasionally
            if (epoch + 1) % 10 == 0:
                # For generating text, use the model without DDP wrapper
                unwrapped_model = model.module
                sample_text = generate_text(unwrapped_model, dataset, text[:50], 150)
                print(f"\nGenerated text:\n{sample_text}\n")
    
    # Clean up distributed training resources
    dist.destroy_process_group()
    
    # Final operations only on main process
    if is_main_process:
        # Save final model
        final_path = 'final_model.pt'
        model_state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
        torch.save({
            'epoch': max(config.epochs - 1, start_epoch),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'config': {k: v for k, v in config.__dict__.items()},
            'vocab': {
                'char_to_idx': dataset.char_to_idx,
                'idx_to_char': dataset.idx_to_char
            }
        }, final_path)
        print(f"Saved final model to {final_path}")
        
        # Final evaluation using best model
        print("\nTraining complete!")
        best_model_state = torch.load(checkpoint_path)['model_state_dict']
        # Load directly to module without DDP for evaluation
        unwrapped_model = model.module
        unwrapped_model.load_state_dict(best_model_state)
        val_loss = evaluate(unwrapped_model, dataloader, criterion, config)
        print(f"Final validation loss: {val_loss:.4f}")
        
        # Generate final text sample
        sample_text = generate_text(unwrapped_model, dataset, text[:50], 1000, temperature=0.7)
        print(f"\nGenerated text:\n{sample_text}\n")
        
        # Plot training curve (existing plotting code)
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            # Your existing plotting code...
        except ImportError:
            print("Could not generate training curve plot: matplotlib not available")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get number of available GPUs
    ngpus_per_node = torch.cuda.device_count()
    
    if ngpus_per_node > 1:
        print(f"Using {ngpus_per_node} GPUs for distributed training")
        # Use multiprocessing to spawn one process per GPU
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print("Only one GPU detected, running without distributed training")
        # Fall back to single GPU training
        main_worker(0, 1, args)

if __name__ == "__main__":
    main()
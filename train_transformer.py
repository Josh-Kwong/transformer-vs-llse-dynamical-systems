"""
train_transformer.py


- 3 transformer layers
- Hidden dimension: 72
- 6 attention heads (head dimension: 12)
- Next-step prediction with MSE loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class SequenceDataset(Dataset):
   
    
    def __init__(self, sequences):
        
        self.sequences = torch.FloatTensor(sequences)
        self.n_sequences = sequences.shape[0]
        self.T = sequences.shape[1]
        self.d = sequences.shape[2]
        
    def __len__(self):
        
        return self.n_sequences * (self.T - 1)
    
    def __getitem__(self, idx):
        
        seq_idx = idx // (self.T - 1)
        time_idx = idx % (self.T - 1)
        
        input_seq = self.sequences[seq_idx, :time_idx + 1] 
        target = self.sequences[seq_idx, time_idx + 1]    
        
        return input_seq, target, time_idx + 1



def collate_fn(batch):
    
    input_seqs, targets, seq_lens = zip(*batch)
    

    max_len = max(seq_lens)
    batch_size = len(batch)
    d = targets[0].shape[0]
    
    
    padded_inputs = torch.zeros(batch_size, max_len, d)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, (seq, length) in enumerate(zip(input_seqs, seq_lens)):
        padded_inputs[i, :length] = seq
        attention_mask[i, :length] = True
    
    targets = torch.stack(targets)
    
    return padded_inputs, targets, attention_mask



class PositionalEncoding(nn.Module):
    #Sinusoidal positional encoding
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        
        return x + self.pe[:x.size(1)]

class TransformerPredictor(nn.Module):
    
    
    def __init__(self, input_dim=5, hidden_dim=72, n_layers=3, 
                 n_heads=6, head_dim=12, dropout=0.1, max_len=20):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,  
            dropout=dropout,
            activation='gelu',  
            batch_first=True,   
            norm_first=True     # Pre-norm 
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
       
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        
        self._init_weights()
    
    def _init_weights(self):
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, attention_mask=None):
        
        x = self.input_projection(x)  
        
        
        x = self.pos_encoder(x)
        
        
        if attention_mask is not None:
            
            padding_mask = ~attention_mask
        else:
            padding_mask = None
        
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        
        if attention_mask is not None:
            
            seq_lens = attention_mask.sum(dim=1) - 1  
            last_hidden = x[torch.arange(x.size(0)), seq_lens]
        else:
            last_hidden = x[:, -1]  
        
       
        prediction = self.output_projection(last_hidden)  
        
        return prediction



def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (input_seqs, targets, attention_mask) in enumerate(pbar):
        
        input_seqs = input_seqs.to(device)
        targets = targets.to(device)
        attention_mask = attention_mask.to(device)
        
        
        predictions = model(input_seqs, attention_mask)
        
        
        loss = F.mse_loss(predictions, targets)
        
        
        optimizer.zero_grad()
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        
        if scheduler is not None:
            scheduler.step()
        
        
        total_loss += loss.item()
        n_batches += 1
        
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/n_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return total_loss / n_batches

def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for input_seqs, targets, attention_mask in dataloader:
            input_seqs = input_seqs.to(device)
            targets = targets.to(device)
            attention_mask = attention_mask.to(device)
            
            predictions = model(input_seqs, attention_mask)
            loss = F.mse_loss(predictions, targets)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Also save as "latest"
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

def train_transformer(
    train_sequences,
    val_sequences=None,
    n_epochs=10,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0.01,
    checkpoint_dir='checkpoints',
    save_every=1000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    
    
    print("="*70)
    print("TRAINING TRANSFORMER MODEL")
    print("="*70)
    print(f"Device: {device}")
    print(f"Training sequences: {train_sequences.shape[0]:,}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoints every {save_every} steps")
    print("="*70)
    
    
    train_dataset = SequenceDataset(train_sequences)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  
        pin_memory=True if device == 'cuda' else False
    )
    
    if val_sequences is not None:
        val_dataset = SequenceDataset(val_sequences)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
    else:
        val_loader = None
    
    model = TransformerPredictor(
        input_dim=5,
        hidden_dim=72,
        n_layers=3,
        n_heads=6,
        head_dim=12,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999) 
    )
    
    
    total_steps = len(train_loader) * n_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    
    print("\nStarting training...")
    print("="*70)
    
    global_step = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'steps': [],
        'epochs': []
    }
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        training_history['train_loss'].append(train_loss)
        training_history['epochs'].append(epoch)
        
        
        if val_loader is not None:
            val_loss = validate(model, val_loader, device)
            training_history['val_loss'].append(val_loss)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
       
        global_step += len(train_loader)
        training_history['steps'].append(global_step)
        
        
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, 
                          train_loss, checkpoint_dir)
        
        print("-"*70)
    
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    final_path = save_checkpoint(model, optimizer, scheduler, n_epochs, 
                                 global_step, train_loss, checkpoint_dir)
    
    
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"✓ Saved training history: {history_path}")
    
    
    plot_training_curves(training_history, checkpoint_dir)
    
    print("="*70)
    
    return model, training_history

def plot_training_curves(history, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = history['epochs']
    ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    
    if history['val_loss']:
        ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
   
    train_sequences = np.load('train_sequences.npy')
    
    
    n_val = int(0.05 * len(train_sequences))
    val_sequences = train_sequences[-n_val:]
    train_sequences = train_sequences[:-n_val]
    
    
   
    model, history = train_transformer(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        n_epochs=10,              
        batch_size=256,           
        learning_rate=1e-3,       
        weight_decay=0.01,       
        checkpoint_dir='checkpoints',
        save_every=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n✓ Training complete! Model checkpoints saved to 'checkpoints/' directory")
    print("✓ You can now proceed to evaluation (Section 4.3)")
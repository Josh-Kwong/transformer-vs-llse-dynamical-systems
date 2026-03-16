"""
evaluate_transformer.py

Evaluate trained transformer model on test sequences (Section 4.3).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Import from training script
from train_transformer import TransformerPredictor, SequenceDataset, collate_fn



def llse_predict_sequence(sequence):
    """
    LLSE predictor from equation (2) in the problem.
    
    Given sequence [x0, x1, ..., x_i], predict x_{i+1}.
    
    Returns:
        predictions: array of shape (T-1,) containing predictions for x1, ..., x_{T-1}
        errors: squared errors for each prediction
    """
    T, d = sequence.shape
    predictions = []
    errors = []
    
    for i in range(T - 1):
        if i == 0:
            # Predict x1 from x0: no history, so predict 0
            pred = np.zeros(d)
        else:
            # Build matrices for LLSE predictor
            # X_future = [x1, x2, ..., x_i]^T  (i × d)
            # X_past = [x0, x1, ..., x_{i-1}]^T  (i × d)
            X_future = sequence[1:i+1]  # shape (i, d)
            X_past = sequence[0:i]      # shape (i, d)
            
            # LLSE predictor: x̂_{i+1} = X_future^T @ (X_past^T)^† @ x_i
            # Equivalently: x̂_{i+1} = X_future^T @ pinv(X_past) @ x_i
            try:
                pred = X_future.T @ np.linalg.pinv(X_past.T) @ sequence[i]
            except:
                pred = np.zeros(d)
        
        predictions.append(pred)
        error = np.sum((sequence[i + 1] - pred) ** 2)
        errors.append(error)
    
    return np.array(predictions), np.array(errors)

def evaluate_llse_on_sequences(sequences):
    """
    Evaluate LLSE predictor on multiple sequences.
    
    Returns:
        errors_by_timestep: shape (n_sequences, T-1)
    """
    n_sequences = len(sequences)
    T = sequences.shape[1]
    
    errors_by_timestep = np.zeros((n_sequences, T - 1))
    
    print("Evaluating LLSE predictor...")
    for i in tqdm(range(n_sequences)):
        _, errors = llse_predict_sequence(sequences[i])
        errors_by_timestep[i] = errors
    
    return errors_by_timestep



def evaluate_transformer_on_sequences(model, sequences, device='cuda', batch_size=256):
    """
    Evaluate transformer model on sequences.
    
    Returns:
        errors_by_timestep: shape (n_sequences, T-1)
    """
    model.eval()
    
    
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    n_sequences = sequences.shape[0]
    T = sequences.shape[1]
    
    errors_by_timestep = np.zeros((n_sequences, T - 1))
    
    print("Evaluating transformer")
    with torch.no_grad():
        for batch_idx, (input_seqs, targets, attention_mask) in enumerate(tqdm(dataloader)):
            input_seqs = input_seqs.to(device)
            targets = targets.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get predictions
            predictions = model(input_seqs, attention_mask)
            
            
            squared_errors = torch.sum((predictions - targets) ** 2, dim=1).cpu().numpy()
            
            
            for i, error in enumerate(squared_errors):
                global_idx = batch_idx * batch_size + i
                seq_idx = global_idx // (T - 1)
                time_idx = global_idx % (T - 1)
                
                if seq_idx < n_sequences:
                    errors_by_timestep[seq_idx, time_idx] = error
    
    return errors_by_timestep

def load_model_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = TransformerPredictor(
        input_dim=5,
        hidden_dim=72,
        n_layers=3,
        n_heads=6,
        head_dim=12,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    
    return model, checkpoint



def plot_comparison(transformer_errors, llse_errors, title, save_path=None):
    
    T = transformer_errors.shape[1]
    time_steps = np.arange(1, T + 1)
    
    
    trans_mean = transformer_errors.mean(axis=0)
    trans_std = transformer_errors.std(axis=0)
    
    
    llse_mean = llse_errors.mean(axis=0)
    llse_std = llse_errors.std(axis=0)
    
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    
    ax.errorbar(time_steps, trans_mean, yerr=trans_std, 
                label='Transformer', marker='o', capsize=3, alpha=0.8)
    
    
    ax.errorbar(time_steps, llse_mean, yerr=llse_std,
                label='LLSE Predictor', marker='s', capsize=3, alpha=0.8)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")
    
    plt.show()

def plot_training_progress(checkpoint_dir, test_sequences, device='cuda'):
    """
    Plot how test error changes during training (Section 4.3, question 3).
    """
    
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith('checkpoint_step_') and fname.endswith('.pt'):
            step = int(fname.split('_')[2].split('.')[0])
            checkpoints.append((step, os.path.join(checkpoint_dir, fname)))
    
    checkpoints.sort()
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    
    steps = []
    avg_errors = []
    
    for step, checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        model, _ = load_model_checkpoint(checkpoint_path, device)
        
        
        errors = evaluate_transformer_on_sequences(model, test_sequences, device)
        
        
        avg_error = errors.mean()
        
        steps.append(step)
        avg_errors.append(avg_error)
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, avg_errors, marker='o', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Test MSE', fontsize=12)
    ax.set_title('Test Error vs Training Progress', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(checkpoint_dir, 'test_error_vs_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {save_path}")
    plt.show()



def evaluate_single_system(checkpoint_path, test_system_idx=0, device='cuda'):
    """
    Section 4.3, Questions 1-2:
    Evaluate on one test system's 1000 sequences.
    """
    print("\n" + "="*70)
    print("SECTION 4.3, QUESTIONS 1-2: Single System Evaluation")
    print("="*70)
    

    print("Loading test data...")
    test_sequences_by_system = np.load('test_sequences_by_system.npy')
    test_system_sequences = test_sequences_by_system[test_system_idx]
    print(f"✓ Using test system #{test_system_idx}")
    print(f"  {test_system_sequences.shape[0]} sequences")
    
    
    model, _ = load_model_checkpoint(checkpoint_path, device)
    
    
    trans_errors = evaluate_transformer_on_sequences(
        model, test_system_sequences, device
    )
    
    
    llse_errors = evaluate_llse_on_sequences(test_system_sequences)
    
    
    plot_comparison(
        trans_errors, llse_errors,
        f'Transformer vs LLSE Predictor (Test System #{test_system_idx})',
        save_path=f'evaluation_system_{test_system_idx}.png'
    )
    
    
    print("\n" + "-"*70)
    print("RESULTS SUMMARY")
    print("-"*70)
    print(f"Transformer - Avg Error: {trans_errors.mean():.6f} ± {trans_errors.std():.6f}")
    print(f"LLSE        - Avg Error: {llse_errors.mean():.6f} ± {llse_errors.std():.6f}")
    
    if trans_errors.mean() < llse_errors.mean():
        print("\n✓ Transformer outperforms LLSE!")
    else:
        print("\n⚠ LLSE outperforms Transformer (may need more training)")
    
    print("="*70)
    
    return trans_errors, llse_errors

def evaluate_all_checkpoints(checkpoint_dir, device='cuda'):
    """
    Section 4.3, Question 3:
    Evaluate all checkpoints on full test set.
    """
    print("\n" + "="*70)
    print("SECTION 4.3, QUESTION 3: Training Progress Evaluation")
    print("="*70)
    
    
    print("Loading test data...")
    test_sequences_flat = np.load('test_sequences_flat.npy')
    print(f"✓ Loaded {len(test_sequences_flat):,} test sequences")
    
    
    plot_training_progress(checkpoint_dir, test_sequences_flat, device)
    
    print("="*70)



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    
    trans_errors, llse_errors = evaluate_single_system(
        checkpoint_path='checkpoints/checkpoint_latest.pt',
        test_system_idx=0,
        device=device
    )
    
    
    evaluate_all_checkpoints('checkpoints', device=device)
    
    print("\n✓ Evaluation complete!")
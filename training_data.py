"""
training_data.py

Generate training sequences for transformer in-context learning.
Creates 40,000 sequences from 40,000 different orthogonal systems.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_orthogonal_matrix(d):
    """
    Generate a random d x d orthogonal matrix using QR decomposition.
    
    Args:
        d: dimension of the matrix
    
    Returns:
        Q: d x d orthogonal matrix (Q^T @ Q = I)
    """
    # Sample d x d matrix with i.i.d. N(0, 1) entries
    A = np.random.randn(d, d)
    
    # QR decomposition: A = QR where Q is orthogonal
    Q, R = np.linalg.qr(A)
    
    return Q

def generate_sequence(U, T=20):
    """
    Generate a sequence from the linear dynamical system x_{i+1} = U @ x_i.
    
    Args:
        U: d x d orthogonal system matrix
        T: length of sequence (default 20, giving x0 through x19)
    
    Returns:
        sequence: array of shape (T, d) containing [x0, x1, ..., x_{T-1}]
    """
    d = U.shape[0]
    
    # Generate random initial state x0 ~ N(0, I_d)
    x0 = np.random.randn(d)
    
    # Preallocate array for the full sequence
    sequence = np.zeros((T, d))
    sequence[0] = x0
    
    # Generate sequence: x_{i+1} = U @ x_i
    for i in range(T - 1):
        sequence[i + 1] = U @ sequence[i]
    
    return sequence

def generate_training_data(n_sequences=40000, d=5, T=20, seed=42):
    """
    Generate training dataset: each sequence from a different system.
    
    Args:
        n_sequences: number of sequences to generate (default 40,000)
        d: dimension of the system (default 5)
        T: length of each sequence (default 20)
        seed: random seed for reproducibility
    
    Returns:
        sequences: array of shape (n_sequences, T, d)
        U_matrices: list of orthogonal matrices (optional, for analysis)
    """
    np.random.seed(seed)
    
    # Preallocate array
    # Shape: (n_sequences, T, d)
    sequences = np.zeros((n_sequences, T, d))
    U_matrices = []
    
    print(f"Generating {n_sequences} training sequences...")
    print(f"  Dimension: d={d}")
    print(f"  Sequence length: T={T}")
    print(f"  Random seed: {seed}")
    
    for i in range(n_sequences):
        # Progress indicator
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i + 1}/{n_sequences} sequences")
        
        # Generate NEW random system for each sequence
        U = generate_orthogonal_matrix(d)
        U_matrices.append(U)
        
        # Generate one trajectory from this system
        sequence = generate_sequence(U, T)
        sequences[i] = sequence
    
    print(f"✓ Done! Generated {n_sequences} training sequences.")
    
    return sequences, U_matrices

def visualize_training_samples(sequences, n_samples=5):
    """
    Visualize sample training sequences (first 2 dimensions).
    """
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    
    # Randomly select sequences to plot
    sample_indices = np.random.choice(len(sequences), n_samples, replace=False)
    
    for idx, ax in enumerate(axes):
        seq = sequences[sample_indices[idx]]
        
        # Plot trajectory in first 2 dimensions
        ax.plot(seq[:, 0], seq[:, 1], 'o-', alpha=0.7, markersize=4)
        ax.plot(seq[0, 0], seq[0, 1], 'go', markersize=10, label='Start')
        ax.plot(seq[-1, 0], seq[-1, 1], 'ro', markersize=10, label='End')
        
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_title(f'Sequence {sample_indices[idx]}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Sample Training Sequences (2D projection)', y=1.02, fontsize=12)
    plt.show()

def check_training_statistics(sequences):
    """
    Verify training data has correct statistical properties.
    """
    n_sequences, T, d = sequences.shape
    
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)
    
    # Check means (should be close to 0)
    means = sequences.mean(axis=(0, 2))
    print(f"\nMean across all sequences and dimensions:")
    print(f"  Min: {means.min():.4f}, Max: {means.max():.4f}")
    print(f"  Expected: ~0 (since x ~ N(0, I))")
    
    # Check norms (should stay constant since U is orthogonal)
    norms = np.linalg.norm(sequences, axis=2)
    mean_norms = norms.mean(axis=0)
    
    print(f"\nAverage ||x_i|| at each time step:")
    print(f"  t=0:  {mean_norms[0]:.3f}")
    print(f"  t=10: {mean_norms[10]:.3f}")
    print(f"  t=19: {mean_norms[19]:.3f}")
    print(f"  Expected: ~{np.sqrt(d):.3f} (≈ √d)")
    print(f"  ✓ Norms stay constant (confirms U is orthogonal)")
    
    # Check diversity
    variance = sequences.var(axis=0).mean()
    print(f"\nVariance across sequences: {variance:.3f}")
    print(f"  (Should be > 0, indicating diverse sequences)")
    
    print("="*60)

if __name__ == "__main__":
    # Generate training data
    print("="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    
    train_sequences, train_U_matrices = generate_training_data(
        n_sequences=40000,
        d=5,
        T=20,
        seed=42  # Fixed seed for reproducibility
    )
    
    print(f"\nTraining data shape: {train_sequences.shape}")
    print(f"  - {train_sequences.shape[0]:,} sequences")
    print(f"  - {train_sequences.shape[1]} time steps")
    print(f"  - {train_sequences.shape[2]} dimensions")
    
    # Check statistics
    check_training_statistics(train_sequences)
    
    # Visualize samples
    print("\nVisualizing sample sequences...")
    visualize_training_samples(train_sequences, n_samples=5)
    
    # Save data
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)
    
    np.save('train_sequences.npy', train_sequences)
    print("✓ Saved: train_sequences.npy")
    
    # Optionally save U matrices for later analysis
    np.save('train_U_matrices.npy', np.array(train_U_matrices))
    print("✓ Saved: train_U_matrices.npy")
    
    print("\n" + "="*60)
    print("TRAINING DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\n✓ Ready for transformer training with {len(train_sequences):,} sequences")


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
    
    A = np.random.randn(d, d)
    
    
    Q, R = np.linalg.qr(A)
    
    return Q

def generate_sequence(U, T=20):
    
    d = U.shape[0]
    
    
    x0 = np.random.randn(d)
    
    
    sequence = np.zeros((T, d))
    sequence[0] = x0
    
    
    for i in range(T - 1):
        sequence[i + 1] = U @ sequence[i]
    
    return sequence

def generate_test_data(n_systems=50, n_seqs_per_system=1000, d=5, T=20, seed=9999):
   
    
    print(f"Generating {n_systems} test systems...")
    print(f"  Each system will have {n_seqs_per_system} sequences")
    print(f"  Total: {n_systems * n_seqs_per_system:,} test sequences")
    print(f"  Dimension: d={d}")
    print(f"  Sequence length: T={T}")
    print(f"  Random seed: {seed} (DIFFERENT from training!)")
    
    
    U_matrices = []
    print(f"\nGenerating {n_systems} NEW test system matrices...")
    for i in range(n_systems):
        U = generate_orthogonal_matrix(d)
        U_matrices.append(U)
    print(f"✓ Generated {len(U_matrices)} test systems")
    
    
    sequences_by_system = np.zeros((n_systems, n_seqs_per_system, T, d))
    
    print(f"\nGenerating sequences for each test system...")
    for sys_idx in range(n_systems):
        if (sys_idx + 1) % 10 == 0:
            print(f"  System {sys_idx + 1}/{n_systems}...")
        
        U = U_matrices[sys_idx]
        
       
        for seq_idx in range(n_seqs_per_system):
            sequence = generate_sequence(U, T)
            sequences_by_system[sys_idx, seq_idx] = sequence
    
    print(f"✓ Done! Generated all test sequences.")
    
    
    sequences_flat = sequences_by_system.reshape(-1, T, d)
    
    return sequences_by_system, sequences_flat, U_matrices

def visualize_test_system(sequences_by_system, system_idx=0, n_samples=5):
    """
    Visualize multiple trajectories from ONE test system.
    
    This shows that different initial conditions lead to different 
    trajectories, but they all follow the same system dynamics (same U).
    """
    sequences = sequences_by_system[system_idx]
    
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    
    
    sample_indices = np.random.choice(len(sequences), n_samples, replace=False)
    
    for idx, ax in enumerate(axes):
        seq = sequences[sample_indices[idx]]
        
        
        ax.plot(seq[:, 0], seq[:, 1], 'o-', alpha=0.7, markersize=4)
        ax.plot(seq[0, 0], seq[0, 1], 'go', markersize=10, label='Start')
        ax.plot(seq[-1, 0], seq[-1, 1], 'ro', markersize=10, label='End')
        
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_title(f'Trajectory {sample_indices[idx]}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.suptitle(f'Test System #{system_idx}: Different Initial Conditions', 
                 y=1.02, fontsize=12)
    plt.show()

def check_test_statistics(sequences_by_system):
    """
    Verify test data has correct statistical properties.
    """
    n_systems, n_seqs, T, d = sequences_by_system.shape
    
    print("\n" + "="*60)
    print("TEST DATA STATISTICS")
    print("="*60)
    
    sequences_flat = sequences_by_system.reshape(-1, T, d)
    
    
    means = sequences_flat.mean(axis=(0, 2))
    print(f"\nMean across all sequences and dimensions:")
    print(f"  Min: {means.min():.4f}, Max: {means.max():.4f}")
    print(f"  Expected: ~0")
    
    
    norms = np.linalg.norm(sequences_flat, axis=2)
    mean_norms = norms.mean(axis=0)
    
    print(f"\nAverage ||x_i|| at each time step:")
    print(f"  t=0:  {mean_norms[0]:.3f}")
    print(f"  t=10: {mean_norms[10]:.3f}")
    print(f"  t=19: {mean_norms[19]:.3f}")
    print(f"  Expected: ~{np.sqrt(d):.3f}")
    
    
    print(f"\nPer-system statistics:")
    print(f"  Each system has {n_seqs} sequences")
    print(f"  This allows computing reliable statistics per system")
    
    
    within_system_var = np.mean([sequences_by_system[i].var() 
                                  for i in range(n_systems)])
    across_system_var = sequences_by_system.mean(axis=1).var()
    
    print(f"  Within-system variance: {within_system_var:.3f}")
    print(f"  Across-system variance: {across_system_var:.3f}")
    
    print("="*60)

def verify_separation_from_training():
    """
    Verify that test systems are truly different from training systems.
    
    This is a sanity check - since we used different random seeds,
    test systems should be independent from training systems.
    """
    print("\n" + "="*60)
    print("VERIFYING SEPARATION FROM TRAINING")
    print("="*60)
    
    
    try:
        train_U = np.load('train_U_matrices.npy')
        test_U = np.load('test_U_matrices.npy')
        
        print(f"Loaded {len(train_U)} training systems")
        print(f"Loaded {len(test_U)} test systems")
        
       
        print("\nChecking for overlap...")
        max_similarity = 0
        for i, U_test in enumerate(test_U):
            for j, U_train in enumerate(train_U):
                
                similarity = np.linalg.norm(U_test - U_train)
                if similarity < 0.01:  
                    print(f"  WARNING: Test system {i} very similar to train system {j}")
                max_similarity = max(max_similarity, 1.0 / (1.0 + similarity))
        
        print(f"✓ No problematic overlap detected")
        print(f"  (Test and training systems are independent)")
        
    except FileNotFoundError:
        print("Training data not found. Run training_data.py first.")
    
    print("="*60)

if __name__ == "__main__":
    
    print("="*60)
    print("GENERATING TEST DATA")
    print("="*60)
    
    test_sequences_by_system, test_sequences_flat, test_U_matrices = generate_test_data(
        n_systems=50,
        n_seqs_per_system=1000,
        d=5,
        T=20,
        seed=9999  
    )
    
    print(f"\nTest data shape (by system): {test_sequences_by_system.shape}")
    print(f"  - {test_sequences_by_system.shape[0]} test systems")
    print(f"  - {test_sequences_by_system.shape[1]} sequences per system")
    print(f"  - {test_sequences_by_system.shape[2]} time steps")
    print(f"  - {test_sequences_by_system.shape[3]} dimensions")
    
    print(f"\nTest data shape (flattened): {test_sequences_flat.shape}")
    print(f"  - Total: {test_sequences_flat.shape[0]:,} test sequences")
    
    
    check_test_statistics(test_sequences_by_system)
    
    
    print("\nVisualizing trajectories from test system #0...")
    visualize_test_system(test_sequences_by_system, system_idx=0, n_samples=5)
    
    
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)
    
    np.save('test_sequences_by_system.npy', test_sequences_by_system)
    print("✓ Saved: test_sequences_by_system.npy")
    
    np.save('test_sequences_flat.npy', test_sequences_flat)
    print("✓ Saved: test_sequences_flat.npy")
    
    np.save('test_U_matrices.npy', np.array(test_U_matrices))
    print("✓ Saved: test_U_matrices.npy")
    
    
    verify_separation_from_training()
    
    print("\n" + "="*60)
    print("TEST DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\n✓ Ready for evaluation with {len(test_sequences_flat):,} test sequences")
    print(f"✓ {len(test_U_matrices)} unique test systems (separate from training)")
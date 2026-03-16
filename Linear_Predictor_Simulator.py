import numpy as np
import matplotlib.pyplot as plt

def generate_orthogonal_matrix(d):
    """Generate a random d x d orthogonal matrix using QR decomposition"""
    # Generate random matrix
    A = np.random.randn(d, d)
    # QR decomposition gives us an orthogonal Q
    Q, R = np.linalg.qr(A)
    return Q

def simulate_system(U, x0, T):
    """
    Simulate the linear dynamical system for T time steps
    
    Args:
        U: d x d orthogonal matrix
        x0: d-dimensional initial state
        T: number of time steps
    
    Returns:
        states: array of shape (T+1, d) containing [x0, x1, ..., xT]
    """
    d = len(x0)
    states = np.zeros((T + 1, d))
    states[0] = x0
    
    for i in range(T):
        states[i + 1] = U @ states[i]
    
    return states

def llse_predict(states, i):
    """
    LLSE predictor: x̂_{i+1} = X_out @ X_in† @ x_i
    
    Args:
        states: array containing x0, x1, ..., x_i
        i: current time step
    
    Returns:
        x̂_{i+1}: predicted next state
    """
    if i == 0:
        # Can't predict at i=0 (no history)
        return None
    
    # X_in = [x0, x1, ..., x_{i-1}]  (d x i matrix)
    X_in = states[:i].T  # shape: (d, i)
    
    # X_out = [x1, x2, ..., x_i]  (d x i matrix)
    X_out = states[1:i+1].T  # shape: (d, i)
    
    # Current state
    x_i = states[i]
    
    # Compute pseudoinverse of X_in
    X_in_pinv = np.linalg.pinv(X_in)
    
    # Prediction: x̂_{i+1} = X_out @ X_in† @ x_i
    x_hat = X_out @ X_in_pinv @ x_i
    
    return x_hat

def compute_squared_error(x_true, x_pred):
    """Compute ||x_true - x_pred||²"""
    if x_pred is None:
        return np.nan
    return np.sum((x_true - x_pred) ** 2)

def run_simulation(d, n_trajectories=1000, T=20):
    """
    Run full simulation for given dimension d
    
    Args:
        d: dimension of the system
        n_trajectories: number of initial states to try
        T: number of time steps
    
    Returns:
        avg_errors: average squared error at each time step
    """
    # Generate one orthogonal matrix for this dimension
    U = generate_orthogonal_matrix(d)
    
    # Store errors for all trajectories
    all_errors = np.zeros((n_trajectories, T + 1))
    
    for traj in range(n_trajectories):
        # Generate random initial state x0 ~ N(0, I)
        x0 = np.random.randn(d)
        
        # Simulate the system
        states = simulate_system(U, x0, T)
        
        # Compute prediction error at each time step
        for i in range(T):
            # Predict x_{i+1}
            x_pred = llse_predict(states, i)
            
            # True x_{i+1}
            x_true = states[i + 1]
            
            # Compute squared error
            error = compute_squared_error(x_true, x_pred)
            all_errors[traj, i + 1] = error
    
    # Average over all trajectories
    avg_errors = np.nanmean(all_errors, axis=0)
    
    return avg_errors

# Run simulations for d = 2, 5, 10
dimensions = [2, 5, 10]
T = 20
n_trajectories = 1000

results = {}
for d in dimensions:
    print(f"Running simulation for d={d}...")
    avg_errors = run_simulation(d, n_trajectories, T)
    results[d] = avg_errors

# Plot results
plt.figure(figsize=(10, 6))

for d in dimensions:
    time_steps = np.arange(T + 1)
    plt.plot(time_steps, results[d], marker='o', label=f'd={d}', linewidth=2)

# Add theoretical prediction: error = max(0, d - i)
for d in dimensions:
    theoretical = np.maximum(0, d - np.arange(T + 1))
    plt.plot(np.arange(T + 1), theoretical, '--', alpha=0.5, 
             label=f'd={d} (theory)')

plt.xlabel('Time Step i', fontsize=12)
plt.ylabel('Average Squared Error', fontsize=12)
plt.title('LLSE Predictor: Average Squared Error vs Time Step', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary
print("\n=== Summary ===")
for d in dimensions:
    print(f"\nd={d}:")
    print(f"  Error at i=1: {results[d][1]:.3f} (theory: {d-1})")
    print(f"  Error at i=d: {results[d][d]:.3f} (theory: 0)")
    print(f"  Error at i={T}: {results[d][T]:.3f} (theory: 0)")
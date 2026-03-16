# In-Context Learning for Linear Dynamical Systems

A transformer trained to predict the next state of an unknown linear dynamical system, using only the observed sequence as context. The model outperforms the mathematically optimal linear least squares estimator (LLSE) in the low data context through training.
<img width="1485" height="824" alt="image" src="https://github.com/user-attachments/assets/f1c2b9b5-ee57-4d95-9511-d50990637763" />
## The Problem

Given a linear dynamical system x_{i+1} = U * x_i, where U is an unknown 5×5 orthogonal matrix and x_0 is a random initial state, the goal is to predict the next state x_{i+1} from the observed history [x_0, x_1, ..., x_i]. U is never revealed and the model has to figure out the dynamics purely from the sequence.
## Why the Transformer Wins
The classical approach (LLSE via pseudoinverse) is optimal given only the current sequence, but it has no memory of anything outside that sequence. With fewer than d=5 observations, the system is underdetermined and the LLSE error is high.
The transformer, trained on 40,000 sequences from different orthogonal systems, has learned a prior over what these systems look like. When it sees a new sequence with only 1 or 2 states, it doesn't start from scratch — it draws on patterns from training to make better early predictions. This is visible in the graph: the transformer (blue) consistently has lower error than the LLSE (orange) in the first few time steps. Both converge to zero once enough context is available, since at that point the pseudoinverse can fully solve for U.
This is effectively Bayesian inference — the transformer has an implicit prior from training, while the LLSE has no prior at all.

## Project Structure
train_transformer.py — Defines and trains the transformer model. Architecture: 3 encoder layers, hidden dimension 72, 6 attention heads. Each training sequence of 20 time steps generates 19 variable-length examples (predict x_1 from [x_0], predict x_2 from [x_0, x_1], etc.), trained with MSE loss using AdamW and cosine learning rate scheduling.
test_sequences.py — Generates 50 held-out test systems (1,000 trajectories each) using a separate random seed to ensure complete separation from training data.
evaluate_transformer.py — Evaluates the trained transformer against the LLSE baseline on test sequences and produces the comparison plot. The LLSE predictor computes x̂_{i+1} = X_out @ pinv(X_in) @ x_i at each time step.
llse_simulation.py — Standalone simulation of the LLSE predictor across dimensions d=2, 5, 10, verifying that empirical error matches the theoretical prediction E[error] = max(0, d − i).

## Setup and Usage
python 3.8+ with pytorch and numpy is required
files:
python train_transformer.py #trains the model

python test_sequences.py #generate test sequences

python evaluate_transformer.py #evaluate model

Training will take several hours varying based on your cpu.


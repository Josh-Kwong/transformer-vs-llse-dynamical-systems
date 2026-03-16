"""
test_train_transformer.py

Tests for the transformer training pipeline.
Run with: python test_train_transformer.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from train_transformer import (
    SequenceDataset,
    collate_fn,
    PositionalEncoding,
    TransformerPredictor,
    train_epoch,
    validate,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW


def make_dummy_sequences(n=10, T=20, d=5):
    return np.random.randn(n, T, d).astype(np.float32)




def test_dataset_length():
    sequences = make_dummy_sequences(n=10, T=20, d=5)
    dataset = SequenceDataset(sequences)
    # 10 sequences * 19 examples each
    assert len(dataset) == 10 * 19, f"Expected 190, got {len(dataset)}"
    print("test_dataset_length passed")


def test_dataset_getitem_shapes():
    sequences = make_dummy_sequences(n=5, T=20, d=5)
    dataset = SequenceDataset(sequences)

    for idx in [0, 1, 18, 19, 50]:
        input_seq, target, seq_len = dataset[idx]
        assert target.shape == (5,)
        assert input_seq.shape == (seq_len, 5)
        assert 1 <= seq_len <= 19

    print("test_dataset_getitem_shapes passed")


def test_dataset_indexing_logic():
    """Make sure the flat idx correctly maps back to (sequence, timestep)."""
    sequences = make_dummy_sequences(n=3, T=5, d=2)
    dataset = SequenceDataset(sequences)

    # T=5 means 4 examples per sequence
    # idx 0 -> seq 0, time 0 (input=[x0], target=x1)
    # idx 3 -> seq 0, time 3 (input=[x0..x3], target=x4)
    # idx 4 -> seq 1, time 0 (wraps to next sequence)

    input_seq, target, seq_len = dataset[0]
    assert seq_len == 1
    assert torch.allclose(input_seq[0], dataset.sequences[0, 0])
    assert torch.allclose(target, dataset.sequences[0, 1])

    input_seq, target, seq_len = dataset[3]
    assert seq_len == 4
    assert torch.allclose(target, dataset.sequences[0, 4])

    # should wrap to sequence 1
    input_seq, target, seq_len = dataset[4]
    assert seq_len == 1
    assert torch.allclose(input_seq[0], dataset.sequences[1, 0])
    assert torch.allclose(target, dataset.sequences[1, 1])

    print("test_dataset_indexing_logic passed")




def test_collate_fn_shapes():
    sequences = make_dummy_sequences(n=20, T=20, d=5)
    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    padded_inputs, targets, attention_mask = next(iter(loader))

    assert padded_inputs.dim() == 3
    assert padded_inputs.shape[0] == 8
    assert padded_inputs.shape[2] == 5
    assert targets.shape == (8, 5)
    assert attention_mask.shape == padded_inputs.shape[:2]

    print("test_collate_fn_shapes passed")


def test_collate_fn_masking():
    """Check that padded positions are zeroed out and masked properly."""
    sequences = make_dummy_sequences(n=20, T=10, d=3)
    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    padded_inputs, targets, attention_mask = next(iter(loader))

    for i in range(padded_inputs.shape[0]):
        real_len = int(attention_mask[i].sum().item())
        if real_len < padded_inputs.shape[1]:
            padding = padded_inputs[i, real_len:]
            assert torch.all(padding == 0), "Padding should be zeros"
            assert torch.all(~attention_mask[i, real_len:])

    print("test_collate_fn_masking passed")




def test_model_output_shape():
    model = TransformerPredictor(input_dim=5, hidden_dim=72, n_layers=3, n_heads=6)
    x = torch.randn(4, 8, 5)
    mask = torch.ones(4, 8, dtype=torch.bool)

    output = model(x, mask)
    assert output.shape == (4, 5), f"Expected (4, 5), got {output.shape}"
    print("test_model_output_shape passed")


def test_model_variable_lengths():
    """Feed sequences of different lengths in one batch, make sure nothing breaks."""
    model = TransformerPredictor(input_dim=5, hidden_dim=72, n_layers=3, n_heads=6)

    x = torch.randn(4, 10, 5)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :7] = True
    mask[2, :1] = True   # just one token
    mask[3, :10] = True  # full length

    output = model(x, mask)
    assert output.shape == (4, 5)
    assert not torch.isnan(output).any(), "Got NaN in output"
    print("test_model_variable_lengths passed")


def test_model_no_mask():
    model = TransformerPredictor(input_dim=5, hidden_dim=72, n_layers=3, n_heads=6)
    x = torch.randn(4, 8, 5)

    output = model(x, attention_mask=None)
    assert output.shape == (4, 5)
    assert not torch.isnan(output).any()
    print("test_model_no_mask passed")


def test_param_count():
    model = TransformerPredictor(input_dim=5, hidden_dim=72, n_layers=3, n_heads=6)
    n_params = sum(p.numel() for p in model.parameters())
    # small model, should be somewhere in the tens of thousands
    assert 10_000 < n_params < 500_000, f"Unexpected: {n_params}"
    print(f"test_param_count passed ({n_params:,} params)")


def test_positional_encoding():
    pe = PositionalEncoding(d_model=72, max_len=20)
    x = torch.zeros(1, 10, 72)
    out = pe(x)

    # adjacent positions should get different encodings
    for i in range(9):
        assert not torch.allclose(out[0, i], out[0, i + 1])

    print("test_positional_encoding passed")


def test_gradient_flow():
    """Every parameter in the model should get a nonzero gradient."""
    model = TransformerPredictor(input_dim=5, hidden_dim=72, n_layers=3, n_heads=6)

    x = torch.randn(4, 6, 5)
    mask = torch.ones(4, 6, dtype=torch.bool)
    target = torch.randn(4, 5)

    pred = model(x, mask)
    loss = F.mse_loss(pred, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    print("test_gradient_flow passed")




def test_loss_decreases():
    """Train for a handful of steps and make sure loss goes down."""
    sequences = make_dummy_sequences(n=50, T=20, d=5)
    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = TransformerPredictor(input_dim=5, hidden_dim=72, n_layers=3, n_heads=6)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # grab initial loss before any training
    model.eval()
    with torch.no_grad():
        inputs, targets, mask = next(iter(loader))
        initial_loss = F.mse_loss(model(inputs, mask), targets).item()

    model.train()
    for _ in range(20):
        inputs, targets, mask = next(iter(loader))
        loss = F.mse_loss(model(inputs, mask), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"test_loss_decreases passed ({initial_loss:.4f} -> {final_loss:.4f})")


def test_train_epoch():
    sequences = make_dummy_sequences(n=30, T=10, d=5)
    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = TransformerPredictor(input_dim=5, hidden_dim=72, n_layers=3, n_heads=6)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    avg_loss = train_epoch(model, loader, optimizer, scheduler=None, device='cpu', epoch=1)
    assert isinstance(avg_loss, float) and avg_loss > 0 and not np.isnan(avg_loss)
    print(f"test_train_epoch passed (avg loss: {avg_loss:.4f})")


def test_validate():
    sequences = make_dummy_sequences(n=30, T=10, d=5)
    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = TransformerPredictor(input_dim=5, hidden_dim=72, n_layers=3, n_heads=6)
    val_loss = validate(model, loader, device='cpu')

    assert isinstance(val_loss, float) and val_loss > 0 and not np.isnan(val_loss)
    print(f"test_validate passed (val loss: {val_loss:.4f})")




def test_end_to_end():
    """Generate actual orthogonal system data and push it through the full pipeline."""
    d = 5
    T = 10
    n_seq = 20

    sequences = np.zeros((n_seq, T, d), dtype=np.float32)
    for i in range(n_seq):
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        x = np.random.randn(d)
        for t in range(T):
            sequences[i, t] = x
            x = Q @ x

    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    model = TransformerPredictor(input_dim=d, hidden_dim=72, n_layers=3, n_heads=6)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    inputs, targets, mask = next(iter(loader))
    loss = F.mse_loss(model(inputs, mask), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(torch.tensor(loss.item()))
    print(f"test_end_to_end passed (loss: {loss.item():.4f})")


if __name__ == "__main__":
    print("Running tests\n")

    tests = [
        test_dataset_length,
        test_dataset_getitem_shapes,
        test_dataset_indexing_logic,
        test_collate_fn_shapes,
        test_collate_fn_masking,
        test_model_output_shape,
        test_model_variable_lengths,
        test_model_no_mask,
        test_param_count,
        test_positional_encoding,
        test_gradient_flow,
        test_loss_decreases,
        test_train_epoch,
        test_validate,
        test_end_to_end,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
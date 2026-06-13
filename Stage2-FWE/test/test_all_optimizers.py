"""
Test script to verify all optimizers can be imported and run in OpenToMe framework.
"""

import os
import sys
import torch
import torch.nn as nn

# Add opentome to path
sys.path.insert(0, '/path/to/data/Optimizer/OpenToMe_Optimizer/OpenToMe')

from opentome.optimizer import *

# Simple test model
class SimpleModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=4, vocab_size=1000):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def create_named_parameters(model):
    """Get named parameters in format expected by some optimizers."""
    return list(model.named_parameters())


def test_optimizer(optimizer_class, model, test_name, **optimizer_kwargs):
    """Test if an optimizer can be instantiated and perform a step."""
    try:
        # Reset model parameters for clean test
        model.reset_parameters() if hasattr(model, 'reset_parameters') else None

        # Check if optimizer needs named_parameters
        import inspect
        sig = inspect.signature(optimizer_class.__init__)
        first_param = list(sig.parameters.keys())[1]  # Skip 'self'

        if first_param == 'named_parameters':
            optimizer = optimizer_class(create_named_parameters(model), **optimizer_kwargs)
        else:
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        # Create dummy input and target
        batch_size = 4
        seq_len = 32
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        output = model(input_ids)
        loss = nn.functional.cross_entropy(output.view(-1, vocab_size), target.view(-1))

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        print(f"✓ {test_name}: PASSED")
        return True

    except Exception as e:
        print(f"✗ {test_name}: FAILED - {str(e)}")
        return False


def run_all_tests():
    """Run tests for all optimizers."""
    print("=" * 60)
    print("OpenToMe Optimizer Verification Tests")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'skipped': 0}

    # Create model for testing
    model = SimpleModel(hidden_size=128, num_layers=2, vocab_size=256)

    # Test configuration
    base_lr = 1e-3
    base_kwargs = {'lr': base_lr, 'weight_decay': 0.01}

    # List of optimizers to test
    optimizers_to_test = [
        # SGG Optimizers
        ('AdamWSGG', AdamWSGG, {'model': model, **base_kwargs}),
        ('AdafactorSGG', AdafactorSGG, base_kwargs),
        ('LambSGG', LambSGG, {'model': model, **base_kwargs}),
        ('ShampooSGG', ShampooSGG, {'model': model, **base_kwargs}),

        # SAC Optimizers
        ('AdamWSAC', AdamWSAC, {'model': model, **base_kwargs}),
        ('Adam_miniSAC', Adam_miniSAC, {
            'named_parameters': create_named_parameters(model),
            'lr': base_lr,
            'dim': 128,
            'n_heads': 8,
            'weight_decay': 0.01
        }),
        ('ShampooSAC', ShampooSAC, {'model': model, **base_kwargs}),
        ('SACLion', SACLion, {'model': model, **base_kwargs}) if 'SACLion' in globals() else None,

        # Standard Optimizers
        ('Adam_mini', Adam_mini, {
            'named_parameters': create_named_parameters(model),
            'lr': base_lr,
            'dim': 128,
            'n_heads': 8,
            'weight_decay': 0.01
        }),
        ('Lamb', Lamb, base_kwargs),
        ('Shampoo', Shampoo, base_kwargs),
        ('GaLore_AdamW', GaLore_AdamW, {**base_kwargs, 'rank': 64, 'update_proj_gap': 200, 'scale': 1.0}),
        ('GaLoreAdafactor', GaLoreAdafactor, {'lr': base_lr}),
        ('Adan', Adan, base_kwargs),
        ('APOLLO_AdamW', APOLLO_AdamW, {**base_kwargs, 'rank': 64, 'update_proj_gap': 200, 'scale': 1.0}),
        ('CAME', CAME, {'lr': base_lr}),
        ('Conda', Conda, {**base_kwargs, 'rank': 64}),
        ('Lion', Lion, {'lr': 1e-4, 'weight_decay': 0.01}),
        ('MARS', MARS, {'lr': base_lr}),
        ('Muon', Muon, {'lr': base_lr, 'weight_decay': 0.1}),
        ('NAdam', NAdam, base_kwargs),
        ('RAdam', RAdam, base_kwargs),
        ('SophiaG', SophiaG, {'lr': base_lr, 'update_freq': 2}),
        ('SOAP', SOAP, {'lr': base_lr}),
        ('AdaBelief', AdaBelief, {'lr': base_lr}),
        ('AdamP', AdamP, {'lr': base_lr}),
        ('AdamWLegacy', AdamWLegacy, base_kwargs),
        ('Adopt', Adopt, base_kwargs),
        ('Kron', Kron, {'lr': base_lr}),
        ('LaProp', LaProp, {'lr': base_lr}),
        ('LARS', LARS, {'lr': base_lr}),
        ('NvNovoGrad', NvNovoGrad, {'lr': base_lr}),
        ('Prodigy', Prodigy, {'lr': 1.0}),
    ]

    print(f"\nTesting {len(optimizers_to_test)} optimizers...\n")

    for opt_item in optimizers_to_test:
        if opt_item is None:
            continue

        name = opt_item[0]
        opt_class = opt_item[1]
        kwargs = opt_item[2] if len(opt_item) > 2 else base_kwargs

        # Skip optimizers that require special dependencies
        if opt_class is None:
            print(f"⊘ {name}: SKIPPED (not available)")
            results['skipped'] += 1
            continue

        # Create fresh model for each test
        test_model = SimpleModel(hidden_size=128, num_layers=2, vocab_size=256)

        success = test_optimizer(opt_class, test_model, name, **kwargs)
        if success:
            results['passed'] += 1
        else:
            results['failed'] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed:  {results['passed']}")
    print(f"Failed:  {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print("=" * 60)

    return results['failed'] == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

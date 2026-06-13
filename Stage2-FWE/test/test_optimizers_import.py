"""
Unit tests for OpenToMe optimizers.
Tests that all optimizers can be imported and perform a basic step.
"""

import os
import sys
import torch
import torch.nn as nn

# Add opentome to path
sys.path.insert(0, '/path/to/data/Optimizer/OpenToMe_Optimizer/OpenToMe')


class SimpleModel(nn.Module):
    """Simple transformer-like model for testing."""
    def __init__(self, hidden_size=128, num_layers=2, vocab_size=256):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def get_named_parameters(model):
    """Get named parameters."""
    return list(model.named_parameters())


def test_basic_optimizer(opt_class, model, test_name, **kwargs):
    """Test basic optimizer functionality."""
    try:
        # Create fresh model
        test_model = model()

        # Check optimizer signature
        import inspect
        sig = inspect.signature(opt_class.__init__)
        first_param = list(sig.parameters.keys())[1]

        # Create optimizer based on signature
        if first_param == 'named_parameters':
            optimizer = opt_class(
                get_named_parameters(test_model),
                **kwargs
            )
        elif first_param == 'params':
            optimizer = opt_class(test_model.parameters(), **kwargs)
        else:
            optimizer = opt_class(test_model.parameters(), **kwargs)

        # Create dummy data
        batch_size = 2
        seq_len = 16
        vocab_size = 256

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target = torch.randint(0, vocab_size, (batch_size, seq_len, vocab_size))

        # Forward
        output = test_model(input_ids)
        loss = nn.functional.cross_entropy(output.view(-1, vocab_size), target.view(-1, vocab_size))

        # Backward
        loss.backward()

        # Step
        optimizer.step()
        optimizer.zero_grad()

        print(f"✓ {test_name}: PASSED")
        return True

    except Exception as e:
        print(f"✗ {test_name}: FAILED - {str(e)[:100]}")
        return False


def run_import_tests():
    """Test that all optimizers can be imported."""
    print("=" * 60)
    print("OpenToMe Optimizer Import Tests")
    print("=" * 60)

    from opentome import optimizer

    results = {'passed': 0, 'failed': 0}

    # Test all exports
    for name in optimizer.__all__:
        try:
            opt_class = getattr(optimizer, name)
            print(f"✓ {name}: Imported successfully")
            results['passed'] += 1
        except Exception as e:
            print(f"✗ {name}: Import failed - {str(e)[:80]}")
            results['failed'] += 1

    print("")
    print(f"Import Summary: {results['passed']} passed, {results['failed']} failed")
    print("")

    return results['failed'] == 0


def run_step_tests():
    """Test that optimizers can perform a step."""
    print("=" * 60)
    print("OpenToMe Optimizer Step Tests")
    print("=" * 60)

    from opentome import optimizer

    results = {'passed': 0, 'failed': 0, 'skipped': 0}

    # Test configuration
    base_lr = 1e-3
    test_model = SimpleModel

    # Define test cases
    test_cases = [
        # Standard optimizers
        ('AdamWLegacy', {'lr': base_lr}),
        ('Lion', {'lr': 1e-4}),
        ('Adan', {'lr': 3e-3}),
        ('CAME', {'lr': base_lr}),
        ('Lamb', {'lr': base_lr}),
        ('Shampoo', {'lr': base_lr}),
        ('Adabelief', {'lr': base_lr}),
        ('RAdam', {'lr': base_lr}),
        ('NAdam', {'lr': base_lr}),
        ('Adopt', {'lr': base_lr}),
        ('LaProp', {'lr': base_lr}),
        ('LARS', {'lr': base_lr}),
        ('NvNovoGrad', {'lr': base_lr}),
        ('Prodigy', {'lr': 1.0}),

        # GaLore optimizers
        ('GaLore_AdamW', {'lr': base_lr, 'rank': 64, 'update_proj_gap': 200, 'scale': 1.0}),
        ('GaLoreAdafactor', {'lr': base_lr}),

        # APOLLO
        ('APOLLO_AdamW', {'lr': 1e-2, 'rank': 64, 'update_proj_gap': 200, 'scale': 1.0}),

        # SOAP
        ('SOAP', {'lr': base_lr, 'precond_freq': 10, 'max_precond_dim': 8192}),

        # Sophia
        ('SophiaG', {'lr': base_lr, 'update_freq': 2}),

        # MARS
        ('MARS', {'lr': base_lr}),

        # Muon
        ('Muon', {'lr': base_lr, 'weight_decay': 0.1}),

        # Conda
        ('Conda', {'lr': base_lr, 'rank': 64}),

        # Kron
        ('Kron', {'lr': base_lr}),
    ]

    # SAC optimizers (require model argument)
    sac_test_cases = [
        ('AdamWSAC', {'model': SimpleModel(hidden_size=64), 'lr': base_lr}),
        ('ShampooSAC', {'model': SimpleModel(hidden_size=64), 'lr': base_lr}),
        ('SACLion', {'model': SimpleModel(hidden_size=64), 'lr': 1e-4}),
    ]

    # Adam_mini family (require named_parameters)
    adam_mini_test_cases = [
        ('Adam_mini', {'named_parameters': None, 'lr': 5e-4, 'dim': 128, 'n_heads': 8}),
        ('Adam_miniSAC', {'named_parameters': None, 'lr': 5e-4, 'dim': 128, 'n_heads': 8}),
    ]

    print("\nTesting standard optimizers...")
    for name, kwargs in test_cases:
        try:
            opt_class = getattr(optimizer, name)
            if test_basic_optimizer(opt_class, test_model, name, **kwargs):
                results['passed'] += 1
            else:
                results['failed'] += 1
        except AttributeError:
            print(f"⊘ {name}: SKIPPED (not available)")
            results['skipped'] += 1
        except Exception as e:
            print(f"✗ {name}: FAILED - {str(e)[:100]}")
            results['failed'] += 1

    print("\nTesting SAC optimizers...")
    for name, kwargs in sac_test_cases:
        try:
            opt_class = getattr(optimizer, name)
            # Create model for this test
            model = kwargs.pop('model')
            test_m = model()
            if test_basic_optimizer(opt_class, lambda: test_m, name, **kwargs):
                results['passed'] += 1
            else:
                results['failed'] += 1
        except AttributeError:
            print(f"⊘ {name}: SKIPPED (not available)")
            results['skipped'] += 1
        except Exception as e:
            print(f"✗ {name}: FAILED - {str(e)[:100]}")
            results['failed'] += 1

    print("\nTesting Adam_mini family...")
    for name, kwargs in adam_mini_test_cases:
        try:
            opt_class = getattr(optimizer, name)
            model = SimpleModel()
            kwargs['named_parameters'] = get_named_parameters(model)
            if test_basic_optimizer(opt_class, lambda: SimpleModel(), name, **kwargs):
                results['passed'] += 1
            else:
                results['failed'] += 1
        except AttributeError:
            print(f"⊘ {name}: SKIPPED (not available)")
            results['skipped'] += 1
        except Exception as e:
            print(f"✗ {name}: FAILED - {str(e)[:100]}")
            results['failed'] += 1

    print("")
    print("=" * 60)
    print(f"Step Test Summary: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped")
    print("=" * 60)

    return results['failed'] == 0


if __name__ == '__main__':
    import_success = run_import_tests()
    print("")

    if import_success:
        step_success = run_step_tests()
    else:
        print("Skipping step tests due to import failures")
        step_success = False

    sys.exit(0 if (import_success and step_success) else 1)

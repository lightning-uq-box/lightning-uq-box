# Lightning-UQ-Box LLM Instructions

A toolbox for uncertainty quantification in deep learning, built on PyTorch and Lightning.

## Commands

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Run everything through `uv run` so
it uses the locked environment.

```bash
# Install (editable, with the tests, style, and docs extras)
uv sync --all-extras

# Sync into an environment you manage yourself instead of .venv
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --all-extras

# Lint and type check (run from repo root)
uv run ruff format && uv run ruff check && uv run ty check

# Test
uv run pytest tests --cov=lightning_uq_box                                  # all (skips slow)
uv run pytest tests/uq_methods/test_regression.py                          # single file
uv run pytest tests/uq_methods/test_regression.py::TestPosthoc              # single class
uv run pytest -m "" tests                                                   # include slow

# Docs
cd docs && uv run make clean && uv run make html
```

## Dependencies

- `pyproject.toml` is the only place dependencies are declared: runtime under
  `[project] dependencies`, everything else under `[project.optional-dependencies]` as the
  `style`, `tests`, and `docs` extras. There is no `requirements/` directory.
- After editing any dependency, run `uv lock` and commit `uv.lock` in the same change. CI installs
  with `uv sync --locked` and fails if the lockfile is stale.
- Give version floors a comment saying why that minimum is needed, matching the surrounding entries.
- `ruff` is capped below 0.15 because 0.15 expands the default rule set; bumping it is its own change.
- `ty` is pre-1.0. Every release can add new diagnostics, so a dependabot `ty` bump may need code
  fixes rather than a rubber stamp.

## Project Structure

```
lightning_uq_box/
  uq_methods/    # the UQ methods themselves, each a LightningModule
  models/        # architectures and layers (bnn_layers/, bnnlv/, masked_ensemble/, ...)
  datamodules/   # LightningDataModules, mostly toy datasets
  datasets/      # Dataset classes
  eval_utils/    # metric computation and evaluation helpers
  viz_utils/     # plotting
tests/           # mirrors the package layout
docs/            # Sphinx + MyST, tutorials as notebooks
```

## Code Style

### File Header (required)

```python
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
```

### Formatting (Ruff)

Configured in `pyproject.toml`; `ruff format` decides layout, so do not hand-format.

- Double quotes, no magic trailing commas
- `extend-select = ["D", "I", "UP"]`: pydocstyle, isort, and pyupgrade on top of the defaults
- Notebooks are linted too (`extend-include = ["*.ipynb"]`)
- `D` rules are off for `docs/**` and `tests/**`

### Type Hints (ty)

- [ty](https://docs.astral.sh/ty/) replaces mypy. `[tool.ty.src]` checks `lightning_uq_box` and
  `tests`; only `lightning_uq_box/uq_methods` is still excluded, until it is annotated. New code
  elsewhere must be annotated and must pass `uv run ty check`.
- Attributes registered in `__init__` via `register_buffer` / `register_parameter`, or assigned in
  a subclass and read from a base class, need a class-level declaration (`mu_weight: Parameter`).
  Without one, `nn.Module.__getattr__` types them as `Tensor | Module` and every use is an error.
- Union: `X | Y`, not `Union[X, Y]`; prefer built-in `list`/`dict`/`tuple` over `typing.List` etc.
- `# ty: ignore[<rule>]` only for external library issues, with a comment saying which one.

### Docstrings (Google style)

```python
def predict_step(self, X: Tensor, batch_idx: int = 0) -> dict[str, Tensor]:
    """Short description.

    Args:
        X: input tensor of shape [batch_size x input_dim]
        batch_idx: the index of this batch

    Returns:
        dictionary with predictions and uncertainty estimates

    Raises:
        ValueError: if the model has no dropout layers
    """
```

- Document tensor shapes in `Args` and `Returns`; that is what readers of this library need most
- Period after the first line, then a blank line
- Inline comments explain why, not what

## Adding a UQ Method

1. `lightning_uq_box/uq_methods/my_method.py`, subclassing the appropriate base class in
   `uq_methods/base.py` (`BaseModule`, `DeterministicModel`, `PosthocBase`, ...)
2. Export it from `lightning_uq_box/uq_methods/__init__.py`
3. Add a test config under `tests/configs/<task>/` if the method is exercised through the CLI
4. `tests/uq_methods/test_my_method.py`
5. Document it in `docs/api/uq_methods.rst`
6. Consider a tutorial notebook under `docs/tutorials/`

## Testing

```python
class TestMyMethod:
    @pytest.fixture
    def model(self) -> MyMethod:
        return MyMethod(model=MLP(), loss_fn=nn.MSELoss())

    def test_predict_step(self, model: MyMethod) -> None:
        preds = model.predict_step(torch.randn(4, 1))
        assert "pred" in preds
        assert preds["pred"].shape[0] == 4
```

- Tests are collected from `tests/`, mirroring the package layout
- `addopts = "-m 'not slow'"` skips anything marked `@pytest.mark.slow`; use that marker for tests
  that need downloads or long training runs
- Trainers in tests use `fast_dev_run` or a tiny `max_epochs`; keep the suite CPU-only
- `plt.close()` at the end of plotting tests
- Prefer toy datamodules from `lightning_uq_box.datamodules` over generating data inline

## Lightning Conventions

- UQ methods are `LightningModule`s; put shared logic in the base classes rather than duplicating
  `training_step`/`validation_step` across methods
- Post-hoc methods (temperature scaling, conformal prediction, Laplace) subclass `PosthocBase` and
  fit on a calibration loader; they return `None` from `configure_optimizers`
- `predict_step` returns a dict keyed `pred` for the point prediction, plus method-specific keys for
  the uncertainty estimates: `pred_uct`, `aleatoric_uct`, `epistemic_uct`, `samples`, `logits`
- Keep the `LightningCLI` entry point (`lightning_uq_box/main.py`) working; `tests/test_main.py`
  covers it

## Git

- Branch off `main`, one logical change per PR
- Add the license header to new files
- Do not commit `.venv/`, `dist/`, or `docs/_build/`

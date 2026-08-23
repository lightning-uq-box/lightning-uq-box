# Method × task transition roadmap

This roadmap tracks the migration from concrete method/task classes such as
`MCDropoutRegression` to one canonical method with an explicit task value:

```python
MCDropout(
    model=model,
    num_mc_samples=20,
    loss_fn=nn.CrossEntropyLoss(),
    task=ClassificationTask(mode="multiclass"),
)
```

It is deliberately organized as a stack of reviewable pull requests. A PR in
this stack must only depend on the PR immediately below it (or on `main` once
its parent has merged). Do not combine unrelated families merely to reduce the
number of pull requests.

## Compatibility policy

Canonical APIs are introduced in 0.4. Existing concrete class names and their
historical module imports remain available as deprecated adapters throughout
0.4; adapters must retain their public constructor signatures and state-dict
prefixes. A deprecated adapter emits `DeprecationWarning` with `stacklevel=2`.

The 0.5 removal is a separate release PR and may only remove an adapter after
the canonical replacement has shipped in a 0.4 release with its migration
tests, configuration, and documentation.

## Landed foundation PR

**PR 1: task-aware foundation, Deterministic, and MC Dropout.**

This PR introduces frozen, serializable task values; task/output/capability
contracts; the private runtime and non-mutating prediction writers; and the
canonical `Deterministic` and `MCDropout` APIs. It retains the corresponding
legacy classes as adapters, supplies canonical configs for the four task
families, and updates the transformed tutorials. The public API and changed
utilities have `versionchanged` annotations.

This is a vertical slice, not a declaration that the global inventory is
complete. In particular, the full 109-config classification, distributed
two-process writer fixture, and every checkpoint baseline remain work for the
follow-up PRs below.

## Follow-up PR stack

Create each branch from the preceding PR branch while the stack is open. Rebase
the child onto its parent after changes, then retarget it to `main` as parents
merge. Each PR owns its code, canonical/adaptor configs, focused tests,
`MethodSpec` capabilities, and documentation rows.

| Order | Suggested PR title | Scope and required proof |
| --- | --- | --- |
| 2 | `refactor: add canonical SWAG` | Migrate `swag.py` with local adapters. Preserve moment collection, posterior sampling, and state restoration; test repeated test/predict calls and all declared task capabilities. |
| 3 | `refactor: add canonical SGLD` | Migrate `sgld.py` as `SGLDMethod` so the public optimizer stays unambiguous. Preserve posterior/state semantics and prove the adapters, strict loading, and declared capabilities. |
| 4 | `refactor: add canonical Masksembles` | Migrate `masked_ensemble.py`. Preserve mask construction, setup, freezing, optimization, and aggregation; add canonical configs and adapter/state-key tests. |
| 5 | `refactor: add canonical BNN-VI ELBO` | Migrate `bnn_vi_elbo.py`. Keep variational-layer setup, ELBO/loss and optimizer ownership, freezing, and aggregation method-owned. |
| 6 | `refactor: spike Deep Ensemble reconstruction` | Add serializable member descriptors and prove fresh-process strict reconstruction plus `Trainer(..., ckpt_path=...)`. Document required caller descriptors for historical artifacts. Do not add the canonical public method until this passes. |
| 7 | `refactor: add canonical DeepEnsemble` | Build the canonical Deep Ensemble only for reconstruction-proven task pairs; retain local adapters and cover persistence, strict load, trainer restore, imports, and configs. |
| 8 | `refactor: spike DKL and DUE checkpoint topology` | Save/rebuild GP topology before strict loading for regression and classification DKL/DUE. Derive historical dimensions only from saved hparams/inducing-point state and retain method-owned GP lifecycle. |
| 9 | `refactor: add canonical DKL and DUE` | Expose only topology-proven capabilities, bind local adapters, and preserve explicit GP output conversion contracts. No lazy runtime and no non-strict load fallback. |
| 10 | `refactor: migrate Gaussian, evidential, and quantile methods` | Migrate MVE, DER, and Quantile Regression. Keep distributions and quantile conversion local; fixed-output tests must prove multi-target handling and scalar (`lower_quant`/`upper_quant`) versus pixel (`lower`/`upper`) result keys. |
| 11 | `refactor: migrate mixture and density methods` | Migrate MDN and DensityLayerModel. Keep mixture/density conversions method-specific; do not rely on a generic two-channel interpretation. |
| 12 | `refactor: migrate model rewriters` | Migrate VBLL, SNGP, BNNVI, and BNNLVVI. Assert construction order: rewrite/attachment, runtime registration, then trainer/device/optimizer/checkpoint work. Preserve module structure, freezing, loss, and optimizer ownership. |
| 13 | `refactor: migrate standalone wrappers` | Migrate ZigZag, CARD, and Laplace. Alias the third-party Laplace dependency internally and precisely document post-fit and caller-dependency checkpoint limits. |
| 14 | `docs: close method-task inventory` | Classify all remaining exports: conformal methods, RAPS, temperature scaling, inference-time augmentation, ProbUNet/VAE, and image-to-image helpers. Each is canonical, retained legacy, a post-hoc canonical-output consumer, or explicitly out of scope; do not force an invalid task API. |
| 15 | `docs: publish method-task compatibility matrix` | Make documentation and release notes complete: task serialization, capabilities, binary/multilabel encoding, output keys, persistence, checkpoint boundaries, and old-to-new paths. Check documentation rows against `MethodSpec`, and instantiate canonical examples. |
| 16 | `release: remove 0.4 method-task adapters` | The 0.5-only removal PR. Delete only adapters and legacy configs whose canonical replacement has shipped and been release-noted. |

## Non-negotiable checks for every migration PR

- Declare each public task/mode with `TaskCapability` and `OutputSchema` in
  `MethodSpec`; do not advertise an untested pair.
- Add a minimal smoke factory, fixed-output contract test, canonical config,
  API documentation row, and legacy import/signature/warning test for every
  capability.
- Preserve method-owned behavior: head/loss conversion, sampling, model
  rewrites, freezing, optimizer ownership, lifecycle state, and public result
  keys must not be generalized into task shape heuristics.
- Test batch size one, explicit binary encoding (one-logit BCE/sigmoid or
  two-logit CE/softmax), multilabel targets, and applicable writer behavior.
- For constructor or topology changes, test strict state-dict loading,
  `Trainer` restore with `ckpt_path`, and direct `load_from_checkpoint` when
  all dependencies are serializable or explicitly supplied by the caller.
- Verify writers copy outputs, preserve single-rank `preds.csv` and dense HDF5
  behavior, and add/retain rank plus dataset-index shard/manifest coverage for
  distributed persistence.
- Run focused tests, the affected existing task tests, both configuration
  dialects when applicable, `uv run ruff check`, `uv run ty check`, and the
  relevant documentation build. At each layer boundary, also run MethodSpec
  smokes and the full configuration suite.

## Final completion criteria

The transition is complete only when every declared capability is documented
and tested; 0.4 adapters retain imports, signatures, and applicable strict
loads; task/runtime objects do not become checkpoint hyperparameters or
persistent state; method-specific behavior remains contract-tested; and
single- and multi-process prediction persistence is non-mutating and
collision-free.

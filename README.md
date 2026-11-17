# BehaviorDNA

BehaviorDNA is an experimental Python prototype for inspecting neural-network activations. It computes singular-value spectra and related descriptive statistics for selected layer outputs, and provides a low-dimensional activation projection based on those outputs.

This is analysis code, not a validated model-compression system. The reported compressibility values are heuristics; this repository does not establish that they predict task accuracy or the quality of compressed models.

## What is implemented

- `BehaviorAnalyzer` attaches forward hooks to `torch.nn.Linear` and `torch.nn.Conv2d` layers and reports activation statistics, singular values, cumulative variance, a fitted power-law exponent, and a heuristic compressibility score.
- `ModelBehaviorReport` collects per-layer profiles and generates threshold-based recommendations from those scores.
- `BehaviorDNA` extracts SVD basis vectors from layer outputs and provides projection (`compress_activation`) and reconstruction (`decompress_activation`) helpers.
- `BehaviorComparator` compares two reports using differences in compressibility and fitted exponents plus a variance-similarity metric.
- `BehaviorVisualizer` can plot spectra, cumulative variance, and per-layer scores when Matplotlib is available.

These features describe the implementation in `behavior_dna.py`; the scores and recommendations should not be treated as empirical compression or accuracy results.

## Requirements and use from a checkout

The repository declares Python 3.10 or newer and dependencies on PyTorch and NumPy. To install the dependencies listed by the repository, from the project root run:

```bash
python -m pip install -r requirements.txt
```

There is no console-script/CLI entry point. The source is a top-level `behavior_dna.py` module, so run Python from the repository root to import it. For example:

```python
import torch
import torch.nn as nn
from behavior_dna import BehaviorAnalyzer

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 8),
)
inputs = {"input": torch.randn(5, 64)}

report = BehaviorAnalyzer(num_samples=1).analyze_model(model, inputs)
for layer_idx, profile in report.layer_profiles.items():
    stats = profile.behavior_stats
    if stats is not None:
        print(layer_idx, stats.power_law_alpha, stats.compressibility_score)
```

`analyze_model` switches the model to evaluation mode and runs the supplied input for the configured number of passes under `torch.no_grad()`. The passes reuse the same input; they are not a dataset sampler. The analysis hooks only `Linear` and `Conv2d` modules.

The repository contains `pyproject.toml` metadata and dependency declarations, but no CLI or documented/published installation workflow is established here. This example therefore uses the checked-out source rather than claiming a verified `pip install .` package installation.

## Tests and CI

The test suite can be run from the repository root with:

```bash
python -m pytest -q
```

On the inspected commit, `0876cec2a55d9806f374dd547baf7f34719c26d1`, the tests were run locally with pytest cache and Python bytecode writes disabled: **31 passed** in 6.06 seconds. The run emitted one warning because the plotting test called `plt.show()` with a non-interactive Matplotlib backend. This is a local test result, not a claim about a hosted GitHub Actions run.

`.github/workflows/ci.yml` defines tests on Python 3.10, 3.11, and 3.12, plus compile and import/smoke checks. Hosted CI status for this commit is not reported here.

## Limitations and scope

- The compressibility score is a formula combining a fitted spectral exponent and the fraction of variance in the leading components. It is a heuristic, not an experimentally validated predictor of model size, speed, or retained task performance.
- `BehaviorDNA` projects activations into an SVD-derived basis and reconstructs them; it does not compress model weights, fine-tune a model, or demonstrate preserved task behavior. Its `method` argument is stored but the extraction code uses SVD.
- Analysis captures layer outputs in memory and computes a full NumPy SVD for each analyzed layer. Cost can grow with activation size.
- Layer analysis is limited to `Linear` and `Conv2d` modules. The helper that creates dummy inputs for `BehaviorDNA` assumes a simple input shape (including a fixed 7×7 spatial assumption for convolutional input), so arbitrary model architectures and input signatures are not generally handled.
- The repository contains tests, but no measured benchmarks or task-accuracy/compression-quality results are provided.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for the copyright notice and full terms.

# BehaviorDNA

[![CI](https://github.com/kapildhinwa2010-dev/BehaviorDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/BehaviorDNA/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/BehaviorDNA/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/BehaviorDNA/actions/workflows/codeql.yml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Analyze neural-network behaviour through activation spectra and score layer compressibility.

## What it does

BehaviorAnalyzer collects activations per layer, computes SVD spectra,
power-law exponent, top-k variance and a compressibility score, and
produces per-model reports with recommendations. BehaviorDNA builds a
compressed per-layer representation; BehaviorComparator compares two
models; BehaviorVisualizer plots the spectra (matplotlib optional).

## Installation

    pip install torch numpy pytest

## Usage

    from behavior_dna import BehaviorAnalyzer

    analyzer = BehaviorAnalyzer(num_samples=10)
    report = analyzer.analyze_model(model, inputs)
    for rec in report.recommendations:
        print(rec)

    stats = analyzer.analyze_layer(layer, inputs)
    print(stats.power_law_alpha, stats.compressibility_score)

## API

- BehaviorAnalyzer(num_samples=10): analyze_model, analyze_layer
- BehaviorDNA(model, target_dim=64): get_dna, get_all_dna,
  compress_activation, decompress_activation
- BehaviorComparator: compare_models(model1, model2, inputs)
- BehaviorVisualizer: plot_spectral_decay, plot_cumulative_variance,
  plot_compressibility_scores
- create_behavior_analyzer(...) factory

## Honest results and limitations

- The compressibility score is a spectral heuristic (variance
  concentration + power-law fit). It correlates with compressibility
  of activations, but no experiment here links scores to actual
  post-compression accuracy.
- The "Compression Guidelines" section in earlier revisions stated
  expected ratios ("4-8x with minimal accuracy loss") without
  experiments. Treat all such expectations as unmeasured.
- Power-law fitting on few samples (default 10) is noisy; alpha
  estimates move a lot between runs.
- Analysis collects activations in memory (~2-4x layer size per
  sample) and does full SVD per layer - research scale only.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that behaviour-based compression
  did not preserve task behaviour without retraining, and the related
  carrier-mode recall failed about half of noise trials. The robust
  design kept in the toolkit is the Reed-Solomon slot lattice
  (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).

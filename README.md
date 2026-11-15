# BehaviorDNA

[![CI](https://github.com/kapildhinwa2010-dev/BehaviorDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/BehaviorDNA/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/BehaviorDNA/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/BehaviorDNA/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Behavior Analysis and Compression for Neural Networks**

BehaviorDNA analyzes neural network behavior through activation patterns, identifies compressible knowledge structures, and enables behavior-based compression and understanding. This library provides tools to understand how knowledge is structured in neural network activations and identify compression opportunities.

## Features

- **Activation Spectrum Analysis**: SVD-based analysis of activation patterns
- **Power-Law Distribution Detection**: Identify compressible structures in activations
- **Behavior Compressibility Scoring**: Quantify how compressible each layer is
- **Layer-wise Profiling**: Detailed analysis of each layer's behavior
- **Knowledge Concentration Measurement**: Measure how concentrated knowledge is
- **Runtime Behavior Monitoring**: Monitor behavior during training/inference
- **Behavior DNA Representation**: Create compressed behavior representations
- **Model Comparison**: Compare behavior across different models
- **Visualization Tools**: Plot spectral decay, variance, and compressibility

## Installation

```bash
pip install BehaviorDNA

# or from source
pip install git+https://github.com/kapildhinwa2010-dev/BehaviorDNA.git
```

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Matplotlib (optional, for visualization)
- pytest (for testing)

## Usage

### Basic Analysis

```python
import torch
import torch.nn as nn
from behavior_dna import BehaviorAnalyzer

# Create a model
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 100)
)

# Create analyzer
analyzer = BehaviorAnalyzer(num_samples=10)

# Analyze model
inputs = {'input': torch.randn(10, 100)}
report = analyzer.analyze_model(model, inputs)

# Print report
print(report)

# Get recommendations
for rec in report.recommendations:
    print(f"- {rec}")
```

### Layer Analysis

```python
from behavior_dna import BehaviorAnalyzer

# Create analyzer
analyzer = BehaviorAnalyzer(num_samples=5)

# Analyze specific layer
layer = nn.Linear(100, 256)
inputs = torch.randn(10, 100)
stats = analyzer.analyze_layer(layer, inputs)

print(f"Power-law exponent: {stats.power_law_alpha:.3f}")
print(f"Top 50 variance: {stats.top_50_variance:.1f}%")
print(f"Compressibility: {stats.compressibility_score:.2f}")
print(f"Is compressible: {stats.is_compressible}")
```

### Behavior DNA

```python
import torch.nn as nn
from behavior_dna import BehaviorDNA

# Create model
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 100)
)

# Create BehaviorDNA
behavior_dna = BehaviorDNA(
    model=model,
    target_dim=64,
    num_layers=2
)

# Get DNA for a layer
dna = behavior_dna.get_dna(0)
print(f"DNA shape: {dna.shape}")

# Compress activation
activation = torch.randn(10, 256)
compressed = behavior_dna.compress_activation(activation, layer_idx=0)
print(f"Compressed shape: {compressed.shape}")

# Decompress
decompressed = behavior_dna.decompress_activation(compressed, layer_idx=0)
print(f"Decompressed shape: {decompressed.shape}")
```

### Model Comparison

```python
import torch.nn as nn
from behavior_dna import BehaviorAnalyzer, BehaviorComparator

# Create two models
model1 = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 100))
model2 = nn.Sequential(nn.Linear(100, 256), nn.SiLU(), nn.Linear(256, 100))

# Compare behavior
analyzer = BehaviorAnalyzer(num_samples=5)
comparator = BehaviorComparator(analyzer)

inputs = {'input': torch.randn(10, 100)}
metrics = comparator.compare_models(model1, model2, inputs)

print(f"Compressibility difference: {metrics['compressibility_diff']:.4f}")
print(f"Alpha difference: {metrics['avg_alpha_diff']:.4f}")
print(f"Variance similarity: {metrics['variance_similarity']:.4f}")
```

### Visualization

```python
import torch.nn as nn
from behavior_dna import BehaviorAnalyzer, BehaviorVisualizer

# Create model and analyzer
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 100)
)
analyzer = BehaviorAnalyzer(num_samples=10)

# Analyze
inputs = {'input': torch.randn(10, 100)}
report = analyzer.analyze_model(model, inputs)

# Create visualizer
visualizer = BehaviorVisualizer(report)

# Plot spectral decay
visualizer.plot_spectral_decay(layer_idx=0)

# Plot cumulative variance
visualizer.plot_cumulative_variance()

# Plot compressibility scores
visualizer.plot_compressibility_scores()
```

## Analysis Metrics

### Power-Law Exponent (alpha)
- **High alpha (> 1.5)**: Strong power-law, highly compressible
- **Medium alpha (1.0 - 1.5)**: Moderate compressibility
- **Low alpha (< 1.0)**: Weak compressibility, near white noise

### Variance Metrics
- **Top 10 variance**: Percentage of variance in top 10 components
- **Top 50 variance**: Percentage of variance in top 50 components
- **Top 100 variance**: Percentage of variance in top 100 components

### Compressibility Score
- **0.8 - 1.0**: Highly compressible
- **0.6 - 0.8**: Moderately compressible
- **< 0.6**: Low compressibility

## Compression Guidelines

Based on the analysis, here are recommendations:

### High Compressibility (Score > 0.8)
- Use SVD-based compression with k=32-64
- Expect 4-8x compression with minimal accuracy loss
- Good for: Transformer attention layers, MLP layers

### Medium Compressibility (Score 0.6-0.8)
- Use selective compression on high-score layers
- Expect 2-4x compression with some accuracy trade-off
- Good for: CNN layers, some transformer layers

### Low Compressibility (Score < 0.6)
- Consider alternative compression methods
- Expect < 2x compression
- Good for: Input layers, output layers

## Performance

### Analysis Speed
- **Small models (< 10M params)**: ~1-2 seconds
- **Medium models (10-100M params)**: ~5-10 seconds
- **Large models (> 100M params)**: ~10-30 seconds

### Memory Usage
- **Activation collection**: ~2-4x model size
- **SVD computation**: ~O(n^2) for layer size n

## API Reference

### BehaviorAnalyzer

```python
class BehaviorAnalyzer
```

Analyzes neural network behavior through activation patterns.

**Parameters:**
- `num_samples` (int): Number of samples for analysis (default: 10)
- `layer_indices` (List[int], optional): Specific layers to analyze
- `device` (str): Device to run on (default: 'cpu')

**Methods:**
- `analyze_model(model, inputs, layer_pattern=None)`: Analyze all layers
- `analyze_layer(layer, inputs)`: Analyze a single layer

### BehaviorDNA

```python
class BehaviorDNA(nn.Module)
```

Creates DNA representation from behavior analysis.

**Parameters:**
- `model` (nn.Module): Model to extract behavior from
- `target_dim` (int): Target dimension for DNA (default: 64)
- `num_layers` (int, optional): Number of layers to include
- `method` (str): Compression method (default: 'svd')

**Methods:**
- `get_dna(layer_idx)`: Get DNA for a specific layer
- `get_all_dna()`: Get all DNA vectors
- `compress_activation(activation, layer_idx)`: Compress activation
- `decompress_activation(compressed, layer_idx)`: Decompress activation

### BehaviorComparator

```python
class BehaviorComparator
```

Compares behavior across different models.

**Parameters:**
- `analyzer` (BehaviorAnalyzer, optional): Analyzer instance

**Methods:**
- `compare_models(model1, model2, inputs)`: Compare two models

### BehaviorVisualizer

```python
class BehaviorVisualizer
```

Visualizes behavior analysis results.

**Parameters:**
- `report` (ModelBehaviorReport, optional): Report to visualize

**Methods:**
- `plot_spectral_decay(layer_idx)`: Plot spectral decay
- `plot_cumulative_variance()`: Plot cumulative variance
- `plot_compressibility_scores()`: Plot compressibility scores

### Data Classes

#### BehaviorStats
Statistics for a single layer's behavior analysis.

**Attributes:**
- `layer_idx` (int): Layer index
- `singular_values` (np.ndarray): Singular values from SVD
- `cumulative_variance` (np.ndarray): Cumulative variance
- `power_law_alpha` (float): Power-law exponent
- `top_10_variance` (float): Top 10 variance percentage
- `top_50_variance` (float): Top 50 variance percentage
- `top_100_variance` (float): Top 100 variance percentage
- `compressibility_score` (float): Compressibility score (0-1)
- `is_compressible` (bool): Whether layer is compressible

#### LayerProfile
Profile for a single layer.

**Attributes:**
- `layer_idx` (int): Layer index
- `layer_name` (str): Layer name
- `input_dim` (int): Input dimension
- `output_dim` (int): Output dimension
- `activation_mean` (float): Mean activation value
- `activation_std` (float): Standard deviation of activations
- `sparsity` (float): Sparsity ratio
- `behavior_stats` (BehaviorStats, optional): Behavior statistics

#### ModelBehaviorReport
Comprehensive report for a model.

**Attributes:**
- `model_name` (str): Model name
- `layer_profiles` (Dict[int, LayerProfile]): Layer profiles
- `overall_compressibility` (float): Overall compressibility score
- `best_layer` (int): Index of best layer
- `worst_layer` (int): Index of worst layer
- `recommendations` (List[str]): Recommendations

**Methods:**
- `add_layer(profile)`: Add a layer profile
- `generate_recommendations()`: Generate recommendations

### Factory Function

```python
def create_behavior_analyzer(num_samples=10, layer_indices=None, device='cpu')
```

Factory function to create behavior analyzer.

**Parameters:**
- `num_samples` (int): Number of samples (default: 10)
- `layer_indices` (List[int], optional): Layer indices
- `device` (str): Device (default: 'cpu')

**Returns:**
- `BehaviorAnalyzer`: Analyzer instance

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [ActivationCompress](https://github.com/kapildhinwa2010-dev/ActivationCompress) - Activation compression suite
- [FourierNet](https://github.com/kapildhinwa2010-dev/FourierNet) - Fourier-based weight generation
- [DNA-Encoder](https://github.com/kapildhinwa2010-dev/DNA-Encoder) - DNA-based model encoding
- [CompressorNet](https://github.com/kapildhinwa2010-dev/CompressorNet) - Neural network compression benchmarks

## Citation

If you use this library, please cite:

```bibtex
@misc{BehaviorDNA2024,
  author = {Kapil Dhinwa},
  title = {BehaviorDNA: Behavior Analysis and Compression for Neural Networks},
  year = {2024},
  url = {https://github.com/kapildhinwa2010-dev/BehaviorDNA},
}
```

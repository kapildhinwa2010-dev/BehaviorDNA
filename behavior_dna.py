"""
BehaviorDNA: Behavior Analysis and Compression for Neural Networks

Analyzes neural network behavior through activation patterns,
identifies compressible knowledge structures, and enables
behavior-based compression and understanding.

Key Features:
- Activation spectrum analysis via SVD
- Power-law distribution detection
- Behavior compressibility scoring
- Layer-wise behavior profiling
- Knowledge concentration measurement
- Runtime behavior monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')


@dataclass
class BehaviorStats:
    """Statistics for behavior analysis."""
    layer_idx: int
    singular_values: np.ndarray
    cumulative_variance: np.ndarray
    power_law_alpha: float
    top_10_variance: float
    top_50_variance: float
    top_100_variance: float
    compressibility_score: float
    is_compressible: bool = field(default=False)
    
    def __str__(self) -> str:
        return (
            f"BehaviorStats(layer={self.layer_idx}, "
            f"alpha={self.power_law_alpha:.3f}, "
            f"top10={self.top_10_variance:.1f}%, "
            f"compressible={self.is_compressible})"
        )


@dataclass
class LayerProfile:
    """Profile for a single layer's behavior."""
    layer_idx: int
    layer_name: str
    input_dim: int
    output_dim: int
    activation_mean: float
    activation_std: float
    sparsity: float
    behavior_stats: Optional[BehaviorStats] = None


@dataclass
class ModelBehaviorReport:
    """Comprehensive behavior report for a model."""
    model_name: str
    layer_profiles: Dict[int, LayerProfile] = field(default_factory=dict)
    overall_compressibility: float = 0.0
    best_layer: int = -1
    worst_layer: int = -1
    recommendations: List[str] = field(default_factory=list)
    
    def add_layer(self, profile: LayerProfile) -> None:
        """Add a layer profile to the report."""
        self.layer_profiles[profile.layer_idx] = profile
        
        # Update overall compressibility
        if profile.behavior_stats:
            scores = [
                p.behavior_stats.compressibility_score 
                for p in self.layer_profiles.values() 
                if p.behavior_stats
            ]
            if scores:
                self.overall_compressibility = np.mean(scores)
        
        # Update best/worst layers
        if profile.behavior_stats:
            if self.best_layer == -1 or (
                profile.behavior_stats.compressibility_score > 
                self.layer_profiles[self.best_layer].behavior_stats.compressibility_score
            ):
                self.best_layer = profile.layer_idx
            
            if self.worst_layer == -1 or (
                profile.behavior_stats.compressibility_score < 
                self.layer_profiles[self.worst_layer].behavior_stats.compressibility_score
            ):
                self.worst_layer = profile.layer_idx
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if self.overall_compressibility > 0.8:
            recommendations.append(
                "High compressibility detected. Consider SVD-based compression for all layers."
            )
        elif self.overall_compressibility > 0.6:
            recommendations.append(
                "Moderate compressibility. Use selective compression on high-score layers."
            )
        else:
            recommendations.append(
                "Low compressibility. Consider alternative compression methods."
            )
        
        if self.best_layer != -1:
            recommendations.append(
                f"Layer {self.best_layer} has highest compressibility ({self.layer_profiles[self.best_layer].behavior_stats.compressibility_score:.2f})."
            )
        
        if self.worst_layer != -1:
            recommendations.append(
                f"Layer {self.worst_layer} has lowest compressibility ({self.layer_profiles[self.worst_layer].behavior_stats.compressibility_score:.2f})."
            )
        
        self.recommendations = recommendations
        return recommendations
    
    def __str__(self) -> str:
        return (
            f"ModelBehaviorReport({self.model_name})\n"
            f"  Overall Compressibility: {self.overall_compressibility:.2f}\n"
            f"  Best Layer: {self.best_layer}\n"
            f"  Worst Layer: {self.worst_layer}\n"
            f"  Recommendations: {len(self.recommendations)}"
        )


class BehaviorAnalyzer:
    """
    Analyzes neural network behavior through activation patterns.
    
    Provides tools to understand how knowledge is structured in
    neural network activations and identify compression opportunities.
    
    Args:
        num_samples: Number of samples for analysis
        layer_indices: Specific layers to analyze (None for all)
        device: Device to run analysis on
    """
    
    def __init__(
        self,
        num_samples: int = 10,
        layer_indices: Optional[List[int]] = None,
        device: str = 'cpu'
    ):
        self.num_samples = num_samples
        self.layer_indices = layer_indices
        self.device = device
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[int, List[torch.Tensor]] = defaultdict(list)
    
    def _fit_power_law(
        self,
        singular_values: np.ndarray,
        max_components: int = 200
    ) -> Tuple[float, float]:
        """
        Fit power-law distribution to singular values.
        
        Args:
            singular_values: Array of singular values
            max_components: Maximum number of components to fit
            
        Returns:
            Tuple of (alpha exponent, R-squared)
        """
        if len(singular_values) < 2:
            return 0.0, 0.0
        
        # Use top components
        valid_idx = np.arange(1, min(max_components, len(singular_values)))
        log_i = np.log10(valid_idx)
        log_S = np.log10(singular_values[1:min(max_components, len(singular_values))] + 1e-9)
        
        # Linear regression
        A_fit = np.vstack([log_i, np.ones_like(log_i)]).T
        slope, intercept = np.linalg.lstsq(A_fit, log_S, rcond=None)[0]
        alpha = -slope
        
        # Calculate R-squared
        predicted = slope * log_i + intercept
        ss_res = np.sum((log_S - predicted) ** 2)
        ss_tot = np.sum((log_S - np.mean(log_S)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return alpha, r_squared
    
    def _analyze_activation_matrix(
        self,
        matrix: np.ndarray,
        layer_idx: int
    ) -> BehaviorStats:
        """
        Analyze activation matrix using SVD.
        
        Args:
            matrix: 2D activation matrix (samples x features)
            layer_idx: Layer index
            
        Returns:
            BehaviorStats for the layer
        """
        # SVD
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Variance explained
        var = S ** 2
        total_var = var.sum()
        cum_var = np.cumsum(var) / total_var
        
        # Power-law fit
        alpha, r_squared = self._fit_power_law(S)
        
        # Compute metrics
        n_components = len(S)
        top_10 = cum_var[min(9, n_components-1)] * 100 if n_components > 10 else 100.0
        top_50 = cum_var[min(49, n_components-1)] * 100 if n_components > 50 else 100.0
        top_100 = cum_var[min(99, n_components-1)] * 100 if n_components > 100 else 100.0
        
        # Compressibility score (higher is more compressible)
        # Based on power-law exponent and variance concentration
        compressibility = min(alpha / 2.0, 1.0) * 0.5 + (top_50 / 100.0) * 0.5
        
        # Determine if compressible
        is_compressible = alpha > 1.0 and top_50 > 50.0
        
        return BehaviorStats(
            layer_idx=layer_idx,
            singular_values=S,
            cumulative_variance=cum_var,
            power_law_alpha=alpha,
            top_10_variance=top_10,
            top_50_variance=top_50,
            top_100_variance=top_100,
            compressibility_score=compressibility,
            is_compressible=is_compressible
        )
    
    def _collect_activations(
        self,
        module: nn.Module,
        inp: torch.Tensor,
        out: torch.Tensor,
        layer_idx: int
    ) -> None:
        """Hook function to collect activations."""
        self._activations[layer_idx].append(out.detach().float().cpu())
    
    def analyze_model(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_pattern: Optional[str] = None
    ) -> ModelBehaviorReport:
        """
        Analyze model behavior across all layers.
        
        Args:
            model: PyTorch model to analyze
            inputs: Input tensors for analysis
            layer_pattern: Pattern to match layer names
            
        Returns:
            ModelBehaviorReport with analysis results
        """
        # Clear previous activations
        self._activations.clear()
        self._hooks.clear()
        
        # Collect activations
        layer_idx = 0
        for name, module in model.named_modules():
            # Skip non-linear layers
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            
            # Skip if pattern doesn't match
            if layer_pattern and layer_pattern not in name:
                continue
            
            # Skip if not in specified indices
            if self.layer_indices is not None and layer_idx not in self.layer_indices:
                layer_idx += 1
                continue
            
            # Register hook
            hook = module.register_forward_hook(
                lambda m, i, o, idx=layer_idx: self._collect_activations(m, i, o, idx)
            )
            self._hooks.append(hook)
            layer_idx += 1
        
        # Run model multiple times
        model.eval()
        with torch.no_grad():
            for _ in range(self.num_samples):
                # Move inputs to device
                if isinstance(inputs, dict):
                    device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    model(**device_inputs)
                else:
                    model(inputs.to(self.device))
        
        # Remove hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []
        
        # Analyze each layer
        report = ModelBehaviorReport(model_name=model.__class__.__name__)
        
        for layer_id, activations in self._activations.items():
            # Stack activations
            stacked = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
            matrix = stacked.numpy()
            
            # Get layer info
            layer_name = f"layer_{layer_id}"
            input_dim = stacked.shape[-1]
            output_dim = stacked.shape[-1]
            
            # Compute basic stats
            activation_mean = float(stacked.mean().item())
            activation_std = float(stacked.std().item())
            sparsity = float((stacked.abs() < 1e-6).float().mean().item())
            
            # Analyze behavior
            behavior_stats = self._analyze_activation_matrix(matrix, layer_id)
            
            # Create profile
            profile = LayerProfile(
                layer_idx=layer_id,
                layer_name=layer_name,
                input_dim=input_dim,
                output_dim=output_dim,
                activation_mean=activation_mean,
                activation_std=activation_std,
                sparsity=sparsity,
                behavior_stats=behavior_stats
            )
            
            report.add_layer(profile)
        
        # Generate recommendations
        report.generate_recommendations()
        
        return report
    
    def analyze_layer(
        self,
        layer: nn.Module,
        inputs: torch.Tensor
    ) -> BehaviorStats:
        """
        Analyze a single layer's behavior.
        
        Args:
            layer: Layer to analyze
            inputs: Input tensor
            
        Returns:
            BehaviorStats for the layer
        """
        # Collect activations
        activations = []
        
        def hook(m, i, o):
            activations.append(o.detach().float().cpu())
        
        handle = layer.register_forward_hook(hook)
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                layer(inputs.to(self.device))
        
        handle.remove()
        
        # Stack and analyze
        stacked = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
        matrix = stacked.numpy()
        
        return self._analyze_activation_matrix(matrix, 0)


class BehaviorDNA(nn.Module):
    """
    DNA representation based on behavior analysis.
    
    Creates a compressed representation of model behavior
    that can be used for understanding, compression, or transfer.
    
    Args:
        model: Model to extract behavior from
        target_dim: Target dimension for DNA
        num_layers: Number of layers to include
        method: Compression method ('svd', 'pca', 'autoencoder')
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_dim: int = 64,
        num_layers: Optional[int] = None,
        method: str = 'svd'
    ):
        super().__init__()
        self.model = model
        self.target_dim = target_dim
        self.method = method.lower()
        
        # Extract behavior
        # SVD needs at least target_dim samples to yield target_dim components
        self.analyzer = BehaviorAnalyzer(num_samples=max(5, target_dim))
        self.behavior_dna: Dict[int, torch.Tensor] = {}
        self.basis_vectors: Dict[int, torch.Tensor] = {}
        
        # Extract DNA from model
        self._extract_behavior_dna(num_layers)
    
    def _extract_behavior_dna(self, num_layers: Optional[int] = None) -> None:
        """Extract behavior DNA from model."""
        # Create dummy inputs
        input_dim = self._get_input_dim()
        inputs = {'input': torch.randn(self.analyzer.num_samples, input_dim)}
        
        # Analyze model
        report = self.analyzer.analyze_model(self.model, inputs)
        
        # Extract DNA from each layer
        layer_count = 0
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            
            if num_layers is not None and layer_count >= num_layers:
                break
            
            # Get activations
            activations = []
            
            def hook(m, args, output, acc=None):
                out = output[0] if isinstance(output, (tuple, list)) else output
                acc.append(out.detach().float())
            
            import functools
            handle = module.register_forward_hook(functools.partial(hook, acc=activations))

            with torch.no_grad():
                if isinstance(inputs, dict):
                    self.model(**inputs)
                else:
                    self.model(inputs)
            
            handle.remove()
            
            if activations:
                # Stack activations
                stacked = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
                
                # Compute SVD
                _, _, vt = torch.linalg.svd(stacked, full_matrices=False)
                
                # Store top-k vectors as DNA
                dna = vt[:self.target_dim].T  # (D, k)
                self.behavior_dna[layer_count] = dna
                self.basis_vectors[layer_count] = vt[:self.target_dim].T
            
            layer_count += 1
    
    def _get_input_dim(self) -> int:
        """Get input dimension of model."""
        # Try to find first layer
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
            elif isinstance(module, nn.Conv2d):
                return module.in_channels * 7 * 7  # Assume 7x7 input
        return 64  # Default
    
    def get_dna(self, layer_idx: int) -> torch.Tensor:
        """Get DNA for a specific layer."""
        return self.behavior_dna.get(layer_idx, torch.tensor([]))
    
    def get_all_dna(self) -> Dict[int, torch.Tensor]:
        """Get all DNA vectors."""
        return self.behavior_dna.copy()
    
    def compress_activation(
        self,
        activation: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compress activation using behavior DNA.
        
        Args:
            activation: Input activation
            layer_idx: Layer index
            
        Returns:
            Compressed activation
        """
        if layer_idx not in self.behavior_dna:
            return activation
        
        basis = self.behavior_dna[layer_idx].to(activation.dtype)
        
        # Project
        batch_dim = activation.shape[:-1]
        x_flat = activation.reshape(-1, activation.shape[-1])
        codes = x_flat @ basis
        
        return codes.reshape(*batch_dim, -1)
    
    def decompress_activation(
        self,
        compressed: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Decompress activation using behavior DNA.
        
        Args:
            compressed: Compressed activation
            layer_idx: Layer index
            
        Returns:
            Decompressed activation
        """
        if layer_idx not in self.behavior_dna:
            return compressed
        
        basis = self.behavior_dna[layer_idx].to(compressed.dtype)
        
        # Reconstruct
        batch_dim = compressed.shape[:-1]
        x_flat = compressed.reshape(-1, compressed.shape[-1])
        reconstructed = x_flat @ basis.T
        
        return reconstructed.reshape(*batch_dim, -1)


class BehaviorComparator:
    """
    Compare behavior across different models or configurations.
    
    Args:
        analyzer: BehaviorAnalyzer instance
    """
    
    def __init__(self, analyzer: Optional[BehaviorAnalyzer] = None):
        self.analyzer = analyzer or BehaviorAnalyzer()
    
    def compare_models(
        self,
        model1: nn.Module,
        model2: nn.Module,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compare behavior of two models.
        
        Args:
            model1: First model
            model2: Second model
            inputs: Input tensors
            
        Returns:
            Dictionary of comparison metrics
        """
        # Analyze both models
        report1 = self.analyzer.analyze_model(model1, inputs)
        report2 = self.analyzer.analyze_model(model2, inputs)
        
        # Compare
        metrics = {
            'compressibility_diff': abs(
                report1.overall_compressibility - report2.overall_compressibility
            ),
            'avg_alpha_diff': self._compare_alpha(report1, report2),
            'variance_similarity': self._compare_variance(report1, report2),
        }
        
        return metrics
    
    def _compare_alpha(
        self,
        report1: ModelBehaviorReport,
        report2: ModelBehaviorReport
    ) -> float:
        """Compare power-law exponents."""
        alphas1 = [
            p.behavior_stats.power_law_alpha 
            for p in report1.layer_profiles.values() 
            if p.behavior_stats
        ]
        alphas2 = [
            p.behavior_stats.power_law_alpha 
            for p in report2.layer_profiles.values() 
            if p.behavior_stats
        ]
        
        if not alphas1 or not alphas2:
            return 0.0
        
        return abs(np.mean(alphas1) - np.mean(alphas2))
    
    def _compare_variance(
        self,
        report1: ModelBehaviorReport,
        report2: ModelBehaviorReport
    ) -> float:
        """Compare variance distributions."""
        vars1 = [
            p.behavior_stats.top_50_variance 
            for p in report1.layer_profiles.values() 
            if p.behavior_stats
        ]
        vars2 = [
            p.behavior_stats.top_50_variance 
            for p in report2.layer_profiles.values() 
            if p.behavior_stats
        ]
        
        if not vars1 or not vars2:
            return 0.0
        
        return 1.0 - abs(np.mean(vars1) - np.mean(vars2)) / 100.0


class BehaviorVisualizer:
    """
    Visualize behavior analysis results.
    
    Args:
        report: ModelBehaviorReport to visualize
    """
    
    def __init__(self, report: Optional[ModelBehaviorReport] = None):
        self.report = report
    
    def plot_spectral_decay(self, layer_idx: int) -> None:
        """
        Plot spectral decay for a layer.
        
        Args:
            layer_idx: Layer index
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available, skipping plot")
            return
        
        if self.report is None:
            return
        
        profile = self.report.layer_profiles.get(layer_idx)
        if profile is None or profile.behavior_stats is None:
            return
        
        stats = profile.behavior_stats
        
        plt.figure(figsize=(10, 6))
        plt.loglog(stats.singular_values, 'o-')
        plt.xlabel('Singular Value Index (log)')
        plt.ylabel('Magnitude (log)')
        plt.title(f'Layer {layer_idx}: Spectral Decay (alpha={stats.power_law_alpha:.2f})')
        plt.grid(True, which='both', alpha=0.3)
        plt.show()
    
    def plot_cumulative_variance(self) -> None:
        """Plot cumulative variance for all layers."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available, skipping plot")
            return
        
        if self.report is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        for layer_idx, profile in self.report.layer_profiles.items():
            if profile.behavior_stats:
                stats = profile.behavior_stats
                plt.semilogx(
                    np.arange(1, len(stats.cumulative_variance) + 1),
                    stats.cumulative_variance * 100,
                    label=f'Layer {layer_idx}',
                    alpha=0.7
                )
        
        plt.axhline(95, color='gray', linestyle='--', alpha=0.5, label='95% Target')
        plt.xlabel('Number of Components (log)')
        plt.ylabel('Cumulative Variance Explained (%)')
        plt.title('Cumulative Variance by Layer')
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.show()
    
    def plot_compressibility_scores(self) -> None:
        """Plot compressibility scores for all layers."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available, skipping plot")
            return
        
        if self.report is None:
            return
        
        layers = []
        scores = []
        
        for layer_idx, profile in sorted(self.report.layer_profiles.items()):
            if profile.behavior_stats:
                layers.append(layer_idx)
                scores.append(profile.behavior_stats.compressibility_score)
        
        plt.figure(figsize=(10, 6))
        plt.bar(layers, scores, alpha=0.7)
        plt.axhline(self.report.overall_compressibility, color='red', linestyle='--', 
                   label=f'Overall: {self.report.overall_compressibility:.2f}')
        plt.xlabel('Layer Index')
        plt.ylabel('Compressibility Score')
        plt.title('Compressibility Scores by Layer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def create_behavior_analyzer(
    num_samples: int = 10,
    layer_indices: Optional[List[int]] = None,
    device: str = 'cpu'
) -> BehaviorAnalyzer:
    """
    Factory function to create behavior analyzer.
    
    Args:
        num_samples: Number of samples for analysis
        layer_indices: Specific layers to analyze
        device: Device to run on
        
    Returns:
        BehaviorAnalyzer instance
    """
    return BehaviorAnalyzer(
        num_samples=num_samples,
        layer_indices=layer_indices,
        device=device
    )


__all__ = [
    'BehaviorAnalyzer',
    'BehaviorDNA',
    'BehaviorComparator',
    'BehaviorVisualizer',
    'BehaviorStats',
    'LayerProfile',
    'ModelBehaviorReport',
    'create_behavior_analyzer',
]

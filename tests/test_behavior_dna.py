"""
Tests for BehaviorDNA module.

Comprehensive test suite covering behavior analysis,
DNA extraction, comparison, and visualization.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from behavior_dna import (
    BehaviorAnalyzer,
    BehaviorDNA,
    BehaviorComparator,
    BehaviorVisualizer,
    BehaviorStats,
    LayerProfile,
    ModelBehaviorReport,
    create_behavior_analyzer,
)


class TestBehaviorStats:
    """Test BehaviorStats dataclass."""

    def test_behavior_stats_creation(self):
        """Test creating BehaviorStats."""
        stats = BehaviorStats(
            layer_idx=0,
            singular_values=np.array([1.0, 0.5, 0.25]),
            cumulative_variance=np.array([0.8, 0.95, 1.0]),
            power_law_alpha=1.5,
            top_10_variance=80.0,
            top_50_variance=95.0,
            top_100_variance=99.0,
            compressibility_score=0.85,
            is_compressible=True
        )
        
        assert stats.layer_idx == 0
        assert stats.power_law_alpha == 1.5
        assert stats.top_50_variance == 95.0
        assert stats.compressibility_score == 0.85
        assert stats.is_compressible is True

    def test_behavior_stats_str(self):
        """Test BehaviorStats string representation."""
        stats = BehaviorStats(
            layer_idx=0,
            singular_values=np.array([1.0, 0.5]),
            cumulative_variance=np.array([0.8, 1.0]),
            power_law_alpha=1.5,
            top_10_variance=80.0,
            top_50_variance=95.0,
            top_100_variance=99.0,
            compressibility_score=0.85,
            is_compressible=True
        )
        
        str_repr = str(stats)
        assert "BehaviorStats" in str_repr
        assert "alpha=1.500" in str_repr
        assert "compressible=True" in str_repr


class TestLayerProfile:
    """Test LayerProfile dataclass."""

    def test_layer_profile_creation(self):
        """Test creating LayerProfile."""
        stats = BehaviorStats(
            layer_idx=0,
            singular_values=np.array([1.0]),
            cumulative_variance=np.array([1.0]),
            power_law_alpha=1.0,
            top_10_variance=100.0,
            top_50_variance=100.0,
            top_100_variance=100.0,
            compressibility_score=0.5
        )
        
        profile = LayerProfile(
            layer_idx=0,
            layer_name="test_layer",
            input_dim=100,
            output_dim=200,
            activation_mean=0.0,
            activation_std=1.0,
            sparsity=0.1,
            behavior_stats=stats
        )
        
        assert profile.layer_idx == 0
        assert profile.layer_name == "test_layer"
        assert profile.input_dim == 100
        assert profile.output_dim == 200
        assert profile.activation_mean == 0.0
        assert profile.activation_std == 1.0
        assert profile.sparsity == 0.1
        assert profile.behavior_stats is stats


class TestModelBehaviorReport:
    """Test ModelBehaviorReport dataclass."""

    def test_report_creation(self):
        """Test creating ModelBehaviorReport."""
        report = ModelBehaviorReport(model_name="TestModel")
        
        assert report.model_name == "TestModel"
        assert len(report.layer_profiles) == 0
        assert report.overall_compressibility == 0.0

    def test_add_layer(self):
        """Test adding layers to report."""
        report = ModelBehaviorReport(model_name="TestModel")
        
        stats = BehaviorStats(
            layer_idx=0,
            singular_values=np.array([1.0]),
            cumulative_variance=np.array([1.0]),
            power_law_alpha=1.0,
            top_10_variance=100.0,
            top_50_variance=100.0,
            top_100_variance=100.0,
            compressibility_score=0.8
        )
        
        profile = LayerProfile(
            layer_idx=0,
            layer_name="layer_0",
            input_dim=100,
            output_dim=200,
            activation_mean=0.0,
            activation_std=1.0,
            sparsity=0.1,
            behavior_stats=stats
        )
        
        report.add_layer(profile)
        
        assert len(report.layer_profiles) == 1
        assert 0 in report.layer_profiles
        assert report.overall_compressibility == 0.8

    def test_generate_recommendations(self):
        """Test generating recommendations."""
        report = ModelBehaviorReport(model_name="TestModel")
        
        stats = BehaviorStats(
            layer_idx=0,
            singular_values=np.array([1.0]),
            cumulative_variance=np.array([1.0]),
            power_law_alpha=2.0,
            top_10_variance=90.0,
            top_50_variance=98.0,
            top_100_variance=99.5,
            compressibility_score=0.9
        )
        
        profile = LayerProfile(
            layer_idx=0,
            layer_name="layer_0",
            input_dim=100,
            output_dim=200,
            activation_mean=0.0,
            activation_std=1.0,
            sparsity=0.1,
            behavior_stats=stats
        )
        
        report.add_layer(profile)
        recommendations = report.generate_recommendations()
        
        assert len(recommendations) > 0


class TestBehaviorAnalyzer:
    """Test BehaviorAnalyzer class."""

    def test_analyzer_init(self):
        """Test BehaviorAnalyzer initialization."""
        analyzer = BehaviorAnalyzer(num_samples=5)
        
        assert analyzer.num_samples == 5
        assert analyzer.device == 'cpu'

    def test_analyze_simple_model(self):
        """Test analyzing a simple model."""
        analyzer = BehaviorAnalyzer(num_samples=3)
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Create inputs
        inputs = {'input': torch.randn(5, 64)}
        
        # Analyze
        report = analyzer.analyze_model(model, inputs)
        
        # Check report
        assert report.model_name == "Sequential"
        assert len(report.layer_profiles) >= 2  # At least 2 linear layers

    def test_analyze_layer(self):
        """Test analyzing a single layer."""
        analyzer = BehaviorAnalyzer(num_samples=3)
        
        # Create layer
        layer = nn.Linear(64, 128)
        inputs = torch.randn(5, 64)
        
        # Analyze
        stats = analyzer.analyze_layer(layer, inputs)
        
        # Check stats
        assert isinstance(stats, BehaviorStats)
        assert stats.layer_idx == 0
        assert len(stats.singular_values) > 0
        assert len(stats.cumulative_variance) > 0

    def test_analyze_with_layer_indices(self):
        """Test analyzing specific layers."""
        analyzer = BehaviorAnalyzer(num_samples=3, layer_indices=[0])
        
        # Create model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Create inputs
        inputs = {'input': torch.randn(5, 64)}
        
        # Analyze
        report = analyzer.analyze_model(model, inputs)
        
        # Should only analyze first layer
        assert len(report.layer_profiles) >= 1

    def test_power_law_fit(self):
        """Test power-law fitting."""
        analyzer = BehaviorAnalyzer()
        
        # Create synthetic singular values with power-law
        singular_values = np.array([1.0 / (i + 1) ** 1.5 for i in range(100)])
        
        # Fit power law
        alpha, r_squared = analyzer._fit_power_law(singular_values)
        
        # Should detect power-law
        assert alpha > 0
        assert r_squared >= 0


class TestBehaviorDNA:
    """Test BehaviorDNA class."""

    def test_behavior_dna_init(self):
        """Test BehaviorDNA initialization."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        behavior_dna = BehaviorDNA(
            model=model,
            target_dim=32,
            num_layers=2
        )
        
        assert behavior_dna.target_dim == 32
        assert behavior_dna.method == 'svd'

    def test_get_dna(self):
        """Test getting DNA for a layer."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        behavior_dna = BehaviorDNA(
            model=model,
            target_dim=32,
            num_layers=2
        )
        
        # Get DNA for first layer
        dna = behavior_dna.get_dna(0)
        
        assert isinstance(dna, torch.Tensor)
        assert dna.shape[1] == 32  # target_dim

    def test_get_all_dna(self):
        """Test getting all DNA."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        behavior_dna = BehaviorDNA(
            model=model,
            target_dim=32,
            num_layers=2
        )
        
        all_dna = behavior_dna.get_all_dna()
        
        assert isinstance(all_dna, dict)
        assert len(all_dna) >= 1

    def test_compress_decompress(self):
        """Test compression and decompression."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        behavior_dna = BehaviorDNA(
            model=model,
            target_dim=32,
            num_layers=2
        )
        
        # Create activation
        activation = torch.randn(5, 128)
        
        # Compress
        compressed = behavior_dna.compress_activation(activation, layer_idx=0)
        
        # Check shape
        assert compressed.shape[0] == 5
        assert compressed.shape[1] == 32  # target_dim
        
        # Decompress
        decompressed = behavior_dna.decompress_activation(compressed, layer_idx=0)
        
        # Check shape
        assert decompressed.shape == activation.shape


class TestBehaviorComparator:
    """Test BehaviorComparator class."""

    def test_comparator_init(self):
        """Test BehaviorComparator initialization."""
        analyzer = BehaviorAnalyzer(num_samples=3)
        comparator = BehaviorComparator(analyzer)
        
        assert comparator.analyzer is analyzer

    def test_compare_models(self):
        """Test comparing two models."""
        analyzer = BehaviorAnalyzer(num_samples=3)
        comparator = BehaviorComparator(analyzer)
        
        # Create two models
        model1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        model2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )
        
        # Compare
        inputs = {'input': torch.randn(5, 64)}
        metrics = comparator.compare_models(model1, model2, inputs)
        
        # Check metrics
        assert 'compressibility_diff' in metrics
        assert 'avg_alpha_diff' in metrics
        assert 'variance_similarity' in metrics


class TestBehaviorVisualizer:
    """Test BehaviorVisualizer class."""

    def test_visualizer_init(self):
        """Test BehaviorVisualizer initialization."""
        report = ModelBehaviorReport(model_name="TestModel")
        visualizer = BehaviorVisualizer(report)
        
        assert visualizer.report is report

    def test_plot_spectral_decay_no_matplotlib(self):
        """Test spectral decay plot without matplotlib."""
        report = ModelBehaviorReport(model_name="TestModel")
        
        stats = BehaviorStats(
            layer_idx=0,
            singular_values=np.array([1.0, 0.5, 0.25]),
            cumulative_variance=np.array([0.8, 0.95, 1.0]),
            power_law_alpha=1.5,
            top_10_variance=80.0,
            top_50_variance=95.0,
            top_100_variance=99.0,
            compressibility_score=0.85
        )
        
        profile = LayerProfile(
            layer_idx=0,
            layer_name="layer_0",
            input_dim=100,
            output_dim=200,
            activation_mean=0.0,
            activation_std=1.0,
            sparsity=0.1,
            behavior_stats=stats
        )
        
        report.add_layer(profile)
        visualizer = BehaviorVisualizer(report)
        
        # Should not crash even without matplotlib
        try:
            visualizer.plot_spectral_decay(layer_idx=0)
        except ImportError:
            pass  # Expected if matplotlib not available


class TestFactoryFunction:
    """Test factory function."""

    def test_create_analyzer(self):
        """Test creating analyzer with factory."""
        analyzer = create_behavior_analyzer(
            num_samples=5,
            device='cpu'
        )
        
        assert isinstance(analyzer, BehaviorAnalyzer)
        assert analyzer.num_samples == 5
        assert analyzer.device == 'cpu'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model(self):
        """Test with model that has no linear layers."""
        analyzer = BehaviorAnalyzer(num_samples=3)
        
        # Create model with no linear layers
        model = nn.Sequential(
            nn.ReLU(),
            nn.Sigmoid()
        )
        
        # Analyze
        inputs = {'input': torch.randn(5, 64)}
        report = analyzer.analyze_model(model, inputs)
        
        # Should handle gracefully
        assert isinstance(report, ModelBehaviorReport)

    def test_single_sample(self):
        """Test with single sample."""
        analyzer = BehaviorAnalyzer(num_samples=1)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.randn(1, 64)}
        
        report = analyzer.analyze_model(model, inputs)
        
        assert isinstance(report, ModelBehaviorReport)

    def test_large_model(self):
        """Test with larger model."""
        analyzer = BehaviorAnalyzer(num_samples=2)
        
        # Create larger model
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        inputs = {'input': torch.randn(3, 256)}
        report = analyzer.analyze_model(model, inputs)
        
        assert isinstance(report, ModelBehaviorReport)
        assert len(report.layer_profiles) >= 3

    def test_different_dtypes(self):
        """Test with different dtypes."""
        analyzer = BehaviorAnalyzer(num_samples=2)
        
        model = nn.Sequential(nn.Linear(64, 128))
        
        # Float32
        inputs_f32 = {'input': torch.randn(3, 64, dtype=torch.float32)}
        report_f32 = analyzer.analyze_model(model, inputs_f32)
        assert isinstance(report_f32, ModelBehaviorReport)
        
        # Float64
        inputs_f64 = {'input': torch.randn(3, 64, dtype=torch.float64)}
        report_f64 = analyzer.analyze_model(model, inputs_f64)
        assert isinstance(report_f64, ModelBehaviorReport)

    def test_cpu_and_cuda(self):
        """Test on CPU and CUDA (if available)."""
        # CPU
        analyzer_cpu = BehaviorAnalyzer(num_samples=2, device='cpu')
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.randn(3, 64)}
        report_cpu = analyzer_cpu.analyze_model(model, inputs)
        assert isinstance(report_cpu, ModelBehaviorReport)
        
        # CUDA (if available)
        if torch.cuda.is_available():
            analyzer_cuda = BehaviorAnalyzer(num_samples=2, device='cuda')
            model_cuda = nn.Sequential(nn.Linear(64, 128)).cuda()
            inputs_cuda = {'input': torch.randn(3, 64).cuda()}
            report_cuda = analyzer_cuda.analyze_model(model_cuda, inputs_cuda)
            assert isinstance(report_cuda, ModelBehaviorReport)


class TestNumericalStability:
    """Test numerical stability."""

    def test_very_small_values(self):
        """Test with very small activation values."""
        analyzer = BehaviorAnalyzer(num_samples=2)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.randn(3, 64) * 1e-10}
        
        report = analyzer.analyze_model(model, inputs)
        
        assert isinstance(report, ModelBehaviorReport)

    def test_very_large_values(self):
        """Test with very large activation values."""
        analyzer = BehaviorAnalyzer(num_samples=2)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.randn(3, 64) * 1e10}
        
        report = analyzer.analyze_model(model, inputs)
        
        assert isinstance(report, ModelBehaviorReport)

    def test_zeros(self):
        """Test with all zeros."""
        analyzer = BehaviorAnalyzer(num_samples=2)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.zeros(3, 64)}
        
        report = analyzer.analyze_model(model, inputs)
        
        assert isinstance(report, ModelBehaviorReport)


class TestIntegration:
    """Test integration scenarios."""

    def test_transformer_analysis(self):
        """Test analysis on transformer layer."""
        analyzer = BehaviorAnalyzer(num_samples=2)
        
        from torch.nn import TransformerEncoderLayer
        layer = TransformerEncoderLayer(d_model=128, nhead=8)
        model = layer  # Sequential does not accept src=...; use the layer directly
        
        inputs = torch.randn(5, 10, 128)  # positional: avoids kwarg-name coupling
        report = analyzer.analyze_model(model, inputs)
        
        assert isinstance(report, ModelBehaviorReport)

    def test_multiple_analyses(self):
        """Test running multiple analyses sequentially."""
        analyzer = BehaviorAnalyzer(num_samples=2)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.randn(3, 64)}
        
        # Run multiple analyses
        for _ in range(3):
            report = analyzer.analyze_model(model, inputs)
            assert isinstance(report, ModelBehaviorReport)

    def test_behavior_dna_training(self):
        """Test using BehaviorDNA in training loop."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        behavior_dna = BehaviorDNA(
            model=model,
            target_dim=32,
            num_layers=2
        )
        
        # Simulate training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for _ in range(5):
            x = torch.randn(8, 64)
            output = model(x)
            loss = output.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compress activations
            with torch.no_grad():
                activation = model[0](x)
                compressed = behavior_dna.compress_activation(activation, layer_idx=0)
                assert compressed.shape[1] == 32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

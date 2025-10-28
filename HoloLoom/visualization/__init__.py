"""
HoloLoom Visualization Module
=============================

Extensible visualization system following Edward Tufte principles:
- High data-ink ratio (maximize information, minimize chartjunk)
- Small multiples (repeated charts for comparison)
- Sparklines (intense, simple, word-sized graphics)
- Layered information (progressive disclosure)

Components:
-----------
- MatryoshkaAnalysis: Layer-by-layer embedding breakdown
- StreamOfThought: Temporal reasoning trace
- ResonanceVisualization: Feature extraction display
- WarpSpaceVisualization: Tensor manifold projection
- ConvergenceVisualization: Decision collapse display
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

__all__ = [
    'VisualizationData',
    'MatryoshkaAnalysis',
    'StreamOfThought',
    'ResonanceVisualization',
    'WarpSpaceVisualization',
    'ConvergenceVisualization'
]


@dataclass
class VisualizationData:
    """Container for visualization-ready data following Tufte principles."""
    title: str
    data_type: str  # 'timeseries', 'distribution', 'network', 'text'
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    sparkline: Optional[List[float]] = None  # Compact representation
    summary_stats: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        result = {
            'title': self.title,
            'data_type': self.data_type,
            'data': self._serialize_data(),
            'metadata': self.metadata,
            'summary_stats': self.summary_stats
        }
        if self.sparkline:
            result['sparkline'] = self.sparkline
        return result
    
    def _serialize_data(self) -> Any:
        """Convert numpy/tensor data to lists."""
        if isinstance(self.data, np.ndarray):
            return self.data.tolist()
        elif isinstance(self.data, dict):
            return {k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in self.data.items()}
        return self.data


class MatryoshkaAnalysis:
    """Visualize layer-by-layer embedding analysis."""
    
    @staticmethod
    def create_layer_breakdown(interpretation: Dict[str, Any]) -> List[VisualizationData]:
        """
        Create visualization data for each Matryoshka layer.
        
        Returns small multiples showing:
        - Signal characteristics per layer
        - Feature detection per layer
        - Activation patterns (sparklines)
        """
        visualizations = []
        
        for layer_name, layer_data in interpretation.get('layers', {}).items():
            # Extract embedding characteristics
            embedding = layer_data.get('embedding', {})
            
            # Create sparkline from activation distribution
            activations = embedding.get('raw_embedding', [])
            if activations:
                # Sample 50 points for sparkline
                step = max(1, len(activations) // 50)
                sparkline = [float(a) for a in activations[::step]]
            else:
                sparkline = None
            
            # Summary statistics
            stats = {
                'magnitude': embedding.get('magnitude', 0),
                'mean_activation': embedding.get('mean_activation', 0),
                'sparsity': embedding.get('sparsity', 0),
                'strength_score': MatryoshkaAnalysis._strength_to_score(
                    embedding.get('strength', 'WEAK')
                )
            }
            
            viz = VisualizationData(
                title=f"{layer_name} Analysis",
                data_type='distribution',
                data={
                    'characteristics': embedding,
                    'features': layer_data.get('features', []),
                    'summary': layer_data.get('summary', '')
                },
                metadata={
                    'layer': layer_name,
                    'dimension': layer_data.get('dimension', 0),
                    'purpose': layer_data.get('purpose', '')
                },
                sparkline=sparkline,
                summary_stats=stats
            )
            visualizations.append(viz)
        
        return visualizations
    
    @staticmethod
    def _strength_to_score(strength: str) -> float:
        """Convert strength label to numeric score."""
        mapping = {'WEAK': 0.25, 'MODERATE': 0.5, 'STRONG': 0.75, 'VERY_STRONG': 1.0}
        return mapping.get(strength, 0.0)


class StreamOfThought:
    """Visualize temporal reasoning trace."""
    
    @staticmethod
    def create_thought_stream(spacetime_trace: Dict[str, Any]) -> VisualizationData:
        """
        Create stream of thought visualization.
        
        Shows temporal flow of:
        - Query processing steps
        - Module activations
        - Decision points
        - Memory retrievals
        """
        events = spacetime_trace.get('events', [])
        
        # Create timeline data
        timeline = []
        for idx, event in enumerate(events):
            timeline.append({
                'step': idx + 1,
                'module': event.get('module', 'unknown'),
                'action': event.get('action', ''),
                'timestamp': event.get('timestamp', 0),
                'duration_ms': event.get('duration_ms', 0),
                'metadata': event.get('metadata', {})
            })
        
        # Create sparkline of processing times
        sparkline = [e['duration_ms'] for e in timeline if 'duration_ms' in e]
        
        # Summary statistics
        total_time = sum(e.get('duration_ms', 0) for e in timeline)
        stats = {
            'total_steps': len(timeline),
            'total_time_ms': total_time,
            'avg_step_time_ms': total_time / len(timeline) if timeline else 0,
            'modules_activated': len(set(e['module'] for e in timeline))
        }
        
        return VisualizationData(
            title='Stream of Thought',
            data_type='timeseries',
            data={'timeline': timeline},
            sparkline=sparkline,
            summary_stats=stats,
            metadata={'trace_id': spacetime_trace.get('trace_id', '')}
        )


class ResonanceVisualization:
    """Visualize feature extraction from ResonanceShed."""
    
    @staticmethod
    def create_feature_display(resonance_data: Dict[str, Any]) -> VisualizationData:
        """
        Create visualization of extracted features.
        
        Shows:
        - Feature importance (small multiples)
        - Feature clusters
        - Activation patterns
        """
        features = resonance_data.get('features', [])
        
        # Create feature importance distribution
        importance = [f.get('importance', 0) for f in features]
        
        # Summary statistics
        stats = {
            'total_features': len(features),
            'avg_importance': np.mean(importance) if importance else 0,
            'max_importance': max(importance) if importance else 0,
            'feature_density': len(features) / resonance_data.get('input_length', 1)
        }
        
        return VisualizationData(
            title='Feature Extraction',
            data_type='distribution',
            data={
                'features': features,
                'clusters': resonance_data.get('clusters', [])
            },
            sparkline=importance[:50] if importance else None,
            summary_stats=stats
        )


class WarpSpaceVisualization:
    """Visualize tensor manifold from WarpSpace."""
    
    @staticmethod
    def create_manifold_projection(warp_data: Dict[str, Any]) -> VisualizationData:
        """
        Create 2D projection of high-dimensional warp space.
        
        Uses PCA or t-SNE for dimensionality reduction.
        Shows decision boundary and clusters.
        """
        tensors = warp_data.get('tensors', [])
        
        if not tensors:
            return VisualizationData(
                title='Warp Space Projection',
                data_type='network',
                data={'points': []},
                summary_stats={'dimensions': 0}
            )
        
        # Simple 2D projection (in production, use PCA/t-SNE)
        if isinstance(tensors[0], (list, np.ndarray)):
            # Take first 2 dimensions as proxy
            points = [[float(t[0]), float(t[1])] for t in tensors if len(t) >= 2]
        else:
            points = []
        
        stats = {
            'total_points': len(tensors),
            'dimensions': len(tensors[0]) if tensors else 0,
            'density': len(points) / 100  # Normalized density
        }
        
        return VisualizationData(
            title='Warp Space Projection',
            data_type='network',
            data={'points': points, 'clusters': warp_data.get('clusters', [])},
            summary_stats=stats
        )


class ConvergenceVisualization:
    """Visualize decision collapse from ConvergenceEngine."""
    
    @staticmethod
    def create_decision_display(convergence_data: Dict[str, Any]) -> VisualizationData:
        """
        Create visualization of decision convergence.
        
        Shows:
        - Probability distribution over actions
        - Confidence evolution
        - Decision path
        """
        decisions = convergence_data.get('decisions', [])
        probabilities = convergence_data.get('probabilities', [])
        
        # Create sparkline of confidence over time
        confidence_history = convergence_data.get('confidence_history', [])
        sparkline = confidence_history[:50] if confidence_history else probabilities[:50]
        
        stats = {
            'final_confidence': convergence_data.get('final_confidence', 0),
            'entropy': convergence_data.get('entropy', 0),
            'convergence_steps': len(confidence_history),
            'top_probability': max(probabilities) if probabilities else 0
        }
        
        return VisualizationData(
            title='Decision Convergence',
            data_type='distribution',
            data={
                'decisions': decisions,
                'probabilities': probabilities,
                'path': convergence_data.get('decision_path', [])
            },
            sparkline=sparkline,
            summary_stats=stats
        )

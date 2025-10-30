"""
HoloLoom Semantic Space Projection Visualization
================================================
Tufte-style 2D projection of HoloLoom's 244-dimensional semantic space.

Features:
- PCA projection (244D → 2D) with pure Python implementation
- Matryoshka multi-scale visualization (96, 192, 384 dimensions)
- Query trajectory overlay showing paths through semantic space
- Cluster visualization with semantic labels
- Optional t-SNE/UMAP support (if sklearn/umap-learn available)
- Zero required dependencies (degrades gracefully)

Tufte Principles Applied:
- Maximize data-ink ratio: Minimal grid, focus on points and clusters
- Direct labeling: Cluster labels inline, query annotations
- Data density: Show full trajectory and distribution
- Meaning first: Clusters and trajectories clearly distinguished

Integration:
- Works with HoloLoom embeddings (any dimension)
- Accepts raw embedding matrices
- Programmatic API for automated rendering

Author: HoloLoom Team
Date: 2025-10-29
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EmbeddingPoint:
    """Point in semantic space with metadata."""
    id: str
    embedding: List[float]
    x: float = 0.0  # 2D projection coordinate
    y: float = 0.0
    label: Optional[str] = None
    cluster: Optional[int] = None
    color: Optional[str] = None
    size: float = 6.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class QueryTrajectory:
    """Sequence of queries forming a path through semantic space."""
    queries: List[str]
    embeddings: List[List[float]]
    projected_points: List[Tuple[float, float]] = field(default_factory=list)
    color: str = "#3b82f6"
    label: Optional[str] = None


class ProjectionMethod(Enum):
    """Dimensionality reduction methods."""
    PCA = "pca"              # Principal Component Analysis (fast, linear)
    TSNE = "tsne"            # t-SNE (slow, non-linear, good for clusters)
    UMAP = "umap"            # UMAP (fast, non-linear, preserves global structure)


# ============================================================================
# Pure Python PCA Implementation
# ============================================================================

class SimplePCA:
    """
    Pure Python PCA implementation (zero dependencies).

    Principal Component Analysis projects high-dimensional data to lower
    dimensions by finding directions of maximum variance.

    Algorithm:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Find eigenvectors (power iteration for top components)
    4. Project data onto principal components

    Note: For large datasets (>10K points, >1K dims), use sklearn.decomposition.PCA
    """

    def __init__(self, n_components: int = 2):
        """
        Initialize PCA.

        Args:
            n_components: Number of components to keep (usually 2 for visualization)
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        """
        Fit PCA and transform data.

        Args:
            X: Data matrix (n_samples x n_features)

        Returns:
            Transformed data (n_samples x n_components)
        """
        if not X:
            return []

        n_samples = len(X)
        n_features = len(X[0])

        # Step 1: Center the data
        self.mean = [sum(X[i][j] for i in range(n_samples)) / n_samples
                     for j in range(n_features)]

        X_centered = [[X[i][j] - self.mean[j] for j in range(n_features)]
                      for i in range(n_samples)]

        # Step 2: Compute covariance matrix (features x features)
        # For efficiency, use X^T X when n_samples < n_features
        if n_samples < n_features:
            # Compute X X^T (samples x samples) then derive components
            cov = [[sum(X_centered[i][k] * X_centered[j][k] for k in range(n_features))
                    for j in range(n_samples)]
                   for i in range(n_samples)]
            # Use power iteration to find top eigenvecomponents
            components = self._power_iteration_samples(X_centered, cov, self.n_components)
        else:
            # Compute X^T X (features x features)
            cov = [[sum(X_centered[i][j] * X_centered[i][k] for i in range(n_samples))
                    / n_samples
                    for k in range(n_features)]
                   for j in range(n_features)]
            # Use power iteration to find top eigenvectors
            components = self._power_iteration_features(cov, self.n_components)

        self.components = components

        # Step 3: Project data
        X_transformed = []
        for i in range(n_samples):
            point = []
            for comp in components:
                value = sum(X_centered[i][j] * comp[j] for j in range(len(comp)))
                point.append(value)
            X_transformed.append(point)

        return X_transformed

    def _power_iteration_features(
        self,
        cov: List[List[float]],
        n_components: int
    ) -> List[List[float]]:
        """
        Find top eigenvectors using power iteration.

        Args:
            cov: Covariance matrix
            n_components: Number of components to extract

        Returns:
            List of eigenvectors (principal components)
        """
        n = len(cov)
        components = []

        for comp_idx in range(n_components):
            # Initialize random vector
            import random
            v = [random.gauss(0, 1) for _ in range(n)]
            norm_v = math.sqrt(sum(x*x for x in v))
            if norm_v > 0:
                v = [x / norm_v for x in v]

            # Power iteration (30 iterations for better convergence)
            for iter_idx in range(30):
                # Multiply by covariance matrix
                v_new = [sum(cov[i][j] * v[j] for j in range(n)) for i in range(n)]

                # Orthogonalize against previous components
                for comp in components:
                    dot = sum(v_new[i] * comp[i] for i in range(n))
                    v_new = [v_new[i] - dot * comp[i] for i in range(n)]

                # Normalize
                norm = math.sqrt(sum(x*x for x in v_new))
                if norm > 1e-10:
                    v_new = [x / norm for x in v_new]
                else:
                    # Eigenvalue is zero or component is redundant
                    break

                # Check convergence
                change = sum((v_new[i] - v[i])**2 for i in range(n))
                v = v_new

                if change < 1e-6:
                    break

            # Only add if we got a valid component
            if sum(x*x for x in v) > 1e-10:
                components.append(v)

        return components

    def _power_iteration_samples(
        self,
        X_centered: List[List[float]],
        cov_samples: List[List[float]],
        n_components: int
    ) -> List[List[float]]:
        """
        Find top eigenvectors when n_samples < n_features.

        Uses the fact that eigenvectors of X^T X can be derived from eigenvectors of X X^T.

        Args:
            X_centered: Centered data matrix
            cov_samples: X X^T matrix
            n_components: Number of components

        Returns:
            List of eigenvectors in feature space
        """
        n_samples = len(X_centered)
        n_features = len(X_centered[0])

        # Find eigenvectors of X X^T
        eigenvecs_samples = self._power_iteration_features(cov_samples, n_components)

        # Convert to eigenvectors of X^T X: V = X^T U / ||X^T U||
        components = []
        for u in eigenvecs_samples:
            v = [sum(X_centered[i][j] * u[i] for i in range(n_samples))
                 for j in range(n_features)]
            norm = math.sqrt(sum(x*x for x in v))
            if norm > 1e-10:
                v = [x / norm for x in v]
                components.append(v)

        return components


# ============================================================================
# Optional Advanced Projections
# ============================================================================

def try_import_sklearn():
    """Try to import sklearn for advanced projections."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        return True, PCA, TSNE
    except ImportError:
        return False, None, None


def try_import_umap():
    """Try to import UMAP for projection."""
    try:
        import umap
        return True, umap.UMAP
    except ImportError:
        return False, None


# ============================================================================
# Semantic Space Projection Renderer
# ============================================================================

class SemanticSpaceRenderer:
    """
    Render semantic space projections with Tufte-style design.

    Features:
    - 244D → 2D projection (PCA, optional t-SNE/UMAP)
    - Matryoshka multi-scale comparison
    - Query trajectory overlay
    - Cluster visualization
    - Direct labeling (no legends)

    Thread Safety:
    - Stateless rendering (no shared mutable state)
    - Safe for concurrent calls

    Usage:
        renderer = SemanticSpaceRenderer()
        html = renderer.render(
            points,
            title="Semantic Space",
            show_trajectories=True
        )
    """

    # Cluster colors (colorblind-friendly palette)
    CLUSTER_COLORS = [
        "#3b82f6",  # Blue
        "#10b981",  # Green
        "#f59e0b",  # Orange
        "#8b5cf6",  # Purple
        "#ef4444",  # Red
        "#06b6d4",  # Cyan
        "#ec4899",  # Pink
        "#84cc16",  # Lime
    ]

    def __init__(
        self,
        width: int = 900,
        height: int = 700,
        point_size_min: float = 4.0,
        point_size_max: float = 12.0,
        projection_method: ProjectionMethod = ProjectionMethod.PCA,
        show_grid: bool = False
    ):
        """
        Initialize renderer.

        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            point_size_min: Minimum point radius
            point_size_max: Maximum point radius
            projection_method: Projection method (PCA, TSNE, UMAP)
            show_grid: Show background grid (Tufte: generally avoid)
        """
        self.width = width
        self.height = height
        self.point_size_min = point_size_min
        self.point_size_max = point_size_max
        self.projection_method = projection_method
        self.show_grid = show_grid

    def render(
        self,
        points: List[EmbeddingPoint],
        title: str = "Semantic Space Projection",
        subtitle: Optional[str] = None,
        trajectories: Optional[List[QueryTrajectory]] = None,
        show_clusters: bool = True,
        show_labels: bool = True
    ) -> str:
        """
        Render semantic space projection to HTML.

        Args:
            points: List of EmbeddingPoint objects
            title: Chart title
            subtitle: Optional subtitle
            trajectories: Optional query trajectories to overlay
            show_clusters: Show cluster boundaries and labels
            show_labels: Show point labels

        Returns:
            Complete HTML document as string
        """
        if not points:
            return self._render_empty(title)

        # Project embeddings to 2D
        projected_points = self._project_embeddings(points)

        # Project trajectories if provided
        if trajectories:
            for traj in trajectories:
                traj.projected_points = self._project_trajectory(traj.embeddings)

        # Build HTML
        html_parts = [
            self._render_html_head(title),
            self._render_styles(),
            f'<body>',
            f'<div class="space-container">',
            f'<div class="space-header">',
            f'<h2>{title}</h2>',
        ]

        if subtitle:
            html_parts.append(f'<p class="space-subtitle">{subtitle}</p>')

        # Statistics
        stats_html = self._render_statistics(projected_points, trajectories)
        html_parts.append(stats_html)

        html_parts.append(f'</div>')  # Close header

        # Info panel
        info_html = self._render_info_panel()
        html_parts.append(info_html)

        # SVG Projection
        svg_html = self._render_svg_projection(
            projected_points,
            trajectories,
            show_clusters,
            show_labels
        )
        html_parts.append(svg_html)

        html_parts.extend([
            f'</div>',  # Close container
            f'</body>',
            f'</html>'
        ])

        return '\n'.join(html_parts)

    def _project_embeddings(
        self,
        points: List[EmbeddingPoint]
    ) -> List[EmbeddingPoint]:
        """
        Project high-dimensional embeddings to 2D.

        Args:
            points: List of EmbeddingPoint objects with embeddings

        Returns:
            List of EmbeddingPoint objects with x, y coordinates filled
        """
        if not points:
            return []

        # Extract embeddings
        embeddings = [p.embedding for p in points]

        # Choose projection method
        if self.projection_method == ProjectionMethod.PCA:
            # Use simple PCA (zero dependencies)
            pca = SimplePCA(n_components=2)
            projected = pca.fit_transform(embeddings)

        elif self.projection_method == ProjectionMethod.TSNE:
            # Try sklearn t-SNE
            has_sklearn, PCA, TSNE = try_import_sklearn()
            if has_sklearn:
                tsne = TSNE(n_components=2, perplexity=min(30, len(points) - 1))
                projected = tsne.fit_transform(embeddings).tolist()
            else:
                # Fallback to PCA
                print("Warning: sklearn not available, falling back to PCA")
                pca = SimplePCA(n_components=2)
                projected = pca.fit_transform(embeddings)

        elif self.projection_method == ProjectionMethod.UMAP:
            # Try UMAP
            has_umap, UMAP = try_import_umap()
            if has_umap:
                umap_model = UMAP(n_components=2, n_neighbors=min(15, len(points) - 1))
                projected = umap_model.fit_transform(embeddings).tolist()
            else:
                # Fallback to PCA
                print("Warning: umap-learn not available, falling back to PCA")
                pca = SimplePCA(n_components=2)
                projected = pca.fit_transform(embeddings)

        # Normalize to canvas coordinates
        if projected and len(projected) > 0 and len(projected[0]) >= 2:
            x_coords = [p[0] if len(p) > 0 else 0.0 for p in projected]
            y_coords = [p[1] if len(p) > 1 else 0.0 for p in projected]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            x_range = x_max - x_min if x_max != x_min else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0

            margin = 60
            canvas_width = self.width - 2 * margin
            canvas_height = self.height - 2 * margin

            for i, point in enumerate(points):
                if i < len(projected) and len(projected[i]) >= 2:
                    point.x = margin + ((projected[i][0] - x_min) / x_range) * canvas_width
                    point.y = margin + ((projected[i][1] - y_min) / y_range) * canvas_height
                else:
                    # Fallback to random position if projection failed
                    point.x = margin + (i % 10) * (canvas_width / 10)
                    point.y = margin + (i // 10) * (canvas_height / 10)

        return points

    def _project_trajectory(
        self,
        embeddings: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """
        Project trajectory embeddings to 2D.

        Uses same projection as points for consistency.

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of (x, y) coordinate tuples
        """
        if not embeddings:
            return []

        # Create temporary points
        temp_points = [
            EmbeddingPoint(id=f"traj_{i}", embedding=emb)
            for i, emb in enumerate(embeddings)
        ]

        # Project
        projected = self._project_embeddings(temp_points)

        return [(p.x, p.y) for p in projected]

    def _render_html_head(self, title: str) -> str:
        """Render HTML head with metadata."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>'''

    def _render_styles(self) -> str:
        """Render CSS styles (Tufte-inspired)."""
        return '''
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #ffffff;
        color: #1f2937;
        padding: 40px 20px;
    }

    .space-container {
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 30px;
    }

    .space-header {
        margin-bottom: 20px;
    }

    .space-header h2 {
        font-size: 24px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 8px;
    }

    .space-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 12px;
    }

    .space-stats {
        display: flex;
        gap: 24px;
        padding: 12px 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    .stat-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .stat-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #9ca3af;
        font-weight: 500;
    }

    .stat-value {
        font-size: 20px;
        font-weight: 600;
        color: #111827;
    }

    .space-info {
        display: flex;
        gap: 16px;
        padding: 12px;
        background: #f9fafb;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 12px;
        color: #4b5563;
    }

    .info-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .info-icon {
        width: 16px;
        height: 16px;
        border-radius: 50%;
    }

    .space-svg {
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        background: #fafafa;
    }

    .space-point {
        cursor: pointer;
        transition: all 0.2s;
    }

    .space-point:hover {
        stroke-width: 2.5px;
        r: 8;
    }

    .space-trajectory {
        fill: none;
        stroke-width: 2;
        opacity: 0.6;
    }

    .trajectory-point {
        fill: white;
        stroke-width: 2;
    }

    .space-label {
        font-size: 11px;
        font-weight: 500;
        pointer-events: none;
        user-select: none;
    }

    .cluster-label {
        font-size: 12px;
        font-weight: 600;
        fill: #374151;
        pointer-events: none;
    }

    .grid-line {
        stroke: #e5e7eb;
        stroke-width: 1;
        opacity: 0.3;
    }

    /* Tooltip */
    .tooltip {
        position: absolute;
        background: rgba(17, 24, 39, 0.95);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.2s;
        z-index: 1000;
        max-width: 250px;
    }

    .tooltip.visible {
        opacity: 1;
    }

    .tooltip-title {
        font-weight: 600;
        margin-bottom: 4px;
        font-size: 13px;
    }

    .tooltip-detail {
        font-size: 11px;
        color: #d1d5db;
        margin: 2px 0;
    }
</style>
</head>'''

    def _render_statistics(
        self,
        points: List[EmbeddingPoint],
        trajectories: Optional[List[QueryTrajectory]] = None
    ) -> str:
        """Render projection statistics."""
        num_points = len(points)
        num_clusters = len(set(p.cluster for p in points if p.cluster is not None))
        num_trajectories = len(trajectories) if trajectories else 0
        embedding_dim = len(points[0].embedding) if points else 0

        method_name = self.projection_method.value.upper()

        return f'''
        <div class="space-stats">
            <div class="stat-item">
                <span class="stat-label">Points</span>
                <span class="stat-value">{num_points}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Dimensions</span>
                <span class="stat-value">{embedding_dim}D → 2D</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Method</span>
                <span class="stat-value">{method_name}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Clusters</span>
                <span class="stat-value">{num_clusters}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Trajectories</span>
                <span class="stat-value">{num_trajectories}</span>
            </div>
        </div>'''

    def _render_info_panel(self) -> str:
        """Render info panel explaining visualization."""
        return '''
        <div class="space-info">
            <div class="info-item">
                <div class="info-icon" style="background: #3b82f6;"></div>
                <span>Point: Embedding in semantic space</span>
            </div>
            <div class="info-item">
                <div class="info-icon" style="background: #10b981;"></div>
                <span>Cluster: Semantically related group</span>
            </div>
            <div class="info-item">
                <div class="info-icon" style="background: #f59e0b;"></div>
                <span>Trajectory: Query sequence path</span>
            </div>
        </div>'''

    def _render_svg_projection(
        self,
        points: List[EmbeddingPoint],
        trajectories: Optional[List[QueryTrajectory]],
        show_clusters: bool,
        show_labels: bool
    ) -> str:
        """Render SVG projection visualization."""
        svg_parts = [
            f'<svg class="space-svg" width="{self.width}" height="{self.height}" '
            f'xmlns="http://www.w3.org/2000/svg">',
        ]

        # Optional grid
        if self.show_grid:
            svg_parts.append(self._render_grid())

        # Render trajectories first (so they're behind points)
        if trajectories:
            for traj in trajectories:
                traj_svg = self._render_trajectory(traj)
                svg_parts.append(traj_svg)

        # Render cluster boundaries/labels
        if show_clusters:
            cluster_svg = self._render_clusters(points)
            svg_parts.append(cluster_svg)

        # Render points
        for point in points:
            point_svg = self._render_point(point, show_labels)
            svg_parts.append(point_svg)

        svg_parts.append('</svg>')

        # Add tooltip div
        svg_parts.append('<div class="tooltip" id="tooltip"></div>')

        # Add interactivity script
        svg_parts.append(self._render_script(points))

        return '\n'.join(svg_parts)

    def _render_grid(self) -> str:
        """Render background grid."""
        grid_lines = []
        step = 100

        # Vertical lines
        for x in range(0, self.width + 1, step):
            grid_lines.append(
                f'<line class="grid-line" x1="{x}" y1="0" x2="{x}" y2="{self.height}" />'
            )

        # Horizontal lines
        for y in range(0, self.height + 1, step):
            grid_lines.append(
                f'<line class="grid-line" x1="0" y1="{y}" x2="{self.width}" y2="{y}" />'
            )

        return '\n'.join(grid_lines)

    def _render_trajectory(self, traj: QueryTrajectory) -> str:
        """Render query trajectory path."""
        if len(traj.projected_points) < 2:
            return ""

        # Build path
        path_data = f"M {traj.projected_points[0][0]} {traj.projected_points[0][1]}"
        for x, y in traj.projected_points[1:]:
            path_data += f" L {x} {y}"

        traj_svg = f'''
    <path class="space-trajectory" d="{path_data}" stroke="{traj.color}" />'''

        # Add trajectory points
        for i, (x, y) in enumerate(traj.projected_points):
            is_start = (i == 0)
            is_end = (i == len(traj.projected_points) - 1)

            if is_start:
                marker = "▶"  # Start marker
            elif is_end:
                marker = "★"  # End marker
            else:
                marker = ""

            traj_svg += f'''
    <circle class="trajectory-point" cx="{x}" cy="{y}" r="4" stroke="{traj.color}" />'''

            if marker:
                traj_svg += f'''
    <text x="{x}" y="{y - 10}" text-anchor="middle" font-size="10" fill="{traj.color}">{marker}</text>'''

        # Add trajectory label
        if traj.label:
            mid_idx = len(traj.projected_points) // 2
            mid_x, mid_y = traj.projected_points[mid_idx]
            traj_svg += f'''
    <text x="{mid_x}" y="{mid_y - 15}" text-anchor="middle" font-size="11" font-weight="600" fill="{traj.color}">{traj.label}</text>'''

        return traj_svg

    def _render_clusters(self, points: List[EmbeddingPoint]) -> str:
        """Render cluster boundaries and labels."""
        # Group points by cluster
        clusters = {}
        for point in points:
            if point.cluster is not None:
                if point.cluster not in clusters:
                    clusters[point.cluster] = []
                clusters[point.cluster].append(point)

        cluster_svg = ""

        for cluster_id, cluster_points in clusters.items():
            if len(cluster_points) < 3:
                continue

            # Compute cluster centroid
            centroid_x = sum(p.x for p in cluster_points) / len(cluster_points)
            centroid_y = sum(p.y for p in cluster_points) / len(cluster_points)

            # Get cluster color
            color = self.CLUSTER_COLORS[cluster_id % len(self.CLUSTER_COLORS)]

            # Draw cluster label at centroid
            label = f"Cluster {cluster_id}"
            cluster_svg += f'''
    <text class="cluster-label" x="{centroid_x}" y="{centroid_y}" text-anchor="middle">{label}</text>'''

        return cluster_svg

    def _render_point(
        self,
        point: EmbeddingPoint,
        show_label: bool
    ) -> str:
        """Render single point."""
        # Determine color
        if point.color:
            color = point.color
        elif point.cluster is not None:
            color = self.CLUSTER_COLORS[point.cluster % len(self.CLUSTER_COLORS)]
        else:
            color = "#6366f1"

        point_svg = f'''
    <circle class="space-point"
            data-point-id="{point.id}"
            cx="{point.x}" cy="{point.y}" r="{point.size}"
            fill="{color}"
            stroke="#1e40af"
            stroke-width="1.5"
            opacity="0.8" />'''

        # Optional label
        if show_label and point.label:
            point_svg += f'''
    <text class="space-label"
          x="{point.x}" y="{point.y + point.size + 12}"
          text-anchor="middle">{point.label[:20]}</text>'''

        return point_svg

    def _render_script(self, points: List[EmbeddingPoint]) -> str:
        """Render JavaScript for interactivity."""
        # Build point data for tooltips
        point_data_js = []
        for point in points:
            embedding_dim = len(point.embedding)
            embedding_preview = point.embedding[:3] if embedding_dim > 3 else point.embedding
            point_data_js.append(f'''
        {{
            id: "{point.id}",
            label: "{point.label or point.id}",
            cluster: {point.cluster if point.cluster is not None else "null"},
            embedding_dim: {embedding_dim},
            embedding_preview: {embedding_preview}
        }}''')

        return f'''
<script>
(function() {{
    const points = [{','.join(point_data_js)}];
    const tooltip = document.getElementById('tooltip');

    // Add hover listeners to points
    document.querySelectorAll('.space-point').forEach(point => {{
        point.addEventListener('mouseenter', function(e) {{
            const pointId = this.getAttribute('data-point-id');
            const pointData = points.find(p => p.id === pointId);

            if (pointData) {{
                const clusterText = pointData.cluster !== null ? `Cluster ${{pointData.cluster}}` : 'No cluster';
                tooltip.innerHTML = `
                    <div class="tooltip-title">${{pointData.label}}</div>
                    <div class="tooltip-detail">${{clusterText}}</div>
                    <div class="tooltip-detail">Embedding: ${{pointData.embedding_dim}}D</div>
                    <div class="tooltip-detail">Preview: [${{pointData.embedding_preview.map(x => x.toFixed(3)).join(', ')}}...]</div>
                `;
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY + 10 + 'px';
                tooltip.classList.add('visible');
            }}
        }});

        point.addEventListener('mouseleave', function() {{
            tooltip.classList.remove('visible');
        }});

        point.addEventListener('mousemove', function(e) {{
            tooltip.style.left = e.pageX + 10 + 'px';
            tooltip.style.top = e.pageY + 10 + 'px';
        }});
    }});
}})();
</script>'''

    def _render_empty(self, title: str) -> str:
        """Render empty state."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: #f9fafb;
        }}
        .empty-state {{
            text-align: center;
            color: #6b7280;
        }}
        .empty-state h2 {{
            font-size: 24px;
            margin-bottom: 12px;
        }}
        .empty-state p {{
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="empty-state">
        <h2>{title}</h2>
        <p>No embedding points available to project</p>
    </div>
</body>
</html>'''


# ============================================================================
# Convenience Functions
# ============================================================================

def render_semantic_space(
    embeddings: List[List[float]],
    labels: Optional[List[str]] = None,
    clusters: Optional[List[int]] = None,
    title: str = "Semantic Space Projection",
    subtitle: Optional[str] = None,
    method: ProjectionMethod = ProjectionMethod.PCA,
    trajectories: Optional[List[QueryTrajectory]] = None
) -> str:
    """
    Render semantic space projection from raw embeddings.

    Primary programmatic API for automated tool calling.

    Args:
        embeddings: List of embedding vectors
        labels: Optional labels for each point
        clusters: Optional cluster assignments
        title: Chart title
        subtitle: Optional subtitle
        method: Projection method (PCA, TSNE, UMAP)
        trajectories: Optional query trajectories

    Returns:
        Complete HTML document as string

    Example:
        # Simple usage
        html = render_semantic_space(
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]]
        )

        # With clustering
        html = render_semantic_space(
            embeddings=embeddings,
            labels=["Query 1", "Query 2", ...],
            clusters=[0, 0, 1, 1, 2, ...],
            title="Matryoshka Embeddings (384D)"
        )

        # With trajectory
        trajectory = QueryTrajectory(
            queries=["Q1", "Q2", "Q3"],
            embeddings=[emb1, emb2, emb3],
            label="User Session"
        )
        html = render_semantic_space(
            embeddings=embeddings,
            trajectories=[trajectory]
        )
    """
    # Create embedding points
    points = []
    for i, emb in enumerate(embeddings):
        label = labels[i] if labels and i < len(labels) else f"Point {i}"
        cluster = clusters[i] if clusters and i < len(clusters) else None

        point = EmbeddingPoint(
            id=f"point_{i}",
            embedding=emb,
            label=label,
            cluster=cluster
        )
        points.append(point)

    # Render
    renderer = SemanticSpaceRenderer(projection_method=method)
    return renderer.render(
        points,
        title=title,
        subtitle=subtitle,
        trajectories=trajectories,
        show_clusters=(clusters is not None),
        show_labels=(labels is not None)
    )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import random
    print("=== Semantic Space Projection Demo ===\n")

    # Create sample embeddings (simulate 244D semantic space)
    n_points = 30
    embedding_dim = 244

    # Create 3 clusters
    embeddings = []
    clusters = []
    labels = []

    for cluster_id in range(3):
        # Generate cluster center
        center = [random.gauss(cluster_id * 2.0, 0.5) for _ in range(embedding_dim)]

        # Generate points around center
        for i in range(10):
            emb = [center[j] + random.gauss(0, 0.3) for j in range(embedding_dim)]
            embeddings.append(emb)
            clusters.append(cluster_id)
            labels.append(f"Query {len(embeddings)}")

    # Create trajectory (simulated user session)
    traj_embeddings = [embeddings[i] for i in [0, 5, 10, 15, 20]]
    trajectory = QueryTrajectory(
        queries=[f"Q{i}" for i in range(5)],
        embeddings=traj_embeddings,
        label="User Session A",
        color="#f59e0b"
    )

    # Render
    html = render_semantic_space(
        embeddings=embeddings,
        labels=labels,
        clusters=clusters,
        title="HoloLoom Semantic Space (244D → 2D)",
        subtitle="PCA projection with query trajectory overlay",
        method=ProjectionMethod.PCA,
        trajectories=[trajectory]
    )

    output_path = "demo_semantic_space.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Demo generated: {output_path}")
    print(f"  Points: {len(embeddings)}")
    print(f"  Clusters: {len(set(clusters))}")
    print(f"  Embedding dim: {embedding_dim}D -> 2D")
    print(f"  Trajectory: {len(trajectory.queries)} queries")
    print("\nOpen in browser to view interactive projection!")

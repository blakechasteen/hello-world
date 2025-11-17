"""Demo: AR Heatmap Visualization

Demonstrates heatmap rendering with 8 colormaps:
- HOT: Black -> Red -> Yellow -> White
- COOL: Black -> Blue -> Cyan -> White
- VIRIDIS: Blue -> Green -> Yellow
- PLASMA: Purple -> Orange -> Yellow
- JET: Blue -> Cyan -> Green -> Yellow -> Red
- TURBO: Blue -> Cyan -> Green -> Yellow -> Orange -> Red
- INFERNO: Black -> Purple -> Orange -> Yellow
- GRAYSCALE: Black -> Gray -> White

Includes gradient analysis and difference detection.

Implemented: 2025-11-17
Run: PYTHONPATH=. python demos/demo_ar_heatmaps.py
"""

import asyncio
import numpy as np

from HoloLoom.visualization.ar_heatmap import (
    Heatmap,
    ARHeatmapRenderer,
    ColormapType,
    HeatmapAnimator,
)


async def create_dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dummy camera frame."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200
    return frame


async def demo_colormap(colormap: ColormapType, description: str):
    """Demo a single colormap."""
    print(f"\nColormap: {colormap.value.upper()}")
    print(f"Description: {description}")
    print("-" * 50)

    # Create synthetic heatmap data
    # Simple gradient: bright on left, dark on right
    grid = np.linspace(0, 1, 40).reshape(1, 40)
    grid = np.repeat(grid, 30, axis=0)

    heatmap = Heatmap(
        grid=grid,
        position=(0, 0, 1),
        colormap=colormap,
        label=colormap.value,
        auto_normalize=True,
    )

    # Render
    renderer = ARHeatmapRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render(heatmap, frame, (0, 0, 0))

    print(f"Grid shape: {grid.shape}")
    print(f"Value range: [{grid.min():.3f}, {grid.max():.3f}]")
    print(f"Rendered: {result.shape}")
    print(f"Colormap: {heatmap.colormap.value}")
    print(f"Opacity: {heatmap.opacity}")


async def demo_temperature_heatmap():
    """Demo: Real-world temperature sensor grid."""
    print("\nScenario: Temperature Sensor Grid")
    print("=" * 50)

    # Simulate temperature sensor readings (with hot spot in center)
    grid = np.zeros((10, 15))

    # Create gaussian hot spot
    for i in range(10):
        for j in range(15):
            dist_sq = (i - 5) ** 2 + (j - 7.5) ** 2
            grid[i, j] = np.exp(-dist_sq / 10)

    # Add some noise
    grid += np.random.normal(0, 0.05, grid.shape)
    grid = np.clip(grid, 0, 1)

    print("Grid statistics:")
    print(f"  Shape: {grid.shape}")
    print(f"  Min: {grid.min():.3f}")
    print(f"  Max: {grid.max():.3f}")
    print(f"  Mean: {grid.mean():.3f}")
    print(f"  Std: {grid.std():.3f}")

    # Render with HOT colormap (intuitive for temperature)
    heatmap = Heatmap(
        grid=grid,
        position=(0, 0, 1),
        size=(1.5, 1.0),
        colormap=ColormapType.HOT,
        label="Temperature (Hot Spot)",
        auto_normalize=False,
    )

    renderer = ARHeatmapRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render(heatmap, frame, (0, 0, 0))

    print(f"Rendered with HOT colormap: {result.shape}")
    print(f"Hot spot center: grid[5,7.5] ≈ {grid[5, 7]} (peak temperature)")


async def demo_activity_heatmap():
    """Demo: Activity/occupancy heatmap."""
    print("\nScenario: Occupancy Detection Heatmap")
    print("=" * 50)

    # Simulate occupancy in a room (high activity = bright)
    grid = np.zeros((8, 12))

    # Add people activity locations
    people_positions = [(2, 3), (5, 8), (6, 4)]

    for py, px in people_positions:
        for i in range(8):
            for j in range(12):
                dist = np.sqrt((i - py) ** 2 + (j - px) ** 2)
                activity = np.exp(-dist / 1.5)
                grid[i, j] = max(grid[i, j], activity)

    print("Activity patterns:")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Detected activity regions: {len(people_positions)}")
    print(f"  Coverage: {(grid > 0.1).sum() / grid.size * 100:.1f}%")

    # Render with VIRIDIS (good for activity/occupancy)
    heatmap = Heatmap(
        grid=grid,
        position=(0, 0, 1),
        size=(2.0, 1.5),
        colormap=ColormapType.VIRIDIS,
        label="Activity Detection",
        auto_normalize=False,
    )

    renderer = ARHeatmapRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render(heatmap, frame, (0, 0, 0))

    print(f"Rendered with VIRIDIS colormap: {result.shape}")
    print(f"Peak activity: {grid.max():.3f}")


async def demo_gradient_analysis():
    """Demo: Gradient magnitude visualization."""
    print("\nScenario: Spatial Gradient Analysis")
    print("=" * 50)

    # Create gradient field
    x = np.linspace(-3, 3, 40)
    y = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(x, y)

    # Create field with variable slopes
    field = np.sin(X) * np.cos(Y)

    print("Field statistics:")
    print(f"  Shape: {field.shape}")
    print(f"  Range: [{field.min():.3f}, {field.max():.3f}]")

    # Compute gradient
    gy, gx = np.gradient(field)
    gradient_mag = np.sqrt(gx**2 + gy**2)

    print("\nGradient statistics:")
    print(f"  Min: {gradient_mag.min():.3f}")
    print(f"  Max: {gradient_mag.max():.3f}")
    print(f"  Mean: {gradient_mag.mean():.3f}")

    # Render gradient with PLASMA
    heatmap = Heatmap(
        grid=gradient_mag,
        position=(0, 0, 1),
        colormap=ColormapType.PLASMA,
        label="Gradient Magnitude",
        auto_normalize=True,
    )

    renderer = ARHeatmapRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render(heatmap, frame, (0, 0, 0))

    print(f"Rendered gradient with PLASMA colormap: {result.shape}")
    print(f"Steepest regions: {(gradient_mag > gradient_mag.mean() * 1.5).sum()} pixels")


async def demo_difference_detection():
    """Demo: Difference between two heatmaps."""
    print("\nScenario: Change Detection")
    print("=" * 50)

    # Create two slightly different grids
    t1 = np.random.rand(10, 12)
    t2 = t1.copy()

    # Introduce some changes
    t2[2:4, 3:5] += 0.3  # Bright change
    t2[6:8, 8:10] -= 0.2  # Dark change
    t2 = np.clip(t2, 0, 1)

    print("Time series data:")
    print(f"  T1 range: [{t1.min():.3f}, {t1.max():.3f}]")
    print(f"  T2 range: [{t2.min():.3f}, {t2.max():.3f}]")

    # Create difference heatmap
    renderer = ARHeatmapRenderer()
    diff_heatmap = await renderer.create_difference_heatmap(t1, t2, (0, 0, 1))

    print("\nDifference analysis:")
    print(f"  Max change: {diff_heatmap.grid.max():.3f}")
    print(f"  Mean change: {diff_heatmap.grid.mean():.3f}")
    print(f"  Changed pixels (>0.1): {(diff_heatmap.grid > 0.1).sum()}")

    # Render difference
    frame = await create_dummy_frame()
    result = await renderer.render(diff_heatmap, frame, (0, 0, 0))

    print(f"Rendered difference with JET colormap: {result.shape}")


async def demo_multiple_heatmaps():
    """Demo: Rendering multiple heatmaps together."""
    print("\nScenario: Composite Multi-Layer Visualization")
    print("=" * 50)

    # Create 3 different data layers
    layer1 = np.random.rand(20, 20)  # Base layer
    layer2 = np.sin(np.linspace(0, 2 * np.pi, 20)).reshape(-1, 1)  # Pattern
    layer2 = np.repeat(layer2, 20, axis=1)
    layer3 = np.eye(20)  # Diagonal

    heatmaps = [
        Heatmap(grid=layer1, colormap=ColormapType.HOT, label="Layer 1 (Base)"),
        Heatmap(grid=layer2, colormap=ColormapType.COOL, label="Layer 2 (Pattern)"),
        Heatmap(grid=layer3, colormap=ColormapType.VIRIDIS, label="Layer 3 (Diagonal)"),
    ]

    print(f"Rendering {len(heatmaps)} heatmaps:")
    for i, hm in enumerate(heatmaps, 1):
        print(f"  Layer {i}: {hm.colormap.value} colormap")

    renderer = ARHeatmapRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render_multiple(heatmaps, frame, (0, 0, 0))

    print(f"Composite result: {result.shape}")
    print("Each layer uses a different colormap for visual distinction")


async def demo_animated_heatmap():
    """Demo: Animated heatmap updates."""
    print("\nScenario: Animated Heatmap")
    print("=" * 50)

    animator = HeatmapAnimator(update_interval=0.2)

    # Create initial heatmap
    initial_grid = np.random.rand(15, 20)
    heatmap = Heatmap(
        grid=initial_grid,
        colormap=ColormapType.PLASMA,
        label="Animated",
    )

    await animator.register_heatmap("animated_hm", heatmap)
    print("Registered animated heatmap")

    # Define update function (simulates data change)
    async def update_func():
        """Generate new data with temporal evolution."""
        return np.random.rand(15, 20)

    # Start animation
    await animator.start_animation("animated_hm", update_func)
    print("Animation started")

    # Let it update a few times
    for i in range(3):
        await asyncio.sleep(0.3)
        current = await animator.get_heatmap("animated_hm")
        print(f"  Update {i+1}: grid range [{current.grid.min():.3f}, {current.grid.max():.3f}]")

    # Stop animation
    await animator.stop_animation("animated_hm")
    print("Animation stopped")


async def main():
    """Main demo function."""
    print("=" * 70)
    print("AR HEATMAP VISUALIZATION DEMO")
    print("=" * 70)
    print()

    print("This demo showcases:")
    print("  • 8 different colormaps for various data types")
    print("  • Real-world scenarios (temperature, occupancy, etc.)")
    print("  • Gradient magnitude analysis")
    print("  • Change detection between heatmaps")
    print("  • Multi-layer composite visualization")
    print("  • Real-time heatmap animation")
    print()

    try:
        # Demo all colormaps
        print("\n1. COLORMAP DEMONSTRATIONS")
        print("=" * 70)

        await demo_colormap(ColormapType.HOT, "Black -> Red -> Yellow -> White")
        await demo_colormap(ColormapType.COOL, "Black -> Blue -> Cyan -> White")
        await demo_colormap(ColormapType.VIRIDIS, "Blue -> Green -> Yellow")
        await demo_colormap(ColormapType.PLASMA, "Purple -> Orange -> Yellow")
        await demo_colormap(ColormapType.JET, "Blue -> Cyan -> Green -> Yellow -> Red")
        await demo_colormap(ColormapType.TURBO, "Blue -> ... -> Red -> Magenta")
        await demo_colormap(ColormapType.INFERNO, "Black -> Purple -> Orange -> Yellow")
        await demo_colormap(ColormapType.GRAYSCALE, "Black -> White")

        # Demo real-world scenarios
        print("\n2. REAL-WORLD SCENARIOS")
        print("=" * 70)

        await demo_temperature_heatmap()
        await demo_activity_heatmap()
        await demo_gradient_analysis()
        await demo_difference_detection()
        await demo_multiple_heatmaps()
        await demo_animated_heatmap()

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("  ✓ Demonstrated 8 colormaps")
    print("  ✓ Showed 5 real-world scenarios")
    print("  ✓ Demonstrated gradient analysis")
    print("  ✓ Showed change detection")
    print("  ✓ Demonstrated animation capability")
    print()
    print("Key Features:")
    print("  ✓ 8 distinct colormaps for different data types")
    print("  ✓ Automatic data normalization")
    print("  ✓ Multi-layer rendering support")
    print("  ✓ Gradient magnitude computation")
    print("  ✓ Difference detection between grids")
    print("  ✓ Real-time animation with configurable updates")
    print("  ✓ Flexible opacity and intensity controls")
    print()


if __name__ == "__main__":
    print()
    asyncio.run(main())

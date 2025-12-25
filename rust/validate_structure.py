"""
Validate Rust Clustering Structure
==================================

Checks that all required files exist and contain expected components.
Run without Rust/Cargo to verify project structure before building.

Usage:
    python rust/validate_structure.py
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Base directory
RUST_DIR = Path(__file__).parent


def check_file_exists(path: Path, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    if path.exists():
        return True, f"[OK] {description}: {path.name}"
    return False, f"[FAIL] {description}: {path.name} NOT FOUND"


def check_file_contains(path: Path, patterns: List[str], description: str) -> Tuple[bool, str]:
    """Check if a file contains expected patterns."""
    if not path.exists():
        return False, f"[FAIL] {description}: File not found"

    content = path.read_text(encoding='utf-8')
    missing = [p for p in patterns if not re.search(p, content)]

    if not missing:
        return True, f"[OK] {description}: All {len(patterns)} patterns found"
    return False, f"[FAIL] {description}: Missing patterns: {missing}"


def validate():
    """Run all validations."""
    print("=" * 60)
    print("Rust Clustering Structure Validation")
    print("=" * 60)

    results = []
    all_pass = True

    # =====================================================
    # Check workspace structure
    # =====================================================
    print("\n[Workspace Structure]")
    print("-" * 40)

    workspace_files = [
        (RUST_DIR / "Cargo.toml", "Workspace Cargo.toml"),
        (RUST_DIR / "hololoom-core" / "Cargo.toml", "Core crate Cargo.toml"),
        (RUST_DIR / "hololoom-python" / "Cargo.toml", "Python bindings Cargo.toml"),
        (RUST_DIR / "hololoom-wasm" / "Cargo.toml", "WASM bindings Cargo.toml"),
    ]

    for path, desc in workspace_files:
        passed, msg = check_file_exists(path, desc)
        print(f"  {msg}")
        all_pass &= passed

    # =====================================================
    # Check hololoom-core
    # =====================================================
    print("\n[hololoom-core (Algorithms)]")
    print("-" * 40)

    core_files = [
        (RUST_DIR / "hololoom-core" / "src" / "lib.rs", "Core lib.rs"),
        (RUST_DIR / "hololoom-core" / "src" / "vector_ops.rs", "Vector operations"),
        (RUST_DIR / "hololoom-core" / "src" / "clustering.rs", "Clustering algorithms"),
        (RUST_DIR / "hololoom-core" / "src" / "simd" / "mod.rs", "SIMD module"),
        (RUST_DIR / "hololoom-core" / "src" / "simd" / "avx2.rs", "AVX2 implementation"),
    ]

    for path, desc in core_files:
        passed, msg = check_file_exists(path, desc)
        print(f"  {msg}")
        all_pass &= passed

    # Check lib.rs exports
    lib_rs = RUST_DIR / "hololoom-core" / "src" / "lib.rs"
    exports = [
        r"pub use vector_ops",
        r"cosine_similarity_batch",
        r"normalize_batch_flat",
        r"pub use clustering",
        r"kmeans",
        r"silhouette_score",
    ]
    passed, msg = check_file_contains(lib_rs, exports, "Core exports")
    print(f"  {msg}")
    all_pass &= passed

    # Check SIMD dispatch in vector_ops
    vector_ops = RUST_DIR / "hololoom-core" / "src" / "vector_ops.rs"
    simd_patterns = [
        r"is_x86_feature_detected!\(\"avx2\"\)",
        r"cosine_similarity_batch_avx2",
        r"normalize_batch_avx2",
    ]
    passed, msg = check_file_contains(vector_ops, simd_patterns, "SIMD dispatch")
    print(f"  {msg}")
    all_pass &= passed

    # Check Rayon parallelization in clustering
    clustering = RUST_DIR / "hololoom-core" / "src" / "clustering.rs"
    rayon_patterns = [
        r"use rayon::prelude::\*",
        r"par_iter",  # Matches both .par_iter() and .into_par_iter()
    ]
    passed, msg = check_file_contains(clustering, rayon_patterns, "Rayon parallelization")
    print(f"  {msg}")
    all_pass &= passed

    # =====================================================
    # Check hololoom-python
    # =====================================================
    print("\n[hololoom-python (PyO3 Bindings)]")
    print("-" * 40)

    python_files = [
        (RUST_DIR / "hololoom-python" / "src" / "lib.rs", "Python bindings lib.rs"),
        (RUST_DIR / "hololoom-python" / "pyproject.toml", "Maturin config"),
    ]

    for path, desc in python_files:
        passed, msg = check_file_exists(path, desc)
        print(f"  {msg}")
        all_pass &= passed

    # Check PyO3 function exports
    python_lib = RUST_DIR / "hololoom-python" / "src" / "lib.rs"
    pyo3_patterns = [
        r'#\[pyo3\(name = "cosine_similarity_batch"\)\]',
        r'#\[pyo3\(name = "normalize_batch"\)\]',
        r'#\[pyo3\(name = "kmeans"',
        r'#\[pyo3\(name = "silhouette_score"\)\]',
        r"fn hololoom_rust",
    ]
    passed, msg = check_file_contains(python_lib, pyo3_patterns, "PyO3 exports")
    print(f"  {msg}")
    all_pass &= passed

    # =====================================================
    # Check hololoom-wasm
    # =====================================================
    print("\n[hololoom-wasm (WASM Bindings)]")
    print("-" * 40)

    wasm_files = [
        (RUST_DIR / "hololoom-wasm" / "src" / "lib.rs", "WASM bindings lib.rs"),
    ]

    for path, desc in wasm_files:
        passed, msg = check_file_exists(path, desc)
        print(f"  {msg}")
        all_pass &= passed

    # Check wasm-bindgen exports
    wasm_lib = RUST_DIR / "hololoom-wasm" / "src" / "lib.rs"
    wasm_patterns = [
        r"#\[wasm_bindgen\]",
        r"pub fn cosine_similarity_batch",
        r"pub fn normalize_batch",
        r"pub fn kmeans",
        r"pub fn silhouette_score",
        r"KMeansResultJS",
    ]
    passed, msg = check_file_contains(wasm_lib, wasm_patterns, "WASM exports")
    print(f"  {msg}")
    all_pass &= passed

    # =====================================================
    # Check CI/CD
    # =====================================================
    print("\n[CI/CD Configuration]")
    print("-" * 40)

    ci_file = RUST_DIR.parent / ".github" / "workflows" / "rust-clustering.yml"
    passed, msg = check_file_exists(ci_file, "GitHub Actions workflow")
    print(f"  {msg}")
    all_pass &= passed

    if ci_file.exists():
        ci_patterns = [
            r"maturin",
            r"wasm-pack",
            r"cargo test",
            r"ubuntu-latest",
            r"windows-latest",
        ]
        passed, msg = check_file_contains(ci_file, ci_patterns, "CI coverage")
        print(f"  {msg}")
        all_pass &= passed

    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "=" * 60)
    if all_pass:
        print("[SUCCESS] ALL VALIDATIONS PASSED")
        print("\nNext steps:")
        print("  1. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        print("  2. Build Python package:")
        print("     cd rust/hololoom-python")
        print("     pip install maturin")
        print("     maturin develop --release")
        print("  3. Run benchmarks:")
        print("     python demos/benchmark_rust_clustering.py")
    else:
        print("[FAILED] SOME VALIDATIONS FAILED")
        print("   Fix the issues above before building.")

    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys
    success = validate()
    sys.exit(0 if success else 1)

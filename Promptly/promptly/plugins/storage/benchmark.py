#!/usr/bin/env python3
"""
Storage Backend Performance Benchmark

Compare performance characteristics of different storage backends.

Usage:
    python -m Promptly.promptly.plugins.storage.benchmark \\
        --backends sqlite postgresql mongodb redis \\
        --operations 1000 \\
        --output benchmark_results.json

    # Or use as a library
    from Promptly.promptly.plugins.storage.benchmark import run_benchmark

    results = run_benchmark(
        backends=['sqlite', 'mongodb', 'redis'],
        num_operations=1000
    )
"""

import argparse
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

try:
    from . import create_storage_backend, get_available_backends
except ImportError:
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
    from Promptly.promptly.plugins.storage import create_storage_backend, get_available_backends


class BenchmarkRunner:
    """
    Performance benchmark for storage backends

    Tests:
    - Write performance (save_prompt)
    - Read performance (get_prompt)
    - List performance (list_prompts)
    - Branch operations (create_branch, switch_branch)
    - Concurrent operations (if supported)
    """

    def __init__(self, backend_type: str, config: Optional[Dict] = None):
        """
        Initialize benchmark for a backend

        Args:
            backend_type: Backend type ('sqlite', 'postgresql', etc.)
            config: Backend-specific configuration
        """
        self.backend_type = backend_type
        self.config = config or {}
        self.backend = None
        self.temp_dir = None
        self.results = {
            'backend': backend_type,
            'timestamp': datetime.utcnow().isoformat(),
            'operations': {},
            'errors': []
        }

    def setup(self) -> None:
        """Setup backend for benchmarking"""
        # Create temporary directory for file-based backends
        self.temp_dir = tempfile.mkdtemp(prefix=f'promptly_bench_{self.backend_type}_')

        # Get storage path based on backend type
        if self.backend_type == 'sqlite':
            storage_path = str(Path(self.temp_dir) / 'benchmark.db')
        elif self.backend_type == 'json':
            storage_path = str(Path(self.temp_dir) / 'data')
        elif self.backend_type == 'git':
            storage_path = str(Path(self.temp_dir) / 'repo')
        elif self.backend_type == 'postgresql':
            # Use test database (must be pre-configured)
            storage_path = self.config.get(
                'connection_string',
                'postgresql://promptly:promptly@localhost/promptly_benchmark'
            )
        elif self.backend_type == 'mongodb':
            storage_path = self.config.get(
                'connection_string',
                'mongodb://localhost:27017/promptly_benchmark'
            )
        elif self.backend_type == 'redis':
            storage_path = self.config.get(
                'connection_string',
                'redis://localhost:6379/15'  # Use high DB number for tests
            )
        else:
            storage_path = str(Path(self.temp_dir) / 'data')

        # Create backend
        self.backend = create_storage_backend(self.backend_type, **self.config)
        self.backend.init_storage(storage_path)

    def cleanup(self) -> None:
        """Cleanup after benchmarking"""
        if self.backend:
            try:
                # Cleanup backend-specific data
                if self.backend_type == 'redis':
                    # Flush test database
                    try:
                        self.backend.flush_database(confirm=True)
                    except:
                        pass
                elif self.backend_type in ['postgresql', 'mongodb']:
                    # Note: Manual cleanup may be needed for remote databases
                    pass

                self.backend.close()
            except:
                pass

        # Remove temporary directory
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def benchmark_writes(self, num_operations: int) -> Dict[str, float]:
        """
        Benchmark write performance

        Args:
            num_operations: Number of prompts to write

        Returns:
            Dict with timing statistics
        """
        print(f"  Running write benchmark ({num_operations} operations)...")

        times = []
        errors = 0

        for i in range(num_operations):
            prompt_data = {
                'name': f'benchmark_prompt_{i}',
                'content': f'This is benchmark prompt {i} with some content for testing.',
                'branch': 'main',
                'metadata': {
                    'benchmark': True,
                    'index': i,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }

            try:
                start = time.perf_counter()
                self.backend.save_prompt(prompt_data)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                errors += 1
                self.results['errors'].append(f"Write error: {e}")

        return self._calculate_stats(times, errors)

    def benchmark_reads(self, num_operations: int) -> Dict[str, float]:
        """
        Benchmark read performance

        Args:
            num_operations: Number of prompts to read

        Returns:
            Dict with timing statistics
        """
        print(f"  Running read benchmark ({num_operations} operations)...")

        # First create some prompts
        for i in range(min(num_operations, 100)):  # Create up to 100 prompts
            self.backend.save_prompt({
                'name': f'read_test_{i}',
                'content': f'Content {i}',
                'branch': 'main'
            })

        times = []
        errors = 0

        for i in range(num_operations):
            prompt_name = f'read_test_{i % 100}'  # Cycle through available prompts

            try:
                start = time.perf_counter()
                self.backend.get_prompt(prompt_name, branch='main')
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                errors += 1
                self.results['errors'].append(f"Read error: {e}")

        return self._calculate_stats(times, errors)

    def benchmark_list(self, num_prompts: int) -> Dict[str, float]:
        """
        Benchmark list performance

        Args:
            num_prompts: Number of prompts to create before listing

        Returns:
            Dict with timing statistics
        """
        print(f"  Running list benchmark ({num_prompts} prompts)...")

        # Create prompts
        for i in range(num_prompts):
            self.backend.save_prompt({
                'name': f'list_test_{i}',
                'content': f'Content {i}',
                'branch': 'main'
            })

        times = []
        errors = 0

        # Run list operations
        for _ in range(10):  # Run 10 times to get average
            try:
                start = time.perf_counter()
                prompts = self.backend.list_prompts(branch='main')
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                errors += 1
                self.results['errors'].append(f"List error: {e}")

        return self._calculate_stats(times, errors)

    def benchmark_branches(self) -> Dict[str, float]:
        """
        Benchmark branch operations

        Returns:
            Dict with timing statistics
        """
        print(f"  Running branch benchmark...")

        times = []
        errors = 0

        # Create some prompts on main
        for i in range(10):
            self.backend.save_prompt({
                'name': f'branch_test_{i}',
                'content': f'Content {i}',
                'branch': 'main'
            })

        # Benchmark branch creation
        for i in range(5):
            try:
                start = time.perf_counter()
                self.backend.create_branch(f'test_branch_{i}', from_branch='main')
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                errors += 1
                self.results['errors'].append(f"Branch creation error: {e}")

        # Benchmark branch switching
        for i in range(5):
            try:
                start = time.perf_counter()
                self.backend.set_current_branch(f'test_branch_{i}')
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                errors += 1
                self.results['errors'].append(f"Branch switch error: {e}")

        return self._calculate_stats(times, errors)

    def _calculate_stats(self, times: List[float], errors: int) -> Dict[str, float]:
        """Calculate timing statistics"""
        if not times:
            return {
                'count': 0,
                'errors': errors,
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'total': 0,
                'ops_per_sec': 0
            }

        times_sorted = sorted(times)
        count = len(times)
        total = sum(times)
        mean = total / count
        median = times_sorted[count // 2]

        return {
            'count': count,
            'errors': errors,
            'mean': mean * 1000,  # Convert to ms
            'median': median * 1000,
            'min': min(times) * 1000,
            'max': max(times) * 1000,
            'p95': times_sorted[int(count * 0.95)] * 1000,
            'p99': times_sorted[int(count * 0.99)] * 1000,
            'total': total,
            'ops_per_sec': count / total if total > 0 else 0
        }

    def run(self, num_operations: int = 100) -> Dict[str, Any]:
        """
        Run full benchmark suite

        Args:
            num_operations: Number of operations for each test

        Returns:
            Dict with all benchmark results
        """
        print(f"\nBenchmarking {self.backend_type}...")

        try:
            self.setup()

            # Run benchmarks
            self.results['operations']['writes'] = self.benchmark_writes(num_operations)
            self.results['operations']['reads'] = self.benchmark_reads(num_operations)
            self.results['operations']['list'] = self.benchmark_list(min(num_operations, 100))
            self.results['operations']['branches'] = self.benchmark_branches()

            # Get backend statistics if available
            if hasattr(self.backend, 'get_statistics'):
                try:
                    self.results['statistics'] = self.backend.get_statistics()
                except:
                    pass

        except Exception as e:
            self.results['errors'].append(f"Benchmark failed: {e}")
            print(f"  ERROR: {e}")

        finally:
            self.cleanup()

        return self.results


def run_benchmark(backends: List[str],
                 num_operations: int = 100,
                 backend_configs: Optional[Dict[str, Dict]] = None,
                 output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run benchmarks on multiple backends

    Args:
        backends: List of backend types to benchmark
        num_operations: Number of operations for each test
        backend_configs: Optional dict of backend-specific configs
        output_file: Optional file to write results to

    Returns:
        Dict with all benchmark results

    Example:
        results = run_benchmark(
            backends=['sqlite', 'postgresql', 'redis'],
            num_operations=1000,
            backend_configs={
                'postgresql': {
                    'pool_size': 10
                }
            }
        )
    """
    backend_configs = backend_configs or {}
    available = get_available_backends()

    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'num_operations': num_operations,
        'backends': {},
        'summary': {}
    }

    # Run benchmarks
    for backend_type in backends:
        if not available.get(backend_type):
            print(f"\nSkipping {backend_type} (not available)")
            continue

        config = backend_configs.get(backend_type, {})
        runner = BenchmarkRunner(backend_type, config)

        try:
            backend_results = runner.run(num_operations)
            results['backends'][backend_type] = backend_results
        except Exception as e:
            print(f"Error benchmarking {backend_type}: {e}")
            results['backends'][backend_type] = {
                'error': str(e)
            }

    # Generate summary
    results['summary'] = _generate_summary(results['backends'])

    # Write to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults written to {output_file}")

    return results


def _generate_summary(backend_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate performance summary comparing backends"""
    summary = {
        'fastest_write': None,
        'fastest_read': None,
        'fastest_list': None,
        'comparison': {}
    }

    # Find fastest for each operation
    write_speeds = {}
    read_speeds = {}
    list_speeds = {}

    for backend, results in backend_results.items():
        if 'error' in results:
            continue

        ops = results.get('operations', {})

        if 'writes' in ops and ops['writes']['ops_per_sec'] > 0:
            write_speeds[backend] = ops['writes']['ops_per_sec']

        if 'reads' in ops and ops['reads']['ops_per_sec'] > 0:
            read_speeds[backend] = ops['reads']['ops_per_sec']

        if 'list' in ops and ops['list']['ops_per_sec'] > 0:
            list_speeds[backend] = ops['list']['ops_per_sec']

    if write_speeds:
        summary['fastest_write'] = max(write_speeds.items(), key=lambda x: x[1])

    if read_speeds:
        summary['fastest_read'] = max(read_speeds.items(), key=lambda x: x[1])

    if list_speeds:
        summary['fastest_list'] = max(list_speeds.items(), key=lambda x: x[1])

    # Create comparison table
    summary['comparison'] = {
        'write_ops_per_sec': write_speeds,
        'read_ops_per_sec': read_speeds,
        'list_ops_per_sec': list_speeds
    }

    return summary


def print_results(results: Dict[str, Any]) -> None:
    """Pretty print benchmark results"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Operations per test: {results['num_operations']}")
    print()

    # Print results for each backend
    for backend, backend_results in results['backends'].items():
        print(f"\n{backend.upper()}")
        print("-" * 80)

        if 'error' in backend_results:
            print(f"  ERROR: {backend_results['error']}")
            continue

        ops = backend_results.get('operations', {})

        for op_name, op_results in ops.items():
            print(f"\n  {op_name.upper()}:")
            print(f"    Operations: {op_results['count']}")
            print(f"    Errors: {op_results['errors']}")
            print(f"    Mean: {op_results['mean']:.3f} ms")
            print(f"    Median: {op_results['median']:.3f} ms")
            print(f"    Min: {op_results['min']:.3f} ms")
            print(f"    Max: {op_results['max']:.3f} ms")
            print(f"    P95: {op_results['p95']:.3f} ms")
            print(f"    P99: {op_results['p99']:.3f} ms")
            print(f"    Throughput: {op_results['ops_per_sec']:.2f} ops/sec")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    summary = results.get('summary', {})

    if summary.get('fastest_write'):
        backend, speed = summary['fastest_write']
        print(f"\nFastest Write: {backend} ({speed:.2f} ops/sec)")

    if summary.get('fastest_read'):
        backend, speed = summary['fastest_read']
        print(f"Fastest Read: {backend} ({speed:.2f} ops/sec)")

    if summary.get('fastest_list'):
        backend, speed = summary['fastest_list']
        print(f"Fastest List: {backend} ({speed:.2f} ops/sec)")

    # Print comparison table
    comparison = summary.get('comparison', {})
    if comparison:
        print("\nPerformance Comparison (ops/sec):")
        print("-" * 80)

        for metric, speeds in comparison.items():
            if speeds:
                print(f"\n{metric}:")
                for backend, speed in sorted(speeds.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {backend:12s}: {speed:10.2f}")


def main():
    """Command-line interface for benchmark tool"""
    parser = argparse.ArgumentParser(
        description='Benchmark Promptly storage backends',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all available backends
  python benchmark.py --backends sqlite json postgresql mongodb redis git

  # Benchmark specific backends with more operations
  python benchmark.py --backends sqlite redis --operations 5000

  # Save results to file
  python benchmark.py --backends sqlite mongodb --output results.json

Note: Remote backends (PostgreSQL, MongoDB, Redis) must be running
      and accessible with default connection settings.
        """
    )

    parser.add_argument('--backends', nargs='+',
                       default=['sqlite', 'json'],
                       help='Backends to benchmark (default: sqlite json)')
    parser.add_argument('--operations', type=int, default=100,
                       help='Number of operations per test (default: 100)')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON format)')

    args = parser.parse_args()

    # Show available backends
    available = get_available_backends()
    print("Available backends:")
    for backend, is_available in available.items():
        status = "✓" if is_available else "✗ (missing dependencies)"
        print(f"  {backend:12s}: {status}")

    # Run benchmarks
    try:
        results = run_benchmark(
            backends=args.backends,
            num_operations=args.operations,
            output_file=args.output
        )

        print_results(results)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

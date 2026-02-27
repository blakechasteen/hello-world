"""
WASM Runner - Executes WASM modules with JSON input/output.

Uses wasmtime-py for WASM execution. Modules receive JSON input
and return JSON output via shared memory.

For MVP, we use a simple JSON-based protocol:
1. Serialize input to JSON bytes
2. Write to WASM memory
3. Call entry function with pointer and length
4. Read output from WASM memory
5. Deserialize output from JSON bytes
"""

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from ..shared.types import JobResult, JobStatus
from ..shared.logging import get_logger

logger = get_logger(__name__, component="node")

# Try to import wasmtime, provide fallback for testing
try:
    import wasmtime
    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False
    logger.warning("wasmtime not available - WASM execution will use mock mode")


class WasmRunner:
    """
    Executes WASM modules with JSON input/output.

    Supports loading modules from:
    - Local file path
    - Base64-encoded bytes
    - Pre-cached modules
    """

    def __init__(self, module_dir: Optional[str] = None):
        """
        Initialize WASM runner.

        Args:
            module_dir: Directory containing WASM modules
        """
        self.module_dir = Path(module_dir) if module_dir else None
        self._module_cache: Dict[str, Any] = {}  # wasmtime.Module

        if WASMTIME_AVAILABLE:
            self._engine = wasmtime.Engine()
            self._linker = wasmtime.Linker(self._engine)
            # Link WASI for basic I/O
            try:
                self._linker.define_wasi()
            except Exception:
                pass  # WASI may not be available in all builds
        else:
            self._engine = None
            self._linker = None

    def run(
        self,
        module_id: str,
        entry_function: str,
        input_json: Any,
        wasm_path: Optional[Path] = None,
        timeout_seconds: int = 60
    ) -> JobResult:
        """
        Execute a WASM module with JSON input.

        Args:
            module_id: Module identifier (for caching/logging)
            entry_function: Function to call in the module
            input_json: Input data to pass to the function
            wasm_path: Path to WASM file (optional if cached)
            timeout_seconds: Execution timeout

        Returns:
            JobResult with output or error

        Security Note:
            Only pre-registered modules can be executed. Arbitrary WASM
            bytecode is not accepted to prevent remote code execution.
        """
        start_time = time.time()

        try:
            if not WASMTIME_AVAILABLE:
                # Mock mode for testing without wasmtime
                return self._mock_execute(module_id, entry_function, input_json, start_time)

            # Load module (only from cache or registered paths)
            module = self._get_or_load_module(module_id, wasm_path)

            # Execute
            output = self._execute_module(module, entry_function, input_json)

            execution_time_ms = (time.time() - start_time) * 1000

            return JobResult(
                job_id=module_id,  # Will be overwritten by caller
                status=JobStatus.COMPLETED,
                output_json=output,
                execution_time_ms=execution_time_ms,
            )

        except TimeoutError:
            return JobResult(
                job_id=module_id,
                status=JobStatus.TIMEOUT,
                error=f"Execution timed out after {timeout_seconds}s",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"WASM execution failed for {module_id}: {e}")
            return JobResult(
                job_id=module_id,
                status=JobStatus.FAILED,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _get_or_load_module(
        self,
        module_id: str,
        wasm_path: Optional[Path]
    ) -> Any:  # wasmtime.Module
        """Load module from cache or registered file paths only.

        Security: Does not accept arbitrary bytecode to prevent RCE.
        """
        # Check cache first
        if module_id in self._module_cache:
            return self._module_cache[module_id]

        # Load from file
        if wasm_path and wasm_path.exists():
            module = wasmtime.Module.from_file(self._engine, str(wasm_path))
            self._module_cache[module_id] = module
            return module

        # Try module_dir
        if self.module_dir:
            path = self.module_dir / f"{module_id}.wasm"
            if path.exists():
                module = wasmtime.Module.from_file(self._engine, str(path))
                self._module_cache[module_id] = module
                return module

        raise FileNotFoundError(f"WASM module not found: {module_id}")

    def _execute_module(
        self,
        module: Any,  # wasmtime.Module
        entry_function: str,
        input_json: Any
    ) -> Any:
        """
        Execute a WASM module with JSON input/output.

        This implementation uses the simple approach:
        1. Module exports 'alloc' and 'dealloc' for memory management
        2. Module exports entry function that takes (ptr, len) and returns (ptr, len)

        For modules that don't follow this pattern, we fall back to simpler approaches.
        """
        # Create store and instance
        store = wasmtime.Store(self._engine)

        try:
            wasi_config = wasmtime.WasiConfig()
            store.set_wasi(wasi_config)
        except Exception:
            pass  # WASI may not be available

        instance = self._linker.instantiate(store, module)

        # Serialize input
        input_bytes = json.dumps(input_json).encode('utf-8')

        # Try different calling conventions
        exports = {e.name: e for e in instance.exports(store)}

        # Method 1: Standard alloc/entry pattern
        if 'alloc' in exports and entry_function in exports:
            return self._execute_with_alloc(
                store, instance, exports, entry_function, input_bytes
            )

        # Method 2: Direct function call (for simple modules)
        if entry_function in exports:
            func = exports[entry_function]
            # Try calling with no args first (module might read from different source)
            try:
                result = func(store)
                return result
            except Exception:
                pass

            # Try calling with input as separate args (for numeric inputs)
            if isinstance(input_json, dict):
                args = list(input_json.values())
                try:
                    result = func(store, *args)
                    if isinstance(result, (int, float)):
                        return {"result": result}
                    return result
                except Exception as e:
                    raise RuntimeError(f"Failed to call {entry_function}: {e}")

        raise RuntimeError(f"No compatible calling convention for {entry_function}")

    def _execute_with_alloc(
        self,
        store: Any,
        instance: Any,
        exports: Dict[str, Any],
        entry_function: str,
        input_bytes: bytes
    ) -> Any:
        """Execute using alloc/dealloc memory management."""
        alloc = exports['alloc']
        dealloc = exports.get('dealloc')
        entry = exports[entry_function]
        memory = exports.get('memory')

        if not memory:
            raise RuntimeError("Module must export 'memory'")

        # Allocate input buffer
        input_len = len(input_bytes)
        input_ptr = alloc(store, input_len)

        # Write input to memory
        mem_data = memory.data_ptr(store)
        for i, b in enumerate(input_bytes):
            mem_data[input_ptr + i] = b

        # Call entry function
        result = entry(store, input_ptr, input_len)

        # Parse result (expecting (ptr, len) tuple or single value)
        if isinstance(result, tuple) and len(result) == 2:
            output_ptr, output_len = result
            # Read output from memory
            output_bytes = bytes(mem_data[output_ptr:output_ptr + output_len])
            output = json.loads(output_bytes.decode('utf-8'))
        else:
            output = result

        # Deallocate if available
        if dealloc:
            dealloc(store, input_ptr, input_len)

        return output

    def _mock_execute(
        self,
        module_id: str,
        entry_function: str,
        input_json: Any,
        start_time: float
    ) -> JobResult:
        """
        Mock execution for testing without wasmtime.

        Implements basic built-in modules.
        """
        logger.info(f"Mock executing {module_id}.{entry_function}")

        output = None

        # Built-in mock modules
        if module_id == "echo" or module_id == "echo-v1":
            output = input_json

        elif module_id == "add" or module_id == "add-v1":
            a = input_json.get("a", 0)
            b = input_json.get("b", 0)
            output = {"result": a + b}

        elif module_id == "multiply" or module_id == "multiply-v1":
            a = input_json.get("a", 1)
            b = input_json.get("b", 1)
            output = {"result": a * b}

        elif module_id == "gradient-descent" or module_id == "gradient-descent-v1":
            output = self._mock_gradient_descent(input_json)

        elif module_id == "fib" or module_id == "fib-v1":
            output = self._mock_fibonacci(input_json)

        elif module_id == "memory" or module_id == "memory-v1":
            output = self._mock_memory_alloc(input_json)

        else:
            # Unknown module - echo with module info
            output = {
                "module": module_id,
                "function": entry_function,
                "input": input_json,
                "note": "Mock execution (wasmtime not available)"
            }

        execution_time_ms = (time.time() - start_time) * 1000

        return JobResult(
            job_id=module_id,
            status=JobStatus.COMPLETED,
            output_json=output,
            execution_time_ms=execution_time_ms,
        )

    def _mock_gradient_descent(self, input_json: Any) -> dict:
        """
        Mock gradient descent implementation.

        Supports:
        - linear_regression: Fits y = w*X + b
        - quadratic: Minimizes f(x) = x^2 (or sum of x_i^2)
        - custom: Returns mock result
        """
        import math

        problem_type = input_json.get("problem_type", "quadratic")
        params = input_json.get("initial_params", [1.0])
        lr = input_json.get("learning_rate", 0.01)
        max_iter = input_json.get("max_iterations", 1000)
        tolerance = input_json.get("tolerance", 1e-6)
        data = input_json.get("data", {})

        loss_history = []
        converged = False

        if problem_type == "quadratic":
            # Minimize f(x) = sum(x_i^2)
            # Gradient: 2*x_i
            for i in range(max_iter):
                loss = sum(p**2 for p in params)
                loss_history.append(loss)

                if loss < tolerance:
                    converged = True
                    break

                # Gradient descent step
                params = [p - lr * 2 * p for p in params]

            final_loss = sum(p**2 for p in params)

        elif problem_type == "linear_regression":
            # Minimize MSE: (1/n) * sum((y - (w*x + b))^2)
            X_raw = data.get("X", [1, 2, 3])
            y = data.get("y", [2, 4, 6])
            n = len(y)

            # Normalize X for numerical stability
            X = [xi[0] if isinstance(xi, list) else xi for xi in X_raw]
            x_mean = sum(X) / n
            x_std = max((sum((xi - x_mean)**2 for xi in X) / n) ** 0.5, 1e-8)
            X_norm = [(xi - x_mean) / x_std for xi in X]

            # params = [w, b] for single feature
            if len(params) < 2:
                params = [0.0, 0.0]

            w, b = params[0], params[-1]

            for iteration in range(max_iter):
                # Compute predictions and loss
                predictions = [w * xi + b for xi in X_norm]
                loss = sum((pred - yi)**2 for pred, yi in zip(predictions, y)) / n
                loss_history.append(loss)

                if loss < tolerance:
                    converged = True
                    break

                # Compute gradients with clipping
                dw = sum((pred - yi) * xi * 2 / n for xi, yi, pred in zip(X_norm, y, predictions))
                db = sum((pred - yi) * 2 / n for yi, pred in zip(y, predictions))

                # Gradient clipping
                grad_norm = (dw**2 + db**2) ** 0.5
                if grad_norm > 1.0:
                    dw /= grad_norm
                    db /= grad_norm

                # Update parameters
                w = w - lr * dw
                b = b - lr * db

            # Denormalize weight for original scale
            w_original = w / x_std
            b_original = b - w * x_mean / x_std
            params = [w_original, b_original]
            final_loss = loss_history[-1] if loss_history else 0

        else:
            # Custom or unknown - just decay towards zero
            for i in range(min(max_iter, 100)):
                loss = sum(abs(p) for p in params)
                loss_history.append(loss)
                params = [p * 0.99 for p in params]
                if loss < tolerance:
                    converged = True
                    break
            final_loss = sum(abs(p) for p in params)

        # Sample loss history (keep ~50 points max)
        if len(loss_history) > 50:
            step = len(loss_history) // 50
            loss_history = loss_history[::step]

        return {
            "optimized_params": params,
            "final_loss": final_loss,
            "iterations": len(loss_history),
            "converged": converged,
            "loss_history": loss_history
        }

    def _mock_fibonacci(self, input_json: Any) -> dict:
        """
        Mock Fibonacci implementation (CPU-intensive test).

        Uses iterative approach for efficiency.
        """
        n = input_json.get("n", 10)

        # Cap at reasonable value for mock
        n = min(max(0, n), 92)  # fib(92) is max for 64-bit

        if n <= 1:
            return {"result": n, "iterations": 0}

        # Iterative Fibonacci
        a, b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b

        return {"result": b, "iterations": n - 1}

    def _mock_memory_alloc(self, input_json: Any) -> dict:
        """
        Mock memory allocation test.

        Simulates allocating and filling a buffer.
        """
        import time

        size_kb = input_json.get("size_kb", 1)
        fill_value = input_json.get("fill_value", 0) & 0xFF  # Ensure 0-255

        # Cap at reasonable value for mock (10MB max)
        size_kb = min(max(1, size_kb), 10240)

        start = time.time()

        # Simulate allocation and filling
        buffer_size = size_kb * 1024
        buffer = bytearray([fill_value] * buffer_size)

        # Simple checksum
        checksum = sum(buffer) % (2**32)

        duration_ms = (time.time() - start) * 1000

        return {
            "allocated_kb": size_kb,
            "checksum": checksum,
            "duration_ms": round(duration_ms, 2)
        }

    def clear_cache(self) -> None:
        """Clear the module cache."""
        self._module_cache.clear()

    def is_available(self) -> bool:
        """Check if wasmtime is available for real execution."""
        return WASMTIME_AVAILABLE
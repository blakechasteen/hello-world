# WASM Modules Directory

This directory holds WASM modules for execution by node daemons.

## Available Modules

| Module ID | Name | Description | Mock Mode |
|-----------|------|-------------|-----------|
| `add-v1` | Simple Adder | Adds two numbers | Yes |
| `echo-v1` | Echo | Returns input unchanged | Yes |
| `multiply-v1` | Multiplier | Multiplies two numbers | Yes |
| `fib-v1` | Fibonacci | Calculates nth Fibonacci (CPU test) | Yes |
| `memory-v1` | Memory Test | Allocates memory buffer (memory test) | Yes |
| `gradient-descent-v1` | Gradient Descent | ML optimization | Yes |

## Mock Mode

When wasmtime is not installed, nodes run in mock mode with built-in implementations:

- `add` / `add-v1`: Adds a + b
- `echo` / `echo-v1`: Returns input unchanged
- `multiply` / `multiply-v1`: Multiplies a * b
- `fib` / `fib-v1`: Calculates Fibonacci number (CPU stress test)
- `memory` / `memory-v1`: Allocates and fills memory buffer
- `gradient-descent` / `gradient-descent-v1`: Runs gradient descent optimization

## Directory Structure

```
wasm_modules/
├── README.md           # This file
├── add.json            # Add module manifest
├── echo.json           # Echo module manifest
├── multiply.json       # Multiply module manifest
├── fib.json            # Fibonacci module manifest
├── memory.json         # Memory test module manifest
├── gradient_descent.json  # Gradient descent manifest
└── src/                # Source code for building real modules
    ├── rust/           # Rust implementations
    └── assemblyscript/ # AssemblyScript implementations
```

## Manifest Format

Each module requires a JSON manifest:

```json
{
  "id": "my-module-v1",
  "name": "My Module",
  "version": "1.0.0",
  "entry_function": "run",
  "description": "What this module does",
  "input_schema": {
    "type": "object",
    "properties": {
      "param1": {"type": "number", "description": "First parameter"}
    },
    "required": ["param1"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "result": {"type": "number", "description": "Output value"}
    }
  }
}
```

## Building Real WASM Modules

### Option 1: Rust (Recommended)

**Prerequisites:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target
rustup target add wasm32-wasi
```

**Example: Simple Add Module**

Create `src/rust/add/Cargo.toml`:
```toml
[package]
name = "add"
version = "1.0.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

Create `src/rust/add/src/lib.rs`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    a: f64,
    b: f64,
}

#[derive(Serialize)]
struct Output {
    result: f64,
}

#[no_mangle]
pub extern "C" fn run(input_ptr: *const u8, input_len: usize) -> (*const u8, usize) {
    let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
    let input: Input = serde_json::from_slice(input_slice).unwrap();

    let output = Output {
        result: input.a + input.b,
    };

    let output_json = serde_json::to_vec(&output).unwrap();
    let ptr = output_json.as_ptr();
    let len = output_json.len();
    std::mem::forget(output_json);  // Prevent deallocation

    (ptr, len)
}

// Memory allocation functions required by runner
#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, size: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, size);
    }
}
```

**Build:**
```bash
cd src/rust/add
cargo build --target wasm32-wasi --release
cp target/wasm32-wasi/release/add.wasm ../../add.wasm
```

### Option 2: AssemblyScript

**Prerequisites:**
```bash
npm install -g assemblyscript
```

**Example: Simple Add Module**

Create `src/assemblyscript/add/assembly/index.ts`:
```typescript
import { JSON } from "json-as/assembly";

@json
class Input {
  a: f64 = 0;
  b: f64 = 0;
}

@json
class Output {
  result: f64 = 0;
}

export function run(inputPtr: usize, inputLen: usize): usize {
  // Read input from memory
  const inputBytes = new Uint8Array(inputLen);
  for (let i = 0; i < inputLen; i++) {
    inputBytes[i] = load<u8>(inputPtr + i);
  }

  const inputStr = String.UTF8.decode(inputBytes.buffer);
  const input = JSON.parse<Input>(inputStr);

  // Compute result
  const output = new Output();
  output.result = input.a + input.b;

  // Write output to memory
  const outputStr = JSON.stringify(output);
  const outputBytes = String.UTF8.encode(outputStr);

  return outputBytes.byteLength;
}

// Memory exports
export function alloc(size: usize): usize {
  return heap.alloc(size);
}

export function dealloc(ptr: usize, size: usize): void {
  heap.free(ptr);
}
```

**Build:**
```bash
cd src/assemblyscript/add
asc assembly/index.ts -o ../../add.wasm --optimize --exportRuntime
```

### Option 3: TinyGo (Go to WASM)

**Prerequisites:**
```bash
# Install TinyGo
# See: https://tinygo.org/getting-started/install/
```

**Build:**
```bash
tinygo build -o add.wasm -target=wasi ./main.go
```

## Testing Modules

After building, test with the node daemon:

```bash
# Start services
docker-compose up -d

# Test the add module
curl -X POST http://localhost:9091/jobs \
  -H "Content-Type: application/json" \
  -H "X-Shared-Secret: portal-demo-secret-2025" \
  -d '{
    "job_id": "test-001",
    "module_id": "add-v1",
    "entry_function": "run",
    "input_json": {"a": 5, "b": 3},
    "timeout_seconds": 30
  }'

# Expected output: {"result": 8}
```

## Security Notes

1. **Module Registration**: Only pre-registered modules can be executed
2. **No Arbitrary Code**: The runner does not accept raw WASM bytecode
3. **Resource Limits**: Modules have memory and CPU time limits
4. **Sandboxing**: wasmtime provides WASM sandboxing by default

## Adding New Modules

1. Create the module manifest (`module-name.json`)
2. Build the WASM binary (`module-name.wasm`)
3. Place both files in this directory
4. Restart node daemons to pick up new modules

For mock-only modules (development/testing), add the mock implementation to
`hololoom/portal/node_daemon/wasm_runner.py` in the `_mock_execute()` method.

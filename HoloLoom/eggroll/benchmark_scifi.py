from HoloLoom.eggroll.architectures import get_model
import torch
import time
import os
import psutil

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def benchmark_models():
    print("🚀 Benchmarking Sci-Fi Architectures...")
    device = "cpu" # Simulating Frontier node standard execution for now
    if torch.cuda.is_available():
        device = "cuda"
        
    configs = [
        ("TRM (Shallow)", "trm", {"recur_depth": 4, "d_model": 128}),
        ("TRM (Deep)", "trm", {"recur_depth": 50, "d_model": 128}), # 50 layers!
        ("Liquid (Euler)", "liquid", {"d_model": 128, "reservoir_size": 256, "solver": "euler"}),
        ("Liquid (RK4)", "liquid", {"d_model": 128, "reservoir_size": 256, "solver": "rk4"}),
        ("Spiking (LIF)", "spiking", {"d_model": 128, "n_layers": 3}),
    ]
    
    inputs = torch.randint(0, 1000, (4, 64)).to(device) # Batch 4, Seq 64
    
    print(f"{'Model':<20} | {'Params':<10} | {'Time (ms)':<10} | {'Mem (MB)':<10}")
    print("-" * 60)
    
    for name, mtype, kwargs in configs:
        torch.manual_seed(42)
        model = get_model(mtype, vocab_size=1000, **kwargs).to(device)
        params = model.get_trainable_parameters()
        
        # Warmup
        try:
            _ = model(inputs)
        except Exception as e:
            print(f"{name:<20} | FAILED: {str(e)[:30]}")
            continue

        # Timed Run
        start_mem = measure_memory()
        torch.cuda.empty_cache() if device == "cuda" else None
        
        start_time = time.time()
        for _ in range(10): # 10 fwd passes
            loss = model(inputs).mean()
            loss.backward() # Test backward pass too!
        end_time = time.time()
        
        end_mem = measure_memory()
        delta_mem = end_mem - start_mem
        
        avg_time = (end_time - start_time) / 10 * 1000
        
        print(f"{name:<20} | {params:<10} | {avg_time:.2f}      | {delta_mem:.2f}")

if __name__ == "__main__":
    benchmark_models()

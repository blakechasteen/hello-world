# check_hololoom.py
import importlib, traceback, time

def check_module(name):
    try:
        t0 = time.time()
        mod = importlib.import_module(name)
        dt = time.time() - t0
        print(f"✅ {name} imported in {dt:.3f}s")
        return True
    except Exception as e:
        print(f"❌ {name} failed:\n{traceback.format_exc(limit=1)}")
        return False

def main():
    print("\n=== HoloLoom Health Check ===\n")
    modules = [
        "hololoom.orchestrator",
        "hololoom.foundations.dynamics",
        "hololoom.foundations.geometry",
        "hololoom.motif.base",
        "hololoom.embedding.spectral",
        "hololoom.memory.vectorstore",
        "hololoom.policy.unified",
        "hololoom.tools.executor",
    ]
    ok = all(check_module(m) for m in modules)
    print("\n✅ Base imports ok" if ok else "\n⚠️ Some modules failed")

    # Optional deeper test
    try:
        from hololoom.orchestrator import run_query
        out = run_query("health check test")
        print("\nSample output:\n", out)
    except Exception as e:
        print("\n⚠️ run_query failed:\n", e)

if __name__ == "__main__":
    main()
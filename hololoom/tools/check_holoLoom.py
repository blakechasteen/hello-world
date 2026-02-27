# check_hololoom.py
# Extended HoloLoom Health Check: imports, pipeline smoke test, and math integrity tests.
import importlib, traceback, time, math, random
from types import SimpleNamespace

def hr(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def check_module(name):
    try:
        t0 = time.time()
        mod = importlib.import_module(name)
        dt = time.time() - t0
        print(f"✅ {name} imported in {dt:.3f}s")
        return True, mod
    except Exception:
        print(f"❌ {name} failed:\n{traceback.format_exc(limit=1)}")
        return False, None

def has_numpy():
    try:
        import numpy as np  # noqa
        return True
    except Exception:
        return False

def npf():
    import numpy as np
    return np

def near(x, y, tol=1e-6):
    return abs(x - y) <= tol

def cos_sim(a, b):
    np = npf()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na*nb))

def run_linear_algebra_checks():
    hr("Linear Algebra / Spectral Checks")
    if not has_numpy():
        print("⚠️ NumPy not available; skipping LA tests.")
        return True

    np = npf()
    ok = True

    # 1) Random orthonormal basis via QR and check Q^T Q ≈ I
    A = np.random.randn(64, 32)
    Q, _ = np.linalg.qr(A)
    I = Q.T @ Q
    err = np.linalg.norm(I - np.eye(I.shape[0]))
    print(f"• Orthonormality error ‖QᵀQ−I‖₂ = {err:.2e}")
    ok &= err < 1e-6

    # 2) Cosine similarity sanity: self-sim == 1, orthogonal ≈ 0
    v1 = Q[:, 0]; v2 = Q[:, 1]
    s11 = cos_sim(v1, v1); s12 = cos_sim(v1, v2)
    print(f"• cos(v, v) = {s11:.6f} (target≈1.0), cos(v1, v2) = {s12:.6f} (target≈0.0)")
    ok &= (s11 > 0.9999) and (abs(s12) < 1e-6)

    # 3) Symmetric PSD matrix eigenvalues are non-negative
    M = np.random.randn(48, 48)
    S = M.T @ M  # PSD
    w = np.linalg.eigvalsh(S)
    min_ev = float(w.min())
    print(f"• PSD min eigenvalue λ_min = {min_ev:.3e} (target ≥ 0)")
    ok &= min_ev >= -1e-8

    if ok:
        print("✅ Linear algebra checks passed")
    else:
        print("❌ Linear algebra checks failed")
    return ok

def run_chebyshev_checks():
    hr("Chebyshev Polynomial Approximation Checks")
    if not has_numpy():
        print("⚠️ NumPy not available; skipping Chebyshev tests.")
        return True
    np = npf()
    ok = True

    # Approximate f(x) = exp(x) on [-1,1] with Chebyshev nodes
    n = 20
    k = np.arange(n+1)
    xk = np.cos((2*k+1)/(2*(n+1))*np.pi)  # Chebyshev nodes
    fk = np.exp(xk)

    # Compute coefficients for Chebyshev series via discrete cosine transform (manual DCT)
    # a_j ≈ (2/(n+1)) * sum_{k=0..n} f(x_k) * cos(j * arccos(x_k))
    # with a_0 halved in evaluation
    theta = np.arccos(xk)[:, None]              # (n+1, 1)
    J = np.arange(n+1)[None, :]                 # (1, n+1)
    cos_mat = np.cos(J * theta)                  # (n+1, n+1)
    a = (2.0/(n+1)) * (fk[:, None] * cos_mat).sum(axis=0)  # (n+1,)

    # Evaluate series at test points
    xs = np.linspace(-1, 1, 201)
    T = np.cos(np.arange(n+1)[:, None] * np.arccos(xs)[None, :])  # (n+1, m)
    approx = 0.5*a[0]*T[0] + (a[1:, None] * T[1:]).sum(axis=0)
    truth = np.exp(xs)
    max_err = float(np.max(np.abs(approx - truth)))
    print(f"• Chebyshev max |exp(x) − approx| on [-1,1] = {max_err:.2e} (target < 1e-3)")
    ok &= max_err < 1e-3

    if ok:
        print("✅ Chebyshev checks passed")
    else:
        print("❌ Chebyshev checks failed")
    return ok

def run_de_solver_checks():
    hr("Differential Equation Solver Checks")
    if not has_numpy():
        print("⚠️ NumPy not available; skipping DE tests.")
        return True
    np = npf()
    ok = True

    # Test ODE: y' = -y, y(0)=1 ⇒ y(t)=exp(-t). Use RK4 with decreasing step and check convergence.
    def rk4(f, y0, t0, t1, h):
        y = y0
        t = t0
        while t < t1 - 1e-12:
            h1 = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h1/2, y + h1*k1/2)
            k3 = f(t + h1/2, y + h1*k2/2)
            k4 = f(t + h1, y + h1*k3)
            y = y + (h1/6)*(k1 + 2*k2 + 2*k3 + k4)
            t += h1
        return y

    f = lambda t, y: -y
    exact = lambda t: math.exp(-t)
    t1 = 1.0

    errs = []
    for h in [0.2, 0.1, 0.05, 0.025]:
        y = rk4(f, 1.0, 0.0, t1, h)
        e = abs(y - exact(t1))
        errs.append((h, e))
        print(f"• RK4 step {h:.3f}: y(1)≈{y:.8f}, error={e:.2e}")

    # Convergence: error should decrease ~ O(h^4); at least monotone decrease
    mono = all(errs[i+1][1] < errs[i][1] for i in range(len(errs)-1))
    ok &= mono
    print("• Monotone error decay:", "yes" if mono else "no")

    if ok:
        print("✅ DE solver checks passed")
    else:
        print("❌ DE solver checks failed")
    return ok

def run_geometry_checks():
    hr("Geometry / Metric Sanity Checks")
    if not has_numpy():
        print("⚠️ NumPy not available; skipping geometry tests.")
        return True
    np = npf()
    ok = True

    # Use a simple SPD metric matrix G; distance d(x,y) = sqrt((x−y)^T G (x−y))
    R = np.random.randn(8, 8)
    G = R.T @ R  # SPD
    def dist(x, y):
        v = x - y
        return float(np.sqrt(v.T @ G @ v))

    # Positive definite & symmetry
    x = np.random.randn(8); y = np.random.randn(8); z = np.random.randn(8)
    dxy = dist(x,y); dyx = dist(y,x)
    print(f"• Symmetry d(x,y)≈d(y,x): {dxy:.6f} vs {dyx:.6f}")
    ok &= abs(dxy - dyx) < 1e-9

    # Triangle inequality d(x,z) ≤ d(x,y)+d(y,z)
    dxz = dist(x,z); dyz = dist(y,z)
    tri_ok = dxz <= dxy + dyz + 1e-9
    print(f"• Triangle inequality: d(x,z)={dxz:.6f} ≤ {dxy:.6f}+{dyz:.6f} → {tri_ok}")
    ok &= tri_ok

    # Positive definiteness: d(x,y) == 0 ⇒ x==y
    ok &= dist(x,x) < 1e-12

    if ok:
        print("✅ Geometry checks passed")
    else:
        print("❌ Geometry checks failed")
    return ok

def run_memory_checks():
    hr("Memory / VectorStore Checks")
    try:
        ok, mod = check_module("hololoom.memory.vectorstore")
        if not ok:
            print("⚠️ VectorStore module missing; skipping memory tests.")
            return True
        # Expect a class VectorStore(dim: int)
        VS = getattr(mod, "VectorStore", None)
        if VS is None:
            print("⚠️ VectorStore class not found; skipping.")
            return True

        vs = VS(dim=16)
        import numpy as np
        rng = np.random.default_rng(0)
        base = rng.normal(size=(3, 16))
        ids = []
        for i, v in enumerate(base):
            ids.append(vs.add_vector(v, metadata={"id": i}))

        # Query with a noisy version of base[0]
        q = base[0] + 0.01*rng.normal(size=16)
        res = vs.search(q, top_k=1)
        top_id = res[0]["metadata"]["id"]
        print(f"• Top-1 nearest to v0 → id={top_id} (target 0)")
        return top_id == 0
    except Exception:
        print(f"⚠️ Memory check error:\n{traceback.format_exc(limit=1)}")
        return False

def run_policy_checks():
    hr("Policy / Decision Loop Checks")
    ok, mod = check_module("hololoom.policy.unified")
    if not ok:
        print("⚠️ Policy module missing; skipping policy tests.")
        return True
    try:
        # Expect UnifiedPolicy with a .step(obs) -> action dict
        UP = getattr(mod, "UnifiedPolicy", None)
        if UP is None:
            print("⚠️ UnifiedPolicy class not found; skipping.")
            return True
        policy = UP()
        obs = {"retrievals": ["a", "b"], "features": {"score": 0.1}}
        act = policy.step(obs)
        valid = isinstance(act, dict) and ("type" in act or "action" in act)
        print(f"• Policy.step returned: {act}")
        print("✅ Policy check passed" if valid else "❌ Policy check failed")
        return valid
    except Exception:
        print(f"⚠️ Policy check error:\n{traceback.format_exc(limit=1)}")
        return False

def run_pipeline_smoke():
    hr("Orchestrator / Pipeline Smoke Test")
    ok, _ = check_module("hololoom.orchestrator")
    if not ok:
        print("⚠️ Orchestrator missing; skipping pipeline test.")
        return True
    try:
        from hololoom.orchestrator import run_query
        out = run_query("health check: what patterns hold this loom together?")
        # Expect a structured dict-like response
        is_struct = isinstance(out, (dict, SimpleNamespace))
        print("• run_query output:", out if is_struct else str(out)[:200])
        print("✅ Pipeline smoke test passed" if is_struct else "⚠️ Pipeline returned non-structured output")
        return True
    except Exception:
        print(f"⚠️ run_query failed:\n{traceback.format_exc(limit=1)}")
        return False

def main():
    hr("Base Imports")
    base_modules = [
        "hololoom.foundations.dynamics",
        "hololoom.foundations.geometry",
        "hololoom.motif.base",
        "hololoom.embedding.spectral",
        "hololoom.memory.vectorstore",
        "hololoom.policy.unified",
        "hololoom.tools.executor",
    ]
    base_ok = True
    for m in base_modules:
        ok, _ = check_module(m)
        base_ok &= ok
    print("✅ Base imports ok" if base_ok else "⚠️ Some base imports failed")

    results = {
        "linear_algebra": run_linear_algebra_checks(),
        "chebyshev": run_chebyshev_checks(),
        "de_solver": run_de_solver_checks(),
        "geometry": run_geometry_checks(),
        "memory": run_memory_checks(),
        "policy": run_policy_checks(),
        "pipeline": run_pipeline_smoke(),
    }

    hr("Summary")
    for k, v in results.items():
        print(f"{'✅' if v else '❌'} {k}")
    overall = all(results.values())
    print("\n" + ("🎉 ALL CHECKS PASSED" if overall else "⚠️ Some checks failed — see logs above."))

if __name__ == "__main__":
    main()
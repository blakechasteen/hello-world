"""Run all unit tests file-by-file to avoid OOM crashes."""
import subprocess, glob, sys, re, os

test_dir = os.path.join("hololoom", "tests", "unit")
test_files = sorted(glob.glob(os.path.join(test_dir, "test_*.py")))
total_passed = 0
total_failed = 0
total_errors = 0
total_timeout = 0
failed_files = []

for i, tf in enumerate(test_files):
    name = os.path.basename(tf)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", tf, "-q", "--tb=no",
             "-p", "no:timeout", "--no-header"],
            capture_output=True, text=True, timeout=300
        )
        output = result.stdout + result.stderr
        for line in output.strip().split("\n")[::-1]:
            if "passed" in line or "failed" in line or "error" in line:
                p = re.search(r"(\d+) passed", line)
                f = re.search(r"(\d+) failed", line)
                e = re.search(r"(\d+) error", line)
                passed = int(p.group(1)) if p else 0
                failed = int(f.group(1)) if f else 0
                errors = int(e.group(1)) if e else 0
                total_passed += passed
                total_failed += failed
                total_errors += errors
                if failed > 0 or errors > 0:
                    failed_files.append((name, passed, failed, errors))
                break
    except subprocess.TimeoutExpired:
        total_timeout += 1
        failed_files.append((name, 0, 0, -1))

    if (i + 1) % 30 == 0:
        print(f"  Progress: {i+1}/{len(test_files)} files... "
              f"({total_passed}p {total_failed}f {total_errors}e {total_timeout}t)",
              flush=True)

print(f"\n{'='*60}")
print(f"  FULL UNIT SUITE RESULTS ({len(test_files)} files)")
print(f"{'='*60}")
print(f"  Passed:   {total_passed}")
print(f"  Failed:   {total_failed}")
print(f"  Errors:   {total_errors}")
print(f"  Timeout:  {total_timeout}")
print(f"  Total:    {total_passed + total_failed + total_errors}")

if failed_files:
    print(f"\n  Problematic files ({len(failed_files)}):")
    for name, p, f, e in sorted(failed_files):
        if e == -1:
            print(f"    {name:<55} TIMEOUT (>300s)")
        else:
            parts = []
            if f:
                parts.append(f"{f} failed")
            if e:
                parts.append(f"{e} errors")
            print(f"    {name:<55} {p} passed, {', '.join(parts)}")
else:
    print("\n  ALL TESTS PASSED!")

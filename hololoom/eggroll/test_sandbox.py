import os
import sys
import unittest

sys.path.append(os.getcwd())
os.environ["PYTHONPATH"] = os.getcwd()
from hololoom.eggroll.sandbox import NeuralFirewall, SecureExecutor, SecurityViolation


class TestSandbox(unittest.TestCase):

    def test_secure_execution_basic(self):
        print("\n🔒 Testing Basic Execution...")
        executor = SecureExecutor(timeout_sec=5.0)
        code = "result = sum([1, 2, 3])"
        res = executor.execute(code)
        self.assertEqual(res, 6)
        print("   ✅ Basic math works.")

    def test_timeout_enforcement(self):
        print("\n⏳ Testing Timeout Enforcement...")
        executor = SecureExecutor(timeout_sec=2.0)
        code = "while True: pass"
        with self.assertRaises(SecurityViolation) as ctx:
            executor.execute(code)
        print(f"   ✅ Infinite loop caught: {ctx.exception}")

    def test_forbidden_imports(self):
        print("\n🚫 Testing Import Blocking...")
        executor = SecureExecutor()
        code = "import os; os.system('echo hacked')"
        with self.assertRaises(SecurityViolation) as ctx:
            executor.execute(code)
        print(f"   ✅ Malicious import caught: {ctx.exception}")

    def test_file_access_blocked(self):
        print("\n📂 Testing File Access Blocking...")
        executor = SecureExecutor(timeout_sec=5.0)
        # 'open' is not in safe_builtins
        code = "f = open('test.txt', 'w')"
        with self.assertRaises(NameError): # Or SecurityViolation if we caught it earlier
            executor.execute(code)
        print("   ✅ File access blocked (NameError on 'open').")

class MockProbe:
    def __init__(self):
        self.active_steering = {}

class MockGuard:
    def __init__(self):
        self.probe = type('obj', (object,), {'probe': MockProbe()})

class TestNeuralFirewall(unittest.TestCase):

    def test_semantic_containment(self):
        print("\n🧠 Testing Neural Firewall...")

        # 1. Setup Mock System
        guard = MockGuard()
        firewall = NeuralFirewall(guard)

        def safe_generator():
            return "Hello World"

        def unsafe_generator():
            # Simulate the guard detecting something bad DURING generation
            # and setting a negative steering vector
            guard.probe.probe.active_steering = {666: -10.0} # Forbidden feature 666
            return "I am simulating a malicious output."

        # 2. Test Safe
        print("   Testing Safe Generation...")
        out = firewall.generate(safe_generator)
        self.assertEqual(out, "Hello World")
        print("   ✅ Safe generation allowed.")

        # 3. Test Unsafe (Simulated)
        print("   Testing Unsafe Generation...")
        out = firewall.generate(unsafe_generator)
        self.assertIn("Blocked", out)
        print("   ✅ Unsafe generation truncated.")

if __name__ == "__main__":
    unittest.main()

import time
import unittest

from hololoom.eggroll.distributed_backend import LocalBackend


class MockWorker:
    def __init__(self, worker_id, *args, **kwargs):
        self.worker_id = worker_id

    def step(self, *args, **kwargs):
        return f"step_done_{self.worker_id}"

    def set_weights(self, w):
        pass

class TestFaultTolerance(unittest.TestCase):
    def test_local_worker_respawn(self):
        print("\n🧟 Testing LocalBackend Worker Respawn...")
        backend = LocalBackend()

        # Initialize 2 workers
        backend.initialize(num_workers=2, worker_cls=MockWorker)
        time.sleep(1) # Wait for spawn

        # Verify healthy
        status = backend.check_worker_health()
        print(f"   Initial Status: {status}")
        self.assertEqual(status[0], "healthy")
        self.assertEqual(status[1], "healthy")

        # Kill Worker 0
        print("   🔫 Killing Worker 0...")
        backend.workers[0].terminate()
        backend.workers[0].join()

        # Check health (should trigger respawn)
        status_after_death = backend.check_worker_health()
        print(f"   Status after death check: {status_after_death}")

        # Note: implementation of check_worker_health respawns immediately and returns status of OLD worker?
        # Or returns 'dead' then respawns?
        # Let's check implementation:
        # if w.is_alive(): status="healthy" else: status="dead"; respawn()
        # So it returns "dead" for that cycle, but triggers respawn.

        self.assertEqual(status_after_death[0], "dead")

        # Wait for respawn
        time.sleep(1)

        # Check again
        status_resurrected = backend.check_worker_health()
        print(f"   Status after resurrection: {status_resurrected}")
        self.assertEqual(status_resurrected[0], "healthy")

        # shutdown
        backend.shutdown()

if __name__ == "__main__":
    unittest.main()

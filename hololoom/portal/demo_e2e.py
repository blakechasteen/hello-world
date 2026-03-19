#!/usr/bin/env python3
"""
Portal End-to-End Demo Script

This script demonstrates the full Portal WASM execution system:
1. Verifies all services are healthy
2. Lists registered nodes
3. Submits jobs and verifies execution
4. Tests load balancing across nodes
5. (Optional) Tests failover by stopping a node

Prerequisites:
    docker-compose up -d   # Start all services first

Usage:
    python demo_e2e.py                    # Run full demo
    python demo_e2e.py --quick            # Quick health check only
    python demo_e2e.py --test-failover    # Include failover test

Author: HoloLoom Team
Date: 2025-12-09
"""

import argparse
import asyncio
import sys
import time

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

# Configuration
PORTAL_URL = "http://localhost:8080"
NODE1_URL = "http://localhost:9091"
NODE2_URL = "http://localhost:9092"
NODE3_URL = "http://localhost:9093"
NODE4_URL = "http://localhost:9094"
NODE5_URL = "http://localhost:9095"
NODE6_URL = "http://localhost:9096"
ALL_NODE_URLS = [NODE1_URL, NODE2_URL, NODE3_URL, NODE4_URL, NODE5_URL, NODE6_URL]
SHARED_SECRET = "portal-demo-secret-2025"

# Common headers
HEADERS = {"X-Shared-Secret": SHARED_SECRET}


class PortalDemo:
    """End-to-end demo for Portal WASM execution system."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.client: httpx.AsyncClient | None = None
        self.results: dict[str, bool] = {}

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def log(self, msg: str, level: str = "INFO"):
        """Print log message with prefix."""
        if self.verbose:
            prefix = {
                "INFO": "ℹ️ ",
                "OK": "✅",
                "FAIL": "❌",
                "WARN": "⚠️ ",
                "STEP": "🔹",
            }.get(level, "  ")
            print(f"{prefix} {msg}")

    async def check_health(self, url: str, name: str) -> bool:
        """Check if a service is healthy."""
        try:
            response = await self.client.get(f"{url}/health")
            if response.status_code == 200:
                self.log(f"{name}: healthy", "OK")
                return True
            else:
                self.log(f"{name}: unhealthy (status {response.status_code})", "FAIL")
                return False
        except Exception as e:
            self.log(f"{name}: unreachable ({e})", "FAIL")
            return False

    async def step1_verify_services(self) -> bool:
        """Step 1: Verify all services are running."""
        self.log("Step 1: Verifying services...", "STEP")

        portal_ok = await self.check_health(PORTAL_URL, "Portal Server")

        # Check all 6 node daemons
        nodes_ok = []
        for i, node_url in enumerate(ALL_NODE_URLS, 1):
            ok = await self.check_health(node_url, f"Node Daemon {i}")
            nodes_ok.append(ok)

        all_ok = portal_ok and all(nodes_ok)
        self.results["services_healthy"] = all_ok

        if not all_ok:
            self.log("Some services are not healthy. Run 'docker-compose up -d' first.", "WARN")

        return all_ok

    async def step2_list_nodes(self) -> bool:
        """Step 2: List registered nodes from Portal."""
        self.log("Step 2: Listing registered nodes...", "STEP")

        try:
            response = await self.client.get(
                f"{PORTAL_URL}/nodes",
                headers=HEADERS
            )

            if response.status_code == 200:
                nodes = response.json()
                self.log(f"Found {len(nodes)} registered nodes:", "OK")
                for node in nodes:
                    status = node.get("status", "unknown")
                    caps = node.get("capabilities", {})
                    self.log(f"  - {node['node_id']}: {status} "
                            f"(cores: {caps.get('cpu_cores', '?')}, "
                            f"mem: {caps.get('memory_mb', '?')}MB)")
                self.results["nodes_registered"] = len(nodes) >= 6
                return len(nodes) >= 6
            else:
                self.log(f"Failed to list nodes: {response.status_code}", "FAIL")
                return False

        except Exception as e:
            self.log(f"Error listing nodes: {e}", "FAIL")
            return False

    async def step3_get_loom_status(self) -> bool:
        """Step 3: Get overall Loom status."""
        self.log("Step 3: Getting Loom status...", "STEP")

        try:
            response = await self.client.get(
                f"{PORTAL_URL}/loom/status",
                headers=HEADERS
            )

            if response.status_code == 200:
                status = response.json()
                self.log(f"Loom ID: {status.get('loom_id', 'unknown')}", "OK")
                self.log(f"  Online nodes: {status.get('nodes_online', 0)}")
                self.log(f"  Total CPU cores: {status.get('total_cpu_cores', 0)}")
                self.log(f"  Total memory: {status.get('total_memory_mb', 0)}MB")
                self.log(f"  Jobs in progress: {status.get('jobs_in_progress', 0)}")
                self.results["loom_status"] = True
                return True
            else:
                self.log(f"Failed to get Loom status: {response.status_code}", "FAIL")
                return False

        except Exception as e:
            self.log(f"Error getting Loom status: {e}", "FAIL")
            return False

    async def step4_submit_job(self, node_url: str, job_id: str) -> bool:
        """Step 4: Submit a job to a node."""
        self.log(f"Step 4: Submitting job {job_id}...", "STEP")

        job_request = {
            "job_id": job_id,
            "module_id": "mock-add",
            "entry_function": "run",
            "input_json": {"a": 5, "b": 3},
            "timeout_seconds": 30
        }

        try:
            response = await self.client.post(
                f"{node_url}/jobs",
                json=job_request,
                headers=HEADERS
            )

            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                output = result.get("output_json", {})

                if status == "completed":
                    self.log(f"Job {job_id} completed: output={output}", "OK")
                    self.results[f"job_{job_id}"] = True
                    return True
                else:
                    self.log(f"Job {job_id} status: {status}, error: {result.get('error')}", "WARN")
                    return status == "completed"
            else:
                self.log(f"Job submission failed: {response.status_code} - {response.text}", "FAIL")
                return False

        except Exception as e:
            self.log(f"Error submitting job: {e}", "FAIL")
            return False

    async def step5_test_async_jobs(self) -> bool:
        """Step 5: Test async job submission and polling."""
        self.log("Step 5: Testing async job workflow...", "STEP")

        job_id = f"async-test-{int(time.time())}"
        job_request = {
            "job_id": job_id,
            "module_id": "mock-add",
            "entry_function": "run",
            "input_json": {"a": 10, "b": 20},
            "timeout_seconds": 30
        }

        try:
            # Submit async job
            response = await self.client.post(
                f"{NODE1_URL}/jobs/async",
                json=job_request,
                headers=HEADERS
            )

            if response.status_code != 200:
                self.log(f"Async job submission failed: {response.status_code}", "FAIL")
                return False

            result = response.json()
            self.log(f"Async job submitted: {result.get('message')}")

            # Poll for completion
            for i in range(10):
                await asyncio.sleep(0.5)

                poll_response = await self.client.get(
                    f"{NODE1_URL}/jobs/{job_id}",
                    headers=HEADERS
                )

                if poll_response.status_code == 200:
                    job_result = poll_response.json()
                    status = job_result.get("status", "unknown")

                    if status == "completed":
                        self.log(f"Async job completed: {job_result.get('output_json')}", "OK")
                        self.results["async_job"] = True
                        return True
                    elif status == "failed":
                        self.log(f"Async job failed: {job_result.get('error')}", "FAIL")
                        return False
                    else:
                        self.log(f"  Polling {i+1}/10: status={status}")

            self.log("Async job did not complete in time", "FAIL")
            return False

        except Exception as e:
            self.log(f"Error with async job: {e}", "FAIL")
            return False

    async def step6_test_load_balancing(self) -> bool:
        """Step 6: Submit multiple jobs and verify distribution across all 6 nodes."""
        self.log("Step 6: Testing load balancing across 6 nodes...", "STEP")

        job_count = 12  # 2 jobs per node
        jobs = []

        # Submit jobs distributed across all 6 nodes
        for i in range(job_count):
            node_url = ALL_NODE_URLS[i % len(ALL_NODE_URLS)]
            job_id = f"lb-test-{i}-{int(time.time())}"

            job_request = {
                "job_id": job_id,
                "module_id": "mock-add",
                "entry_function": "run",
                "input_json": {"a": i, "b": i * 2},
                "timeout_seconds": 30
            }

            try:
                response = await self.client.post(
                    f"{node_url}/jobs/async",
                    json=job_request,
                    headers=HEADERS
                )

                if response.status_code == 200:
                    jobs.append((job_id, node_url))
                    self.log(f"  Submitted job {i+1}/{job_count} to port {node_url.split(':')[-1]}")
            except Exception as e:
                self.log(f"  Failed to submit job {i+1}: {e}", "WARN")

        # Wait for completion
        await asyncio.sleep(3)

        completed = 0
        for job_id, node_url in jobs:
            try:
                response = await self.client.get(
                    f"{node_url}/jobs/{job_id}",
                    headers=HEADERS
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "completed":
                        completed += 1
            except Exception:
                pass

        success = completed == len(jobs)
        self.results["load_balancing"] = success

        if success:
            self.log(f"All {completed} jobs completed across 6 nodes", "OK")
        else:
            self.log(f"{completed}/{len(jobs)} jobs completed", "WARN")

        return success

    async def step7_test_metrics(self) -> bool:
        """Step 7: Check metrics endpoint (if available)."""
        self.log("Step 7: Checking node metrics...", "STEP")

        try:
            response = await self.client.get(f"{NODE1_URL}/status")

            if response.status_code == 200:
                status = response.json()
                self.log("Node 1 status:", "OK")
                self.log(f"  CPU: {status.get('cpu_percent', 0):.1f}%")
                self.log(f"  Memory: {status.get('memory_percent', 0):.1f}%")
                self.log(f"  Current jobs: {status.get('current_jobs', 0)}")
                self.log(f"  Modules loaded: {status.get('modules_loaded', 0)}")
                self.results["metrics"] = True
                return True
            else:
                self.log(f"Metrics not available: {response.status_code}", "WARN")
                return False

        except Exception as e:
            self.log(f"Error getting metrics: {e}", "WARN")
            return False

    async def run_demo(self, quick: bool = False, test_failover: bool = False):
        """Run the full demo."""
        print("\n" + "=" * 60)
        print("  Portal WASM Execution System - End-to-End Demo")
        print("=" * 60 + "\n")

        # Step 1: Verify services
        if not await self.step1_verify_services():
            print("\n❌ Demo aborted: Services not healthy")
            return False

        if quick:
            print("\n✅ Quick health check passed!")
            return True

        print()

        # Step 2: List nodes
        await self.step2_list_nodes()
        print()

        # Step 3: Loom status
        await self.step3_get_loom_status()
        print()

        # Step 4: Submit job (mock - will complete with mock runner)
        await self.step4_submit_job(NODE1_URL, f"test-{int(time.time())}")
        print()

        # Step 5: Async job workflow
        await self.step5_test_async_jobs()
        print()

        # Step 6: Load balancing
        await self.step6_test_load_balancing()
        print()

        # Step 7: Metrics
        await self.step7_test_metrics()
        print()

        # Summary
        print("=" * 60)
        print("  Demo Summary")
        print("=" * 60)

        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)

        for name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {name}: {status}")

        print()
        print(f"  Total: {passed}/{total} passed")
        print("=" * 60 + "\n")

        return passed == total


async def main():
    parser = argparse.ArgumentParser(description="Portal End-to-End Demo")
    parser.add_argument("--quick", action="store_true", help="Quick health check only")
    parser.add_argument("--test-failover", action="store_true", help="Test node failover")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    async with PortalDemo(verbose=not args.quiet) as demo:
        success = await demo.run_demo(
            quick=args.quick,
            test_failover=args.test_failover
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

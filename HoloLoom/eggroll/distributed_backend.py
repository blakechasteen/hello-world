from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
import multiprocessing
import time
import torch

# --- Abstract Interface ---

class DistributedBackend(ABC):
    """
    Abstract Base Class for HoloLoom's Distributed Execution.
    Supports Local (Processes), Ray (Clusters), and Tuple (MPI/Torch).
    """
    
    @abstractmethod
    def initialize(self, num_workers: int, worker_cls: Callable, *args, **kwargs):
        """Spawns the workers."""
        pass
        
    @abstractmethod
    def scatter_broadcast(self, data: Any):
        """Sends data (e.g. weights) to all workers."""
        pass
        
    @abstractmethod
    def gather_results(self) -> List[Any]:
        """Collects results from all workers."""
        pass
        
    @abstractmethod
    def map_async(self, func_name: str, *args, **kwargs):
        """Triggers a method on all workers asynchronously."""
        pass

    @abstractmethod
    def check_worker_health(self) -> Dict[int, str]:
        """Returns status of all workers: 'healthy', 'dead', 'unresponsive'."""
        pass
        
    @abstractmethod
    def shutdown(self):
        """Cleanup."""
        pass

# --- 1. Local Backend (Multiprocessing) ---

class WorkerNode(multiprocessing.Process):
    def __init__(self, idx, worker_cls, task_queue, result_queue, init_args, init_kwargs):
        super().__init__()
        self.idx = idx
        self.worker_cls = worker_cls
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        self.worker_instance = None
        
    def run(self):
        # Instantiate the worker agent
        print(f"[Worker {self.idx}] Initializing...")
        try:
            self.worker_instance = self.worker_cls(worker_id=self.idx, *self.init_args, **self.init_kwargs)
        except Exception as e:
            print(f"[Worker {self.idx}] Initialization FAILED: {e}")
            self.result_queue.put((self.idx, e))
            return
        
        while True:
            try:
                task = self.task_queue.get(timeout=1.0) # Check regulary for shutdown/ping
            except Exception:
                continue # queue empty
                
            if task is None: # Sentinel
                break
                
            cmd, args, kwargs = task
            
            try:
                if cmd == "set_weights":
                    self.worker_instance.set_weights(*args)
                    self.result_queue.put((self.idx, "ack"))
                elif cmd == "step":
                    result = self.worker_instance.step(*args, **kwargs)
                    self.result_queue.put((self.idx, result))
                elif cmd == "ping":
                    self.result_queue.put((self.idx, "pong"))
                elif hasattr(self.worker_instance, cmd):
                    method = getattr(self.worker_instance, cmd)
                    res = method(*args, **kwargs)
                    self.result_queue.put((self.idx, res))
                else:
                    self.result_queue.put((self.idx, Exception(f"Unknown command: {cmd}")))
            except Exception as e:
                self.result_queue.put((self.idx, e))

class LocalBackend(DistributedBackend):
    def __init__(self):
        self.workers = []
        self.task_queues = []
        self.result_queue = multiprocessing.Queue()
        self.num_workers = 0
        self.worker_cls = None
        self.init_args = []
        self.init_kwargs = {}

    def initialize(self, num_workers: int, worker_cls: Callable, *args, **kwargs):
        self.num_workers = num_workers
        self.worker_cls = worker_cls
        self.init_args = args
        self.init_kwargs = kwargs
        
        for i in range(num_workers):
            self._spawn_worker(i)
            
    def _spawn_worker(self, i: int):
        tq = multiprocessing.Queue()
        w = WorkerNode(i, self.worker_cls, tq, self.result_queue, self.init_args, self.init_kwargs)
        w.start()
        if i < len(self.workers):
            self.workers[i] = w
            self.task_queues[i] = tq
        else:
            self.workers.append(w)
            self.task_queues.append(tq)
            
    def scatter_broadcast(self, weights: Dict[str, torch.Tensor]):
        # Naive broadcast via queues (slow for large models, good for prototyping)
        for tq in self.task_queues:
            tq.put(("set_weights", (weights,), {}))
            
        # Wait for ACKs
        acks = 0
        while acks < self.num_workers:
            res = self.result_queue.get()
            # If we get an exception, handle it?
            acks += 1
            
    def map_async(self, func_name: str, *args, **kwargs):
        for tq in self.task_queues:
            tq.put((func_name, args, kwargs))
            
    def gather_results(self) -> List[Any]:
        results = []
        for _ in range(self.num_workers):
            idx, res = self.result_queue.get()
            if isinstance(res, Exception):
                raise res
            results.append(res)
        return results
        
    def check_worker_health(self) -> Dict[int, str]:
        status = {}
        for i, w in enumerate(self.workers):
            if w.is_alive():
                status[i] = "healthy"
            else:
                status[i] = "dead"
                print(f"⚠️ LocalBackend: Worker {i} is DEAD. Respawning...")
                self._spawn_worker(i) # Auto-Respawn
        return status

    def shutdown(self):
        for tq in self.task_queues:
            tq.put(None)
        for w in self.workers:
            w.join()

# --- 2. Ray Backend (Production) ---
# NOTE: To use on Frontier, install ray[default]

class RayBackend(DistributedBackend):
    def __init__(self, max_retries=3):
        try:
            import ray
            self.ray = ray
        except ImportError:
            raise ImportError("Ray not installed. Run `pip install ray`.")
        self.max_retries = max_retries
        self.actors = []
        
    def initialize(self, num_workers: int, worker_cls: Callable, resources: Dict = None, *args, **kwargs):
        if not self.ray.is_initialized():
            self.ray.init(ignore_reinit_error=True)
            
        # Resource specification per worker (e.g. {"num_gpus": 1})
        self.ActorClass = self.ray.remote(**(resources or {}))(worker_cls)
        
        self.actors = [self.ActorClass.remote(worker_id=i, *args, **kwargs) for i in range(num_workers)]
        
    def scatter_broadcast(self, weights):
        weights_ref = self.ray.put(weights)
        futures = [a.set_weights.remote(weights_ref) for a in self.actors]
        self.ray.get(futures) # Wait for completion
        
    def map_async(self, func_name: str, *args, **kwargs):
        self.pending_futures = []
        for a in self.actors:
            method = getattr(a, func_name)
            self.pending_futures.append(method.remote(*args, **kwargs))
            
    def gather_results(self) -> List[Any]:
        # Robust gather with timeout?
        # For strict evolution, we need all results.
        # If one fails, we might just drop that result or retry?
        try:
            return self.ray.get(self.pending_futures)
        except Exception as e:
            # RayActorError
            print(f"🔥 RayBackend Error during gather: {e}")
            # Identify failed actor and restart?
            # For Phase 2, we just raise.
            raise e

    def check_worker_health(self) -> Dict[int, str]:
        # Ray handles health automatically, but we can check reachability
        status = {}
        # We can ping them
        futures = [a.ping.remote() for a in self.actors if hasattr(a, 'ping')] 
        # But our loom_node doesn't have ping. We can assume healthy if no RayActorError.
        # Implementation left simple for now.
        return {i: "healthy" for i in range(len(self.actors))}

    def shutdown(self):
        for a in self.actors:
            self.ray.kill(a)
        self.ray.shutdown()

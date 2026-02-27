from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Button, Log, Sparkline
from textual.reactive import reactive
import asyncio
import random
import time

# Import EGGROLL components
# Note: In a real app, we'd run the integration in a separate thread/process
# and communicate via queues. For this demo, we'll simulate the loop steps.
from hololoom.eggroll.integration import EggrollIntegration

class DreamCatcherApp(App):
    """EGGROLL Dream Catcher TUI"""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    .box {
        height: 100%;
        border: solid green;
    }
    
    #sidebar {
        width: 30%;
        dock: left;
    }
    
    #main {
        width: 70%;
    }
    
    #status_panel {
        height: 20%;
        border: solid blue;
        padding: 1;
    }
    
    #log_panel {
        height: 80%;
        border: solid yellow;
    }
    
    Sparkline {
        width: 100%;
        height: 100%;
        color: green;
    }
    
    .stat_label {
        color: cyan;
        text-style: bold;
    }
    """
    
    BINDINGS = [("q", "quit", "Quit")]
    
    # Reactive state
    epoch = reactive(0)
    avg_reward = reactive(0.0)
    current_pattern = reactive("Waiting...")
    worker_status = reactive("Idle")
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Horizontal():
            # Left Sidebar: Stats & Controls
            with Vertical(id="sidebar"):
                with Container(classes="box", id="status_panel"):
                    yield Static("Status: Idle", id="status_text")
                    yield Static("Epoch: 0", id="epoch_text")
                    yield Static("Pattern: -", id="pattern_text")
                    yield Static("Reward: 0.00", id="reward_text")
                    yield Button("Start Dreaming", id="start_btn", variant="success")
                
                with Container(classes="box"):
                    yield Static("Reward History", classes="stat_label")
                    yield Sparkline(summary_function=max)
            
            # Main Area: Logs
            with Vertical(id="main"):
                with Container(classes="box", id="log_panel"):
                    yield Log(id="activity_log")
                    
        yield Footer()

    def on_mount(self):
        self.integration = EggrollIntegration(num_workers=5)
        self.is_dreaming = False
        self.rewards_history = []

    async def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "start_btn":
            if not self.is_dreaming:
                self.is_dreaming = True
                event.button.label = "Stop Dreaming"
                event.button.variant = "error"
                self.run_worker(self.dream_loop(), exclusive=True)
            else:
                self.is_dreaming = False
                event.button.label = "Start Dreaming"
                event.button.variant = "success"

    async def dream_loop(self):
        """Async loop to run the dream cycle and update UI."""
        log = self.query_one("#activity_log", Log)
        status = self.query_one("#status_text", Static)
        epoch_text = self.query_one("#epoch_text", Static)
        pattern_text = self.query_one("#pattern_text", Static)
        reward_text = self.query_one("#reward_text", Static)
        sparkline = self.query_one(Sparkline)
        
        log.write_line("[bold green]Initializing Dream Cycle...[/]")
        
        while self.is_dreaming:
            self.epoch += 1
            epoch_text.update(f"Epoch: {self.epoch}")
            status.update("Status: Selecting Pattern (MCTS)...")
            
            # 1. Select Pattern
            pattern = self.integration.shuttle.select_pattern()
            self.current_pattern = pattern
            pattern_text.update(f"Pattern: {pattern}")
            log.write_line(f"Epoch {self.epoch}: Selected pattern '{pattern}' via MCTS")
            await asyncio.sleep(0.5) # UI pause
            
            # 2. Propose Perturbations
            status.update("Status: Proposing Perturbations (Thompson)...")
            perturbations = self.integration.shuttle.propose_perturbations(
                self.integration.mirror_core.model_id, 
                self.integration.num_workers
            )
            rank, scale = self.integration.shuttle.last_selected_arm
            log.write_line(f"  -> Thompson Sampling: Rank={rank}, Scale={scale}")
            await asyncio.sleep(0.5)
            
            # 3. Run Workers
            status.update(f"Status: Running {self.integration.num_workers} Workers...")
            tasks = ["Task 1", "Task 2"] # Placeholder
            
            worker_results = []
            for i, worker in enumerate(self.integration.workers):
                # Simulate work
                await asyncio.sleep(0.2) 
                
                # Real logic from integration (simplified for TUI demo)
                spec = perturbations[i]
                perturbed_model = worker.apply_perturbation(self.integration.mirror_core, spec)
                outputs = await worker.evaluate(perturbed_model, tasks)
                
                # Score
                score = self.integration.warp.score("target", outputs)
                reward = score # Simplified
                
                worker_results.append({
                    "worker_id": i,
                    "reward": reward,
                    "perturb_spec": spec
                })
                log.write_line(f"    Worker {i}: Reward={reward:.4f}")
            
            # 4. Update & Aggregate
            status.update("Status: Aggregating & Updating...")
            avg_reward = sum(r['reward'] for r in worker_results) / len(worker_results)
            self.avg_reward = avg_reward
            reward_text.update(f"Reward: {avg_reward:.4f}")
            
            self.rewards_history.append(avg_reward)
            sparkline.data = self.rewards_history
            
            self.integration.shuttle.update_bandit(worker_results)
            update_matrix = self.integration.aggregate_updates(worker_results)
            self.integration.mirror_core.update_weights(update_matrix)
            
            log.write_line(f"[bold yellow]Epoch {self.epoch} Complete. Avg Reward: {avg_reward:.4f}[/]")
            log.write_line("-" * 40)
            
            await asyncio.sleep(1.0) # Pause between epochs

if __name__ == "__main__":
    app = DreamCatcherApp()
    app.run()

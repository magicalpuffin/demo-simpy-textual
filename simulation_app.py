import asyncio
import random
from dataclasses import dataclass

import simpy
from textual import on
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup, ItemGrid, Vertical, VerticalGroup
from textual.message import Message
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    ProgressBar,
    Static,
    TabbedContent,
)


@dataclass
class SimParams:
    sim_duration: int
    num_machines: int
    process_time: int


class FactorySim:
    """SimPy-based factory simulation with a queue and machine status."""

    def __init__(self, env: simpy.Environment, num_machines: int, process_time: float):
        self.env = env
        self.machine = simpy.Resource(env, num_machines)
        self.process_time = process_time
        self.queue = []  # Parts waiting for processing
        self.active_parts: list[int] = []  # Parts currently being processed

    def process_part(self, part_id: int):
        """Simulate part processing and track queue state."""
        self.queue.append(part_id)  # Add part to queue
        with self.machine.request() as req:
            yield req  # Wait for an available machine
            self.queue.remove(part_id)  # Remove from queue
            self.active_parts.append(part_id)  # Mark as processing

            yield self.env.timeout(
                random.uniform(self.process_time * 0.8, self.process_time * 1.2)
            )  # Processing time

            self.active_parts.remove(part_id)  # Mark as completed


def part_arrival(env: simpy.Environment, sim: FactorySim):
    """Parts arrive at random intervals."""
    part_id = 0
    while True:
        yield env.timeout(
            random.expovariate(1.0 / 2.0)
        )  # New part ~ every 5 time units
        env.process(sim.process_part(part_id))
        part_id += 1


class SimulationControl(VerticalGroup):
    sim_task: asyncio.Task | None
    start_sim_time: float
    end_sim_time: float
    current_sim_time: float

    class SimulationIteration(Message):
        def __init__(self, sim: FactorySim) -> None:
            self.sim = sim
            super().__init__()

    def __init__(self) -> None:
        self.start_sim_time = 0
        self.current_sim_time = 0
        self.end_sim_time = 100
        self.sim_task = None
        self.sim_params = SimParams(sim_duration=100, num_machines=5, process_time=10)

        super().__init__()

    def compose(self) -> ComposeResult:
        self.border_title = "Simulation Control"
        with HorizontalGroup(id="toppart"):
            yield ProgressBar(total=self.end_sim_time, show_eta=False)
            yield Label("0/100", id="progress-label")
            with HorizontalGroup(id="button-menu"):
                yield Button("Start", variant="success", id="start")
                yield Button("Pause", id="pause-resume", variant="warning")
                yield Button("Reset", id="reset", variant="error")

    def on_mount(self):
        self.init_simulation()
        self.update_progress_label()

    @on(Button.Pressed, "#start")
    def start_sim(self):
        self.add_class("started")
        self.sim_task = asyncio.create_task(
            self.run_simulation(self.start_sim_time, self.end_sim_time)
        )

    @on(Button.Pressed, "#pause-resume")
    def pause_resume_sim(self, event: Button.Pressed):
        if self.paused:
            self.resume_sim()
            self.paused = False
            event.button.label = "Pause"
            event.button.variant = "warning"
        else:
            self.stop_sim()
            self.paused = True
            event.button.label = "Resume"
            event.button.variant = "success"

    def stop_sim(self):
        if self.sim_task:
            self.sim_task.cancel()

    def resume_sim(self):
        self.sim_task = asyncio.create_task(
            self.run_simulation(self.current_sim_time, self.end_sim_time)
        )

    @on(Button.Pressed, "#reset")
    def reset_sim(self):
        self.stop_sim()
        self.remove_class("started")
        self.init_simulation()

    def init_simulation(self):
        self.env = simpy.Environment()
        self.sim = FactorySim(
            self.env,
            num_machines=self.sim_params.num_machines,
            process_time=self.sim_params.process_time,
        )
        self.env.process(part_arrival(self.env, self.sim))

        self.current_sim_time = self.start_sim_time
        self.paused = False

        self.query_one(ProgressBar).update(
            total=self.sim_params.sim_duration, progress=0
        )
        self.update_progress_label()
        self.post_message(self.SimulationIteration(self.sim))

    async def run_simulation(self, start, end):
        for i in range(start, end):
            await asyncio.sleep(0.05)
            self.current_sim_time = i + 1
            self.env.run(until=self.current_sim_time)

            self.query_one(ProgressBar).update(progress=self.current_sim_time)
            self.update_progress_label()
            self.post_message(self.SimulationIteration(self.sim))

    def update_params(self, simparams: SimParams):
        self.sim_params = simparams

        self.end_sim_time = simparams.sim_duration

    def update_progress_label(self):
        self.query_one("#progress-label", Label).update(
            f"{self.current_sim_time:.0f}/{self.end_sim_time:.0f}"
        )


class SimulationAnimation(Vertical):
    class QueueDisplay(HorizontalGroup):
        def compose(self) -> ComposeResult:
            self.border_title = "Queue"
            self.queued_num = Label("", id="num")
            self.queued_items = Label("", id="items")
            yield self.queued_num
            yield self.queued_items

        def update(self, num: int):
            self.queued_num.update(f"{num:3,.0f}")
            self.queued_items.update(" ".join(["█" for i in range(num)]))

    class MachineDisplay(Static):
        def compose(self) -> ComposeResult:
            self.border_title = self._content
            self.active_part = Label("", id="part")
            yield self.active_part

        def update_part(self, part_id: None | int):
            self.active_part.update(f"Part: {part_id}")
            if part_id is None:
                self.remove_class("active")
            else:
                self.add_class("active")

    def compose(self) -> ComposeResult:
        self.queue_display = self.QueueDisplay()

        yield self.queue_display
        yield ItemGrid(id="machine-grid")

    def on_mount(self):
        self.update_machine_grid(5)

    def update_text(self, sim: FactorySim):
        self.queue_display.update(sim.machine.queue.__len__())
        for i, machine_display in enumerate(self.query(self.MachineDisplay)):
            machine_display.update_part(
                sim.active_parts[i] if i < len(sim.active_parts) else None
            )

    def update_machine_grid(self, num: int):
        self.query_one("#machine-grid").remove_children()
        self.query_one("#machine-grid").mount_all(
            [self.MachineDisplay(f"machine-{i + 1}") for i in range(num)]
        )


class SimulationInputs(Vertical):
    def __init__(self):
        self.params = SimParams(sim_duration=100, num_machines=5, process_time=10)
        super().__init__()

    class SimulationInputsUpdated(Message):
        def __init__(self, simparams: SimParams) -> None:
            self.simparams = simparams
            super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Duration")
            yield Input(
                value=str(self.params.sim_duration), type="integer", id="sim_duration"
            )
            yield Label("Number of Machines")
            yield Input(
                value=str(self.params.num_machines), type="integer", id="num_machines"
            )
            yield Label("Process Time")
            yield Input(
                value=str(self.params.process_time), type="integer", id="process_time"
            )
            yield Button("Another button")

    def on_mount(self):
        self.post_message(self.SimulationInputsUpdated(self.params))

    @on(Input.Changed)
    def params_updated(self, event: Input.Changed) -> None:
        if event.input.id == "sim_duration":
            self.params.sim_duration = int(event.input.value)
        if event.input.id == "num_machines":
            self.params.num_machines = int(event.input.value)
        if event.input.id == "process_time":
            self.params.process_time = int(event.input.value)

        self.post_message(self.SimulationInputsUpdated(self.params))


class SimulationApp(App):
    """Textual application to visualize the SimPy simulation."""

    CSS_PATH = "./simulation_app.tcss"

    def compose(self) -> ComposeResult:
        """Create UI elements."""
        self.simani = SimulationAnimation(id="sim_animation")
        self.simcontrol = SimulationControl()
        self.siminputs = SimulationInputs()

        yield Header()
        yield Footer()
        yield self.simcontrol

        with TabbedContent("Simulation", "Params", "Logs", "Figures", id="content"):
            with Vertical():
                yield self.simani
            yield self.siminputs
            yield Markdown()
            yield Markdown()

    @on(SimulationControl.SimulationIteration)
    def animate_iteration(self, message: SimulationControl.SimulationIteration):
        self.simani.update_text(message.sim)

    @on(SimulationInputs.SimulationInputsUpdated)
    def simparams_update(self, message: SimulationInputs.SimulationInputsUpdated):
        self.simani.update_machine_grid(message.simparams.num_machines)
        self.simcontrol.update_params(message.simparams)


if __name__ == "__main__":
    SimulationApp().run()

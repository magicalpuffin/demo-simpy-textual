import asyncio
import random
from dataclasses import dataclass
from typing import Literal

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


# todo, update more params to be float
# todo, add random seed
@dataclass
class SimulationParameters:
    start_sim_time: int = 0
    end_sim_time: int = 100
    step_sim_time: int = 1
    step_delay_time: float = 0.05
    num_machines: int = 5
    process_time: float = 10


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
    current_sim_time: float
    params = SimulationParameters

    class SimulationIteration(Message):
        def __init__(self, sim: FactorySim) -> None:
            self.sim = sim
            super().__init__()

    def __init__(self) -> None:
        self.current_sim_time = 0
        self.sim_task = None
        self.params = SimulationParameters()

        super().__init__()

    def compose(self) -> ComposeResult:
        self.border_title = "Simulation Control"
        with HorizontalGroup(id="toppart"):
            yield ProgressBar(total=self.params.end_sim_time, show_eta=False)
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
        print(self.params)
        self.sim_task = asyncio.create_task(
            self.run_simulation(self.params.start_sim_time, self.params.end_sim_time)
        )

    @on(Button.Pressed, "#pause-resume")
    def pause_resume_sim(self, event: Button.Pressed):
        # todo, fix button pause resume state out of sync
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
            self.run_simulation(self.current_sim_time, self.params.end_sim_time)
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
            num_machines=self.params.num_machines,
            process_time=self.params.process_time,
        )
        self.env.process(part_arrival(self.env, self.sim))

        self.current_sim_time = self.params.start_sim_time
        self.paused = False

        self.query_one(ProgressBar).update(total=self.params.end_sim_time, progress=0)
        self.update_progress_label()
        self.post_message(self.SimulationIteration(self.sim))

    async def run_simulation(self, start, end):
        # todo, fix step sim time, should be float
        for i in range(start, end, self.params.step_sim_time):
            await asyncio.sleep(self.params.step_delay_time)
            self.current_sim_time = i + self.params.step_sim_time
            self.env.run(until=self.current_sim_time)

            self.query_one(ProgressBar).update(progress=self.current_sim_time)
            self.update_progress_label()
            self.post_message(self.SimulationIteration(self.sim))

    def update_params(self, simparams: SimulationParameters):
        self.params = simparams

    def update_progress_label(self):
        self.query_one("#progress-label", Label).update(
            f"{self.current_sim_time:.0f}/{self.params.end_sim_time:.0f}"
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


# todo, parameterize inputs better
class SimulationInputs(Vertical):
    def __init__(self):
        self.params = SimulationParameters()
        super().__init__()

    class SimulationInputsUpdated(Message):
        def __init__(self, simparams: SimulationParameters) -> None:
            self.simparams = simparams
            super().__init__()

    class LabeledInput(HorizontalGroup):
        def __init__(
            self,
            label: str,
            type: Literal["text", "number", "integer"],
            input_id: str,
            params: SimulationParameters,
        ) -> None:
            self.label = label
            self.type: Literal["text", "number", "integer"] = type
            self.input_id = input_id
            self.params = params
            super().__init__()

        def compose(self) -> ComposeResult:
            yield Label(self.label)
            yield Input(
                value=str(self.params.__getattribute__(self.input_id)),
                type=self.type,
                id=self.input_id,
            )

        @on(Input.Changed)
        def value_changed(self, event: Input.Changed):
            # todo, have better error handling and validation
            try:
                type_to_cast = {"text": str, "number": float, "integer": int}
                self.params.__setattr__(
                    self.input_id, type_to_cast[self.type](event.input.value)
                )
            except ValueError:
                print("Error setting attribute")

    def compose(self) -> ComposeResult:
        with VerticalGroup() as control_inputs:
            control_inputs.border_title = "Control Inputs"

            yield self.LabeledInput(
                "Start Time", "integer", "start_sim_time", self.params
            )
            yield self.LabeledInput("End Time", "integer", "end_sim_time", self.params)
            yield self.LabeledInput(
                "Step Time", "integer", "step_sim_time", self.params
            )
            yield self.LabeledInput(
                "Step Delay (s)", "number", "step_delay_time", self.params
            )
        with VerticalGroup() as simulation_inputs:
            simulation_inputs.border_title = "Simulation Inputs"
            yield self.LabeledInput(
                "# of Machines", "integer", "num_machines", self.params
            )
            yield self.LabeledInput(
                "Process Time", "number", "process_time", self.params
            )

    # def on_mount(self):
    #     self.post_message(self.SimulationInputsUpdated(self.params))

    @on(Input.Changed)
    def params_updated(self, event: Input.Changed) -> None:
        print("PARAMETERS UDPATED")

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

import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import polars as pl
import simpy
from textual import on
from textual.app import App, ComposeResult
from textual.containers import (
    HorizontalGroup,
    ItemGrid,
    Vertical,
    VerticalGroup,
    VerticalScroll,
)
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
from textual_plotext import PlotextPlot

from machine_shop import MachineShop, MachineShopParams


# todo, update more params to be float
# todo, add random seed
@dataclass
class SimulationControlParams:
    start_sim_time: int = 0
    end_sim_time: int = 1000
    step_sim_time: int = 1
    step_delay_time: float = 0.05


class SimulationControl(VerticalGroup):
    sim_task: asyncio.Task | None
    current_sim_time: float
    simulation_control_params: SimulationControlParams
    machine_shop_params: MachineShopParams

    class SimulationIteration(Message):
        def __init__(self, sim: MachineShop) -> None:
            self.sim = sim
            super().__init__()

    def __init__(self) -> None:
        self.current_sim_time = 0
        self.sim_task = None
        self.simulation_control_params = SimulationControlParams()
        self.machine_shop_params = MachineShopParams()

        super().__init__()

    def compose(self) -> ComposeResult:
        self.border_title = "Simulation Control"
        with HorizontalGroup(id="toppart"):
            yield ProgressBar(
                total=self.simulation_control_params.end_sim_time, show_eta=False
            )
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
        print(self.simulation_control_params)
        self.sim_task = asyncio.create_task(
            self.run_simulation(
                self.simulation_control_params.start_sim_time,
                self.simulation_control_params.end_sim_time,
            )
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
            self.run_simulation(
                self.current_sim_time, self.simulation_control_params.end_sim_time
            )
        )

    @on(Button.Pressed, "#reset")
    def reset_sim(self):
        self.stop_sim()
        self.remove_class("started")
        self.init_simulation()

    def init_simulation(self):
        self.env = simpy.Environment()
        self.sim = MachineShop(self.env, params=self.machine_shop_params)

        self.current_sim_time = self.simulation_control_params.start_sim_time
        self.paused = False

        self.query_one(ProgressBar).update(
            total=self.simulation_control_params.end_sim_time, progress=0
        )
        self.update_progress_label()
        self.post_message(self.SimulationIteration(self.sim))

    async def run_simulation(self, start, end):
        # todo, fix step sim time, should be float
        for i in range(start, end, self.simulation_control_params.step_sim_time):
            await asyncio.sleep(self.simulation_control_params.step_delay_time)
            self.current_sim_time = i + self.simulation_control_params.step_sim_time
            self.env.run(until=self.current_sim_time)

            self.query_one(ProgressBar).update(progress=self.current_sim_time)
            self.update_progress_label()
            self.post_message(self.SimulationIteration(self.sim))

    def update_simulation_control_params(self, params: SimulationControlParams):
        self.simulation_control_params = params

    def update_machine_shop_params(self, params: MachineShopParams):
        self.machine_shop_params = params

    def update_progress_label(self):
        self.query_one("#progress-label", Label).update(
            f"{self.current_sim_time:.0f}/{self.simulation_control_params.end_sim_time:.0f}"
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
            self.active_part = Label("Part: None", id="part")
            self.parts_made = Label("Parts Made", id="parts_made")

            yield self.active_part
            yield self.parts_made

        def update_part(self, part_id: None | int, broken: bool, parts_made: int):
            self.active_part.update(f"Part: {part_id}")
            self.parts_made.update(f"Parts Made: {parts_made}")

            if part_id is None:
                self.remove_class("active")
            else:
                self.add_class("active")
            if broken:
                self.add_class("broken")
            else:
                self.remove_class("broken")

    def compose(self) -> ComposeResult:
        self.queue_display = self.QueueDisplay()

        yield self.queue_display
        yield ItemGrid(id="machine-grid")

    def on_mount(self):
        # todo, have better default
        self.update_machine_grid(5)

    def update_text(self, sim: MachineShop):
        self.queue_display.update(sim.store.items.__len__())
        for i, machine_display in enumerate(self.query(self.MachineDisplay)):
            machine_display.update_part(
                sim.machines[i].part_id,
                sim.machines[i].broken,
                sim.machines[i].parts_made,
            )

    def update_machine_grid(self, num: int):
        self.query_one("#machine-grid").remove_children()
        self.query_one("#machine-grid").mount_all(
            [self.MachineDisplay(f"machine-{i + 1}") for i in range(num)]
        )


class SimulationInputs(Vertical):
    def __init__(self):
        self.simulation_control_params = SimulationControlParams()
        self.machine_shop_params = MachineShopParams()
        super().__init__()

    class SimulationControlParamsUpdated(Message):
        def __init__(self, params: SimulationControlParams) -> None:
            self.params = params
            super().__init__()

    class MachineShopParamsUpdated(Message):
        def __init__(self, params: MachineShopParams) -> None:
            self.params = params
            super().__init__()

    class LabeledInput(HorizontalGroup):
        def __init__(
            self,
            label: str,
            type: Literal["text", "number", "integer"],
            input_id: str,
            params: object,
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
                "Start Time",
                "integer",
                "start_sim_time",
                self.simulation_control_params,
            )
            yield self.LabeledInput(
                "End Time", "integer", "end_sim_time", self.simulation_control_params
            )
            yield self.LabeledInput(
                "Step Time", "integer", "step_sim_time", self.simulation_control_params
            )
            yield self.LabeledInput(
                "Step Delay (s)",
                "number",
                "step_delay_time",
                self.simulation_control_params,
            )
        with VerticalGroup() as simulation_inputs:
            simulation_inputs.border_title = "Simulation Inputs"
            yield self.LabeledInput(
                "# of Machines", "integer", "num_machines", self.machine_shop_params
            )
            yield self.LabeledInput(
                "# of Repairman", "integer", "num_repairman", self.machine_shop_params
            )
            yield self.LabeledInput(
                "Mean Time to Arrive",
                "number",
                "mean_time_to_arrive",
                self.machine_shop_params,
            )
            yield self.LabeledInput(
                "Mean Process Time",
                "number",
                "mean_process_time",
                self.machine_shop_params.machine_params,
            )
            yield self.LabeledInput(
                "Stdv Process Time",
                "number",
                "stdv_process_time",
                self.machine_shop_params.machine_params,
            )
            yield self.LabeledInput(
                "Mean Time to Failure",
                "number",
                "mean_time_to_failure",
                self.machine_shop_params.machine_params,
            )
            yield self.LabeledInput(
                "Repair Time",
                "number",
                "repair_time",
                self.machine_shop_params.machine_params,
            )

    @on(Input.Changed)
    def params_updated(self, event: Input.Changed) -> None:
        print("PARAMETERS UDPATED")
        self.post_message(
            self.SimulationControlParamsUpdated(self.simulation_control_params)
        )
        self.post_message(self.MachineShopParamsUpdated(self.machine_shop_params))


class SimulationFigures(VerticalScroll):
    class LinePlot(PlotextPlot):
        def __init__(self, title: str, xlabel: str, ylabel: str):
            self.title = title
            self.xlabel = xlabel
            self.ylabel = ylabel
            super().__init__()

        def on_mount(self):
            self.plt.title(self.title)
            self.plt.xlabel(self.xlabel)
            self.plt.ylabel(self.ylabel)

        def refresh_plot(self, *args: Sequence[Any]):
            self.plt.clear_data()
            self.plt.plot(*args)
            self.refresh()

    def compose(self) -> ComposeResult:
        self.parts_over_time = self.LinePlot(
            title="Parts Over Time", xlabel="Time", ylabel="# of Parts"
        )
        self.queue_over_time = self.LinePlot(
            title="Queue Over Time", xlabel="Time", ylabel="# in Queue"
        )
        self.idle_duration_over_time = self.LinePlot(
            title="Idle Duration Over Time", xlabel="Time", ylabel="Idle Duration"
        )
        self.broken_duration_over_time = self.LinePlot(
            title="Broken Duration Over Time", xlabel="Time", ylabel="Broken Duration"
        )
        self.parts_cycle_time = self.LinePlot(
            title="Parts Cycle Time", xlabel="Time", ylabel="Cycle Time"
        )

        with HorizontalGroup():
            yield self.parts_over_time
            yield self.queue_over_time

        with HorizontalGroup():
            yield self.idle_duration_over_time
            yield self.broken_duration_over_time

        with HorizontalGroup():
            yield self.parts_cycle_time

    def update_figures(self, sim: MachineShop):
        if len(sim.metrics_log) < 1:
            return

        df = pl.DataFrame(sim.metrics_log).tail(50)
        self.parts_over_time.refresh_plot(
            df.to_dict()["time"].to_list(),
            df.to_dict()["total_parts_made"].to_list(),
        )
        self.queue_over_time.refresh_plot(
            df.to_dict()["time"].to_list(), df.to_dict()["queue_items"].to_list()
        )
        self.idle_duration_over_time.refresh_plot(
            df.to_dict()["time"].to_list(),
            df.to_dict()["total_idle_duration"].to_list(),
        )
        self.broken_duration_over_time.refresh_plot(
            df.to_dict()["time"].to_list(),
            df.to_dict()["total_broken_duration"].to_list(),
        )

        # idle_ratio_df = df.select()

        cycle_time_df = df.select(
            [
                (pl.col("time") - pl.col("time").min()),
                (pl.col("total_parts_made") - pl.col("total_parts_made").min()),
            ]
        )
        cycle_time_df = df.select(
            pl.col("time"),
            (pl.col("time") / pl.col("total_parts_made")).alias("cycle_time"),
        )
        cycle_time_df = cycle_time_df.filter(
            ~pl.col("cycle_time").is_infinite()
        ).fill_nan(0)
        self.parts_cycle_time.refresh_plot(
            cycle_time_df.to_dict()["time"].to_list(),
            cycle_time_df.to_dict()["cycle_time"].to_list(),
        )


class SimulationApp(App):
    """Textual application to visualize the SimPy simulation."""

    CSS_PATH = "./simulation_app.tcss"

    def compose(self) -> ComposeResult:
        """Create UI elements."""
        self.animation = SimulationAnimation(id="sim_animation")
        self.control = SimulationControl()
        self.inputs = SimulationInputs()
        self.figures = SimulationFigures()

        yield Header()
        yield Footer()
        yield self.control

        with TabbedContent("Simulation", "Params", "Logs", "Figures", id="content"):
            yield self.animation
            yield self.inputs
            yield Markdown()
            yield self.figures

    @on(SimulationControl.SimulationIteration)
    def animate_iteration(self, message: SimulationControl.SimulationIteration):
        self.animation.update_text(message.sim)
        self.figures.update_figures(message.sim)

    @on(SimulationInputs.SimulationControlParamsUpdated)
    def simulation_control_params_update(
        self, message: SimulationInputs.SimulationControlParamsUpdated
    ):
        self.control.update_simulation_control_params(message.params)

    @on(SimulationInputs.MachineShopParamsUpdated)
    def machine_shop_params_update(
        self, message: SimulationInputs.MachineShopParamsUpdated
    ):
        self.animation.update_machine_grid(message.params.num_machines)
        self.control.update_machine_shop_params(message.params)


if __name__ == "__main__":
    SimulationApp().run()

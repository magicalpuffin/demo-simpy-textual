"""
Machine shop example

https://simpy.readthedocs.io/en/latest/examples/machine_shop.html

Covers:

- Interrupts
- Resources: PreemptiveResource

Scenario:
  A workshop has *n* identical machines. A stream of jobs (enough to
  keep the machines busy) arrives. Each machine breaks down
  periodically. Repairs are carried out by one repairman. The repairman
  has other, less important tasks to perform, too. Broken machines
  preempt these tasks. The repairman continues them when he is done
  with the machine repair. The workshop works continuously.

"""

import random
from dataclasses import dataclass, field
from typing import Any, TypedDict

import simpy
from simpy.resources.store import StoreGet, StorePut

# # fmt: off
# RANDOM_SEED = 42
# PT_MEAN = 10.0         # Avg. processing time in minutes
# PT_SIGMA = 2.0         # Sigma of processing time
# MTTF = 300.0           # Mean time to failure in minutes
# BREAK_MEAN = 1 / MTTF  # Param. for expovariate distribution
# REPAIR_TIME = 30.0     # Time it takes to repair a machine in minutes
# JOB_DURATION = 30.0    # Duration of other jobs in minutes
# NUM_MACHINES = 10      # Number of machines in the machine shop
# WEEKS = 4              # Simulation time in weeks
# SIM_TIME = WEEKS * 7 * 24 * 60  # Simulation time in minutes
# # fmt: on


@dataclass
class MachineParams:
    mean_process_time: float
    stdv_process_time: float
    mean_time_to_failure: float
    repair_time: float


@dataclass
class MachineShopParams:
    num_machines: int = 5
    num_repairman: int = 1
    mean_time_to_arrive: float = 2
    machine_params: MachineParams = field(
        default_factory=lambda: MachineParams(
            mean_process_time=10,
            stdv_process_time=2,
            mean_time_to_failure=300,
            repair_time=30,
        )
    )


class MachineMetrics(TypedDict):
    time: float | int
    name: str
    part_id: int | None
    parts_made: int
    broken: bool
    broken_duration: float | int
    idle_duration: float | int


class MachineShopMetrics(TypedDict):
    time: float | int
    queue_items: int
    total_parts_made: int
    total_broken_duration: float | int
    total_idle_duration: float | int


class Machine:
    """A machine produces parts and may get broken every now and then.

    If it breaks, it requests a *repairman* and continues the production
    after the it is repaired.

    A machine has a *name* and a number of *parts_made* thus far.

    """

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        repairman: simpy.PreemptiveResource,
        store: simpy.Store,
        params: MachineParams,
    ):
        self.env = env
        self.name = name
        self.parts_made = 0
        self.part_id: int | None = None
        self.broken = False

        self.machine_start = 0
        self.broken_duraton = 0
        self.idle_duration = 0

        self.log_parts: list[tuple[float, int]] = [(0, 0)]
        self.log_broken_duration: list[tuple[float, float]] = [(0, 0)]
        self.log_idle_duration: list[tuple[float, float]] = [(0, 0)]

        # Start "working" and "break_machine" processes for this machine.
        self.process = env.process(
            self.working(
                repairman,
                store,
                params.mean_process_time,
                params.stdv_process_time,
                params.repair_time,
            )
        )
        env.process(self.break_machine(params.mean_time_to_failure))

    def increment_idle_duration(self, duration: float | int):
        self.idle_duration += duration
        self.log_idle_duration.append((self.env.now, self.idle_duration))
        self.idle_ratio = self.idle_duration / (self.env.now - self.machine_start)

    def working(
        self,
        repairman: simpy.PreemptiveResource,
        store: simpy.Store,
        mean_process_time: float,
        stdv_process_time: float,
        repair_time: float,
    ):
        """Produce parts as long as the simulation runs.

        While making a part, the machine may break multiple times.
        Request a repairman when this happens.

        """
        self.machine_start = self.env.now
        while True:
            # Start making a new part
            idle_start = self.env.now
            self.part_id = yield store.get()
            self.increment_idle_duration(self.env.now - idle_start)
            done_in = self.time_per_part(mean_process_time, stdv_process_time)
            while done_in:
                start = self.env.now
                try:
                    # Working on the part
                    yield self.env.timeout(done_in)
                    done_in = 0  # Set to 0 to exit while loop.

                except simpy.Interrupt:
                    self.broken = True
                    broken_start = self.env.now
                    done_in -= self.env.now - start  # How much time left?

                    # Request a repairman. This will preempt its "other_job".
                    with repairman.request(priority=1) as req:
                        yield req
                        yield self.env.timeout(repair_time)

                    self.broken = False
                    self.broken_duraton += self.env.now - broken_start
                    self.log_broken_duration.append((self.env.now, self.broken_duraton))

            # Part is done.
            self.part_id = None
            self.parts_made += 1
            self.log_parts.append((self.env.now, self.parts_made))

    def break_machine(self, mean_time_to_failure: float):
        """Break the machine every now and then."""
        while True:
            yield self.env.timeout(self.time_to_failure(mean_time_to_failure))
            if not self.broken and self.part_id:
                # Only break the machine if it is currently working.
                self.process.interrupt()

    def time_per_part(self, mean: float, stdv: float):
        """Return actual processing time for a concrete part."""
        t = random.normalvariate(mean, stdv)
        # The normalvariate can be negative, but we want only positive times.
        while t <= 0:
            t = random.normalvariate(mean, stdv)
        return t

    def time_to_failure(self, mean_time_to_failure: float):
        """Return time until next failure for a machine."""
        return random.expovariate(1 / mean_time_to_failure)


def other_jobs(
    env: simpy.Environment, repairman: simpy.PreemptiveResource, job_duration: float
):
    """The repairman's other (unimportant) job."""
    while True:
        # Start a new job
        done_in = job_duration
        while done_in:
            # Retry the job until it is done.
            # Its priority is lower than that of machine repairs.
            with repairman.request(priority=2) as req:
                yield req
                start = env.now
                try:
                    yield env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    done_in -= env.now - start


def part_arrival(
    env: simpy.Environment, store: simpy.Store, mean_time_to_arrive: float
):
    part_num = 1
    while True:
        yield env.timeout(random.expovariate(1 / mean_time_to_arrive))
        yield store.put(part_num)
        part_num += 1


class MonitorStore(simpy.Store):
    def __init__(self, env: simpy.Environment, capacity: float | int = float("inf")):
        self.log_queue: list[tuple[float, int]] = []
        super().__init__(env, capacity)

    def put(self, item: Any) -> StorePut:
        self.log_queue.append((self._env.now, len(self.items)))
        return super().put(item)

    def get(self) -> StoreGet:
        self.log_queue.append((self._env.now, len(self.items)))
        return super().get()


class MachineShop:
    def __init__(self, env: simpy.Environment, params: MachineShopParams) -> None:
        self.env = env
        self.repairman = simpy.PreemptiveResource(env, capacity=params.num_repairman)
        self.store = MonitorStore(env)
        self.machines = [
            Machine(
                env,
                f"Machine_{i + 1}",
                self.repairman,
                self.store,
                params.machine_params,
            )
            for i in range(params.num_machines)
        ]

        self.metrics_log: list[MachineShopMetrics] = []
        self.machine_metrics_log: list[list[MachineMetrics]] = [
            [] for i in range(params.num_machines)
        ]

        self.env.process(part_arrival(env, self.store, params.mean_time_to_arrive))
        self.env.process(other_jobs(env, self.repairman, 30))
        self.env.process(self.monitor_metrics(1))

    def monitor_metrics(self, freq: float):
        while True:
            for i, machine in enumerate(self.machines):
                self.machine_metrics_log[i].append(
                    {
                        "time": self.env.now,
                        "name": machine.name,
                        "broken": machine.broken,
                        "broken_duration": machine.broken_duraton,
                        "part_id": machine.part_id,
                        "parts_made": machine.parts_made,
                        "idle_duration": machine.idle_duration,
                    }
                )
            total_parts_made = sum(
                [
                    machine_log[-1]["parts_made"]
                    for machine_log in self.machine_metrics_log
                ]
            )
            total_broken_duration = sum(
                [
                    machine_log[-1]["broken_duration"]
                    for machine_log in self.machine_metrics_log
                ]
            )
            total_idle_duration = sum(
                [
                    machine_log[-1]["idle_duration"]
                    for machine_log in self.machine_metrics_log
                ]
            )

            self.metrics_log.append(
                {
                    "time": self.env.now,
                    "queue_items": len(self.store.items),
                    "total_parts_made": total_parts_made,
                    "total_broken_duration": total_broken_duration,
                    "total_idle_duration": total_idle_duration,
                }
            )
            yield self.env.timeout(freq)


# # Setup and start the simulation
# print("Machine shop")
# random.seed(RANDOM_SEED)  # This helps to reproduce the results

# # Create an environment and start the setup process
# env = simpy.Environment()
# sim = MachineShop(
#     env=env,
#     num_machines=5,
#     num_repairman=1,
#     mean_time_to_arrive=2,
#     mean_process_time=10,
#     stdv_process_time=2,
#     mean_time_to_failure=300,
#     repair_time=30,
# )


# # Execute!
# env.run(until=SIM_TIME)

# # Analysis/results
# print(f"Machine shop results after {WEEKS} weeks")
# for machine in sim.machines:
#     print(f"{machine.name} made {machine.parts_made} parts.")

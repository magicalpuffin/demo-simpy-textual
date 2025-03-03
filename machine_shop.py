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

import simpy

# fmt: off
RANDOM_SEED = 42
PT_MEAN = 10.0         # Avg. processing time in minutes
PT_SIGMA = 2.0         # Sigma of processing time
MTTF = 300.0           # Mean time to failure in minutes
BREAK_MEAN = 1 / MTTF  # Param. for expovariate distribution
REPAIR_TIME = 30.0     # Time it takes to repair a machine in minutes
JOB_DURATION = 30.0    # Duration of other jobs in minutes
NUM_MACHINES = 10      # Number of machines in the machine shop
WEEKS = 4              # Simulation time in weeks
SIM_TIME = WEEKS * 7 * 24 * 60  # Simulation time in minutes
# fmt: on


def time_per_part():
    """Return actual processing time for a concrete part."""
    t = random.normalvariate(PT_MEAN, PT_SIGMA)
    # The normalvariate can be negative, but we want only positive times.
    while t <= 0:
        t = random.normalvariate(PT_MEAN, PT_SIGMA)
    return t


def time_to_failure():
    """Return time until next failure for a machine."""
    return random.expovariate(BREAK_MEAN)


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
    ):
        self.env = env
        self.name = name
        self.parts_made = 0
        self.part_id: int | None = None
        self.broken = False

        # Start "working" and "break_machine" processes for this machine.
        self.process = env.process(self.working(repairman, store))
        env.process(self.break_machine())

    def working(self, repairman: simpy.PreemptiveResource, store: simpy.Store):
        """Produce parts as long as the simulation runs.

        While making a part, the machine may break multiple times.
        Request a repairman when this happens.

        """
        while True:
            # Start making a new part
            self.part_id = yield store.get()
            done_in = time_per_part()
            while done_in:
                start = self.env.now
                try:
                    # Working on the part
                    yield self.env.timeout(done_in)
                    done_in = 0  # Set to 0 to exit while loop.

                except simpy.Interrupt:
                    self.broken = True
                    done_in -= self.env.now - start  # How much time left?

                    # Request a repairman. This will preempt its "other_job".
                    with repairman.request(priority=1) as req:
                        yield req
                        yield self.env.timeout(REPAIR_TIME)

                    self.broken = False

            # Part is done.
            self.part_id = None
            self.parts_made += 1

    def break_machine(self):
        """Break the machine every now and then."""
        while True:
            yield self.env.timeout(time_to_failure())
            if not self.broken and self.part_id:
                # Only break the machine if it is currently working.
                self.process.interrupt()


def other_jobs(env: simpy.Environment, repairman: simpy.PreemptiveResource):
    """The repairman's other (unimportant) job."""
    while True:
        # Start a new job
        done_in = JOB_DURATION
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


class MachineShop:
    def __init__(
        self, env: simpy.Environment, num_machines: int, repair_capacity: int
    ) -> None:
        self.env = env
        self.repairman = simpy.PreemptiveResource(env, capacity=repair_capacity)
        self.store = simpy.Store(env)
        self.machines = [
            Machine(env, f"Machine {i + 1}", self.repairman, self.store)
            for i in range(num_machines)
        ]
        self.env.process(part_arrival(env, self.store, 1.5))
        self.env.process(other_jobs(env, self.repairman))


# Setup and start the simulation
print("Machine shop")
random.seed(RANDOM_SEED)  # This helps to reproduce the results

# Create an environment and start the setup process
env = simpy.Environment()
sim = MachineShop(env, 5, 1)


# Execute!
env.run(until=SIM_TIME)

# Analysis/results
print(f"Machine shop results after {WEEKS} weeks")
for machine in sim.machines:
    print(f"{machine.name} made {machine.parts_made} parts.")

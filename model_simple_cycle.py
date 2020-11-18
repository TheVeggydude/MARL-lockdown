from agent.agent import Agent, State, Parameters
from matplotlib import pyplot as plt
from utils.strconv import params2str


a = 0.2   # exposure rate
b = 1.75  # Measure of time to infection once exposed
g = 0.5   # Rate of recovery
d = 0     # Rate at which loss of immunity occurs.
r = 1     # Cost level??

init_agent_a = State(0.99, 0, 0.01, 0, 10000)
init_agent_b = State(1.0, 0, 0, 0, 10000)
init_agent_c = State(1.0, 0, 0, 0, 10000)

initial_params = Parameters(a, b, g, d, r)
agent_a = Agent(init_agent_a, initial_params)
agent_b = Agent(init_agent_b, initial_params)
agent_c = Agent(init_agent_c, initial_params)

agents = [agent_a, agent_b, agent_c]

print(f"Agent a initial: {init_agent_a}")
print(f"Agent b initial: {init_agent_b}")
print(f"Agent c initial: {init_agent_c}")

# Simulate for 100 iterations
for _ in range(100):

    # Migration step, for a cyclic, one-directional graph
    for index, agent in enumerate(agents):
        next_index = index+1 if index+1 < len(agents) else 0

        # Perform migration
        migration = agent.emigrate(0.02)
        agents[next_index].immigrate(migration)

    # Iterate the agents, which now includes the migrated population.
    for agent in agents:
        agent.iterate()

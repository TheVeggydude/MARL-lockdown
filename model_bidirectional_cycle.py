from agent.agent import Agent
from utils.state import State, Parameters
from utils.plotting import plot_agent_history, plot_compartment_comparison


a = 0.2   # exposure rate
b = 1.75  # Measure of time to infection once exposed
g = 0.5   # Rate of recovery
d = 0     # Rate at which loss of immunity occurs.
r = 1     # Cost level??

init_agent_a = State(0.99, 0, 0.01, 0, 10000)
init_agent_b = State(1.0, 0, 0, 0, 10000)
init_agent_c = State(.98, 0.02, 0, 0, 10000)

initial_params = Parameters(a, b, g, d, r)
agent_a = Agent('A', init_agent_a, initial_params)
agent_b = Agent('B', init_agent_b, initial_params)
agent_c = Agent('C', init_agent_c, initial_params)

agents = [agent_a, agent_b, agent_c]

print(f"Agent a initial: {init_agent_a}")
print(f"Agent b initial: {init_agent_b}")
print(f"Agent c initial: {init_agent_c}")

# Simulate for 100 iterations
for _ in range(80):

    # Migration step, for a cyclic, one-directional graph
    for index, agent in enumerate(agents):
        next_index = index+1 if index+1 < len(agents) else 0

        # Perform migration
        migration = agent.emigrate(0.02)
        agents[next_index].immigrate(migration)

    # Now also do it in reverse order
    for index, agent in enumerate(agents):
        next_index = index - 1 if index - 1 >= 0 else len(agents) - 1

        # Perform migration
        migration = agent.emigrate(0.02)
        agents[next_index].immigrate(migration)

    # Iterate the agents, which now includes the migrated population.
    for agent in agents:
        agent.iterate_rl()

plot_compartment_comparison(agents, 1, "Exposed")
plot_compartment_comparison(agents, 2, "Infected")

for agent in agents:
    plot_agent_history(agent)

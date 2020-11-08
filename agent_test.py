from agent.agent import Agent, State, Parameters
from matplotlib import pyplot as plt


def params2str(params):
    alpha, beta, gamma, delta, rho = params
    return f"alpha: {alpha}, beta: {beta}, gamma: {gamma}, delta {delta}, rho: {rho}"


a = 0.2   # exposure rate
b = 1.75  # Measure of time to infection once exposed
g = 0.5   # Rate of recovery
d = 0     # Rate at which loss of immunity occurs.
r = 1     # Cost level??

init_agent_a = State(0.99, 0, 0.01, 0, 1000)
init_agent_b = State(0.90, 0.05, 0.03, 0.02, 1000)
initial_params = Parameters(a, b, g, d, r)

agent_a = Agent(init_agent_a, initial_params)
agent_b = Agent(init_agent_b, initial_params)

print(f"Agent a initial: {init_agent_a}")
print(f"Agent b initial: {init_agent_b}")

# take 2% of the population from agent a
migration = agent_a.emigrate(0.02)
print(f"Migration from a to b: {migration}")

# and put it in agent b's population
agent_b.immigrate(migration)

print(f"Agent a post migration: {agent_a.state()}")
print(f"Agent b post migration: {agent_b.state()}")

# for _ in range(100):
#     agent_a.iterate()
#
# # Plot results
# plt.plot(agent_a.history()[:, :4])
# plt.title(f"SEIR model ({params2str(initial_params)})")
# plt.ylabel("Population fraction")
# plt.xlabel("Time (Days)")
# plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])
# plt.savefig(f"results/test_agent.png")
# plt.show()

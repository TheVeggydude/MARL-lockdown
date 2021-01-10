from agent.agent import Agent
from utils.state import State, Parameters
from matplotlib import pyplot as plt
from utils.strconv import params2str


a = 0.2     # exposure rate
b = 1.75    # Measure of time to infection once exposed
g = 0.5     # Rate of recovery
d = 0.05     # Rate at which loss of immunity occurs.
r = 0.4     # Effectiveness of measures (range 0-1, lower is more effective)

init_agent_a = State(0.99, 0, 0.01, 0, 10000)

initial_params = Parameters(a, b, g, d, r)
agent_a = Agent('test', init_agent_a, initial_params)

print(f"Agent a initial: {init_agent_a}")
# print(f"Agent b initial: {init_agent_b}")
# print(f"Agent c initial: {init_agent_c}")

# # take 2% of the population from agent a
# migration = agent_a.emigrate(0.02)
# print(f"Migration from a to b: {migration}")
#
# # and put it in agent b's population
# agent_b.immigrate(migration)
#
# print(f"Agent a post migration: {agent_a.state()}")
# print(f"Agent b post migration: {agent_b.state()}")

# Simulate for i iterations
i = 1
print(f"Iterations remaining {i}")

iterations_list = []
agent_iterations = 0

while True:

    agent_a.iterate()
    agent_iterations += 1

    # Explore until goal is found, then try again
    if agent_a.at_goal():
        i -= 1

        # Train until done
        if i == 0:
            break

        # Reset agent in when retraining
        iterations_list.append(agent_iterations)
        agent_iterations = 0
        agent_a.reset()
        agent_a.set_state(init_agent_a)
        print(f"Iterations remaining {i}")

print(f"Iterations over time: {iterations_list}")

# Plot results
plt.plot(agent_a.history()[:, :4])
plt.title(f"SEIR model ({params2str(initial_params)})")
plt.ylabel("Population fraction")
plt.xlabel("Time (Days)")
plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])
plt.savefig(f"results/test_agent.png")
plt.show()

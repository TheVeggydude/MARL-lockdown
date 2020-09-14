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

initial_state = State(0.99, 0, 0.01, 0)
initial_params = Parameters(a, b, d, g, r)
test = Agent(initial_state, initial_params)

for _ in range(100):
    test.iterate()

print(test.history())

# Plot results
plt.plot(test.history()[:, :4])
plt.title(f"SEIR model ({params2str(initial_params)})")
plt.ylabel("Population fraction")
plt.xlabel("Time (Days)")
plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])
plt.savefig(f"results/test_agent.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def params2str(params):
    alpha, beta, gamma, delta, rho = params
    return f"alpha: {alpha}, beta: {beta}, gamma: {gamma}, delta {delta}, rho: {rho}"


def seir_model_with_measurements(init_vals, params, t):
    S_0, E_0, I_0, R_0, C_0 = init_vals
    S, E, I, R, C = [S_0], [E_0], [I_0], [R_0], [C_0]
    alpha, beta, gamma, delta, rho = params
    dt = t[1] - t[0]

    for _ in t[1:]:
        next_S = S[-1] - (rho*beta*S[-1]*I[-1])*dt + (delta*R[-1])*dt  # Add fraction of recovered compartment.
        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt - (delta*R[-1])*dt  # Remove fraction of recovered compartment.
        next_C = C[-1] + (rho*3)*dt  # Cost!

        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
        C.append(next_C)

    return np.stack([S, E, I, R, C]).T


# Define parameters
t_max = 200
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)

N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0, 0
alpha = 0.2  # exposure rate
beta = 1.75   # Measure of time to infection once exposed
gamma = 0.5  # Rate of recovery
delta = 0    # Rate at which loss of immunity occurs.
rho = 1

params = alpha, beta, gamma, delta, rho

# Run simulation
results = seir_model_with_measurements(init_vals, params, t)

# Plot results
plt.plot(results[:, :4])
plt.title(f"SEIR model ({params2str(params)})")
plt.ylabel("Population fraction")
plt.xlabel("Time (Days)")
plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])
plt.savefig(f"results/test_agent.png")
plt.show()

# Plot cost
plt.plot(results[:, -1])
plt.title("SEIR model measurement cost")
plt.ylabel("Cost")
plt.xlabel("Time (Days)")
plt.savefig("results/test_agent_cost.png")
plt.show()

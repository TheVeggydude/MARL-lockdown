import numpy as np
import matplotlib.pyplot as plt

from agent.agent import Agent

# N = 10000
# init_vals = 1 - 1/N, 1/N, 0, 0
#
# alpha = 0.2
# beta = 1.75
# gamma = 0.5
# params = alpha, beta, gamma
#
# test_agent = Agent(init_vals, [alpha, beta, gamma], [0.3, 0.3])
#
# for i in range(100):
#     test_agent.next()
#
# # print(history)
# plt.plot(test_agent.comps[:, 0])
# plt.title(f'Single agent SEIR model ({[alpha, beta, gamma]})')
# plt.show()


def params2str(params):
    alpha, beta, gamma, rho = params
    return f"alpha: {alpha}, beta: {beta}, gamma: {gamma}, rho: {rho}"


def seir_model_with_measurements(init_vals, params, t):
    S_0, E_0, I_0, R_0, C_0 = init_vals
    S, E, I, R, C = [S_0], [E_0], [I_0], [R_0], [C_0]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]

    for _ in t[1:]:
        next_S = S[-1] - (rho*beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        next_C = C[-1] + (rho*3)*dt  # Cost!

        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
        C.append(next_C)

    return np.stack([S, E, I, R, C]).T


# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
print(t)

N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0, 0
alpha = 0.2
beta = 1.75
gamma = 0.5
rho = 1

params = alpha, beta, gamma, rho

# Run simulation
results = seir_model_with_measurements(init_vals, params, t)

# Plot results
plt.plot(results[:, :4])
plt.title(f"SEIR model ({params2str(params)})")
plt.ylabel("Population fraction")
plt.xlabel("Time (Days)")
plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])
plt.savefig(f"results/test_agent.jpg")
plt.show()

# Plot cost
plt.plot(results[:, -1])
plt.title("SEIR model measurement cost")
plt.ylabel("Cost")
plt.xlabel("Time (Days)")
plt.savefig("results/test_agent_cost.jpg")
plt.show()

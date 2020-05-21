import numpy as np
import matplotlib.pyplot as plt

from agent.agent import Agent

test_agent = Agent([900, 0, 100, 0], [0.7, 0.5, 0.1], [0.3, 0.3])

history = np.array([test_agent.comps])
for i in range(100):
    test_agent.next()
    history = np.append(history, [test_agent.comps], axis=0)

print(history)
plt.plot(history[:, 0])
plt.show()

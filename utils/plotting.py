from matplotlib import pyplot as plt


def plot_compartment_comparison(agents, idx, comp_name):
    # Plot desired compartment
    for curr_agent in agents:
        plt.plot(curr_agent.history()[:, idx])

    # Add information
    plt.title(f"{comp_name} comparison for agents")
    plt.ylabel("Population fraction")
    plt.xlabel("Time (days)")
    plt.legend([agent.name for agent in agents])
    plt.savefig(f"results/{comp_name}_comparison.png")
    plt.show()


def plot_agent_history(curr_agent):
    plt.plot(curr_agent.history()[:, :4])

    # Add information
    plt.title(f"History for {curr_agent.name}")
    plt.ylabel("Population fraction")
    plt.xlabel("Time (days)")
    plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])
    plt.savefig(f"results/{curr_agent.name}_history.png")
    plt.show()
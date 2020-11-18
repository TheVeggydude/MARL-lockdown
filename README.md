# MARL-lockdown
A Multi-Agent Reinforcement Learning project that learns to solve the Coordinated Lockdown problem.

## Assumptions
Some assumptions have been made when writing this code. They are explained here.
- First of all, since we use the SEIRS model, we automatically use all of the assumptions that come with that model. For
more information about the model and the assumptions, see [here](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#Other_considerations_within_compartmental_epidemic_models).
- When migrating populations, we assume a slice of the population with a distribution equal to the general population of
the agent migrates to another agent. This means that what whatever compartment a population may belong to, they will all
have the same 'desire' to move.
- Moreover, when migrating we converts between distributions and actual population numbers. In order to guarantee consistency
and human-readable numbers we truncate all floating point population numbers before updating the agent populations.
- The migration rates are static. This means the containment level does not directly correlate to a change in migration.
However, the containment level will influence the spread of the disease and thus change the distribution of the migrating
slice. 
- The containment level is an abstraction from actual implementations of containment. It represents the restriction of
civilian actions and a decrease in conventional spending and reflects an abstract notion of the measures taken by an agent
to combat the exposure of its population to the infected people.
- The cost function is an abstraction of the idea of strain on the people and economy of an agent (which in our model represents
a country or other, similar, larger area).

## Goals
In the end we want to have a tool that allows us to research Reinforcement Learning techniques in a Multi Agent
environment, specifically when used to learn containment policies.

In practice, this means creating a mapping of the state (the SEIR compartments) to an abstract value we call the 'containment level'.
This value, as mentioned in the assumptions, reflects an abstract notion of the measures taken by an agent to combat the
exposure of its population to the infected people.

## Cost function


## Acknowledgements
The model is based on a lot of previous work, such as [this overview](https://institutefordiseasemodeling.github.io/Documentation/hiv/model-seir.html) and [this explanation](https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296).

# Existence of Autonomous Vehicles Coalitions in Mixed Traffic: A Game-Theoretical Analysis

> Connected Autonomous Vehicles (CAVs) are being gradually introduced into urban environments and are expected to operate at scale alongside human-driven vehicles (HDVs) in the coming years. Equipped with connectivity and optimization capabilities, CAVs may form cooperative coalitions or clubs that improve their travel times, potentially at the expense of others. In this paper, we demonstrate that such coalitions can emerge in mixed-autonomy traffic networks with adaptive traffic lights. Using the microscopic traffic simulator SUMO, we simulate a stylized but illustrative urban network with 23 vehicles, 10 CAVs, and 13 HDVs and construct payoff matrices from observed travel times. We analyze coalition stability through the lens of coalitional game theory and find that a particular coalition leads to a Strong Nash equilibrium, where no group of agents can jointly deviate to improve their outcomes. These results demonstrate how coalition formation among CAVs can alter traffic dynamics and present new challenges for the equitable and efficient design of future transportation systems.


# Installation

<!-- start installation -->

- **Prerequisite**: Make sure you have SUMO installed in your system. This procedure should be carried out separately, by following the instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).
- **Option 1**: Install the latest stable version from [PyPI](https://pypi.org/project/routerl/):  
  ```
    pip install routerl
  ```
- **Option 2**: Clone this repository for latest version, and manually install its dependencies: 
  ```
    git clone https://github.com/COeXISTENCE-PROJECT/RouteRL.git
    cd RouteRL
    pip install -r requirements.txt
  ```


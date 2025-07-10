Analysis

> Connected Autonomous Vehicles (CAVs) are being gradually introduced into urban environments and are expected to operate at scale alongside human-driven vehicles (HDVs) in the coming years. Equipped with connectivity and optimization capabilities, CAVs may form cooperative coalitions or clubs that improve their travel times, potentially at the expense of others. In this paper, we demonstrate that such coalitions can emerge in mixed-autonomy traffic networks with adaptive traffic lights. Using the microscopic traffic simulator SUMO, we simulate a stylized but illustrative urban network with 23 vehicles, 10 CAVs, and 13 HDVs and construct payoff matrices from observed travel times. We analyze coalition stability through the lens of coalitional game theory and find that a particular coalition leads to a Strong Nash equilibrium, where no group of agents can jointly deviate to improve their outcomes. These results demonstrate how coalition formation among CAVs can alter traffic dynamics and present new challenges for the equitable and efficient design of future transportation systems.


---

# Traffic Network

> We study a two-route congested network, where vehicles go from the same origin $A$ to the same destination $B$. They are given the choice between route $0$ and route $1$, considering route $0$ is shorter than route $1$, but congestion on either of these two routes may increase travel times. We define the the *Two-Route* network by a given set of parameters: the extra length of route 1 over route 0 depends on a single parameter $x$; congestion is dependent on the set of agents starting at node A, their start times and their behaviours; finally, the junction J uses a traffic light system, working on cycles. 


<p align="center">
  <img src="imgs/network.png" width="50%"/>
</p>

---


# Club Formation

> We showcase an instance of *Two-Route* which allows the formation of a club. In this instance, 15 agents drive through the network, and 5 of them are *human agents* fixed to route 0, which leaves $N = 10$ AV agents. The figure below depicts the **travel times of all AVs** are compared across 5 joint actions. When there is no coalition ($x^0$), when agents [1, 5, 6] are part of the coalition, [1, 5, 6, 7] consist the coalition, [0, 1, 5, 6, 7] are part of the coalition and [0, 1, 5, 6]. For each agent $i$, travel times are normalized across the 5 plots such that the travel time of all agents in $x^0$ is 1. For each action, the club (= the set of agents driving on route 1) is indicated by white dots while agents on route 0 are indicated by black dots. Black (respectively red) arrows show how much time each agent would lose (respectively gain) by changing route unilaterally from a given joint action. The absence of red arrows is therefore equivalent to the joint action being a Nash equilibrium. Blue arrows connect cases where an agent can gain time by changing route unilaterally to the action resulting from this change.



<p align="center">
  <img src="imgs/deviations.png" width="100%"/>
</p>

---


## Installation


- **Prerequisite**: Make sure you have SUMO installed in your system. This procedure should be carried out separately, by following the instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).
- **Option 1**: Install the latest stable version from [PyPI](https://pypi.org/project/routerl/):  
  ```
    pip3 install git+https://github.com/COeXISTENCE-PROJECT/RouteRL.git@urb  
  ```
- **Option 2**: Clone this repository for latest version, and manually install its dependencies: 
  ```
    git clone https://github.com/COeXISTENCE-PROJECT/RouteRL.git
    cd RouteRL
    pip install -r requirements.txt
  ```

---

## Contents
* [Experiments](https://github.com/COeXISTENCE-PROJECT/Coalition_formation_in_mixed_traffic_with_AVs_/tree/main/experiments)
* [Results](https://github.com/COeXISTENCE-PROJECT/Coalition_formation_in_mixed_traffic_with_AVs_/tree/main/results)

---

### Credits

This work is part of [COeXISTENCE](https://www.rafalkucharskilab.pl/research/coexistence/) (ERC Starting Grant, grant agreement No 101075838) and is a team work at Jagiellonian University in Kraków, Poland by: [Natello Descormier](https://github.com/natdesc), [Anastasia Psarou](https://github.com/AnastasiaPsarou), [Grzegorz Jamroz](https://github.com/GrzegorzJamroz) and others, within the research group of [Rafał Kucharski](https://www.rafalkucharskilab.pl).

<p align="center">
  <img src="imgs/credits.png" width="50%"/>
</p>
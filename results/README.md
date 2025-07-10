In the **Results** folder you will find data from the Results section of the paper, as well as the code used to produce the figures shown in the paper.

> We showcase an instance of *Two-Route* which allows the formation of a club. In this instance, 15 agents drive through the network, and 5 of them are *human agents* fixed to route 0, which leaves $N = 10$ AV agents. The figure below depicts the **travel times of all AVs** are compared across 5 joint actions. When there is no coalition ($x^0$), when agents [1, 5, 6] are part of the coalition, [1, 5, 6, 7] consist the coalition, [0, 1, 5, 6, 7] are part of the coalition and [0, 1, 5, 6].

## Dynamic traffic light systems

> We define **traffic light systems** as applications $\varphi$ assigning to each $x \in X$ a traffic light cycle defined by parameters $(tl_0,tl_1,tl_y)$. We call the system *static* when to all $x \in X$ is assigned the same traffic light cycle, and *dynamic* when this mapping depends on the vehicle flows on each route $q(x) = (q_0(x),q_1(x))$ with $q_0(x) = |\{i \in \N : x_i = 0\}|$ and $q_1(x) = N - q_0(x) = |\{i \in \N : x_i = 1\}|$.

> Dynamic traffic light systems allow us to model how traffic lights adapt to incoming vehicle flows. We make the assumption that the parameters of the traffic light cycles depend *only, and automatically*, on the total number of vehicles committing to each route. For example, one may imagine that detectors are located far enough ahead of the traffic light junction to count the number of cars on each route before any of them reaches the intersection, or alternatively, that travel choices are fixed on a set number of iterations (or days of simulation), then that the traffic light system adapts to the results of previous experiments, with an inertia of one or several days.

## Choosing a *Two-Route* instance

> We assume that in this selfish atomic routing game \[Roughgarden, 2007\], the scenario where all AV agents choose route 0 is simultaneously a selfish and a social *Wardrop equilibrium*. That is, respectively, no vehicle can reduce its travel time by changing its route alone, and no other mapping from agents to route choices leads to a lower average travel time \[Wardrop, 1952\].

This "Wardrop equilibrium" translates in our model to a Nash equilibrium. Therefore, our objective is to find an instance of *Two-Route* where the situation with all AV agents on route 0 is a Nash equilibrium (verifying our Wardrop equilibrium assumption) but not a strong equilibrium (so a coalition can exist). Such instances are labeled by yellow dots in the following figure:

<p align="center">
  <img src="imgs/15 agents 50 seconds cycle.png" width="50%"/>
</p>

---

## Links
* [Full repo](https://github.com/COeXISTENCE-PROJECT/Coalition_formation_in_mixed_traffic_with_AVs_/tree/main)
* [Experiments](https://github.com/COeXISTENCE-PROJECT/Coalition_formation_in_mixed_traffic_with_AVs_/tree/main/experiments)

---

### Credits

This work is part of [COeXISTENCE](https://www.rafalkucharskilab.pl/research/coexistence/) (ERC Starting Grant, grant agreement No 101075838) and is a team work at Jagiellonian University in Kraków, Poland by: [Natello Descormier](https://github.com/natdesc), [Anastasia Psarou](https://github.com/AnastasiaPsarou), [Grzegorz Jamróz](https://github.com/GrzegorzJamroz) and others, within the research group of [Rafał Kucharski](https://www.rafalkucharskilab.pl).

<p align="center">
  <img src="imgs/credits.png" width="50%"/>
</p>
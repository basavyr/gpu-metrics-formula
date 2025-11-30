# GPU Quality Score

A mathematical formalism to establish GPU *quality scores*, which can be used for comparison of a multitude of GPUs.

The formalism is based on several steps that need to be followed to determine a GPU *quality* score. These scores (as it will be described in the following sections) will result in concise numbers between $0$ and $1$, where a higher score will signify a better choice in terms of overall performance. The key thing here is to understand that **performance** does not only mean raw compute power, but will also relate in the GPU price and its consumption. 

> This methodology is constructed mainly for **AI workloads** when the budget and efficiency are crucial, thus taking a decision on either purchasing or allocating GPU-as-a-resource (e.g., [runpod](https://www.runpod.io/)) will benefit the most from it. Gaming is not considered (yet).

## Steps to determine GPU quality score
1. define **key metrics** to be used throughout the formalism -> In this current approach, the following are considered:
    - VRAM - $M_j$ (given in GP, HiB)
    - Memory bandwidth - $B_j$ (given in GB/s, HiB)
    - Power consumption - $P_j$ (given in W, LiB)
    - The cost - $C_j$ (given in USD/RON, LiB)
2. for each metric, fix a *reference value*, which will be used to benchmark a specific GPU within the context of that metric's typical range. For example, when considering $M_j$, a reference value of $32\text{GB}$ VRAM will be considered.
    - A reference table will be defined within the formalism.
3. apply concrete **transformation** functions $f_T$ on all metrics, taking into consideration the normalization and the significance of the value (i.e., *higher is better* - HiB or *lower is better* - LiB).
4. set a specific group of **weights** for each metric to decide on their relative importance when determine the final score.
5. The final aggregator mechanism that will generate the quality score can be decided based on the **geometric mean** with the weights defined at the previous step or via the [CES aggregator](https://en.wikipedia.org/wiki/Constant_elasticity_of_substitution).

## Mathematical formalism.

TDB (wip).
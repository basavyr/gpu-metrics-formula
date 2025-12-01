# GPU Quality Score

A mathematical formalism to compute GPU *quality scores* for objective comparison across many models.

The goal is to convert heterogeneous GPU specifications into a single numeric score in the range **0–1**, where a **higher score** indicates a better trade-off between performance, efficiency, and cost. 

> The current methodology is optimized for **AI / deep-learning workloads**, not gaming, as it does not take any typical benchmark results into consideration. It rather just focuses purely on the actual GPU features themselves.

## Usage

The implementation is available via [`test_gpu.py`](src/test_gpu.py).  
Example (for an NVIDIA RTX 5080 from a Romanian retail store — **price in RON**):

```bash
python3 test_gpu.py --name RTX_5080 --vram 16 --bandwidth 960 --tdp 360 --cost 6400
```

## Formal Methodology

### 1. Metrics

Each GPU $j$ is characterized by four metrics:

- **VRAM capacity**: $M_j$ (GB, *higher is better — HiB*)
- **Memory bandwidth**: $B_j$ (GB/s, *HiB*)
- **Power consumption (TDP)**: $P_j$ (W, *lower is better — LiB*)
- **Cost**: $C_j$ (USD/RON but currently is in RON, *LiB*)

These four factors correspond to the essential constraints in deep-learning workloads:
batch size, training throughput, energy cost, and budget.

### 2. Reference Values

Each metric is normalized using a fixed **reference value**:

$$
M_{\text{ref}},\quad B_{\text{ref}},\quad P_{\text{ref}},\quad C_{\text{ref}}.
$$

Currently, the chosen references are **friendly-budget oriented** for typical AI workloads:

| Metric | Reference | Meaning |
|--------|-----------|---------|
| $M_{\text{ref}}$ | 32 GB | reference VRAM |
| $B_{\text{ref}}$ | 1000 GB/s | typical memory bandwidth |
| $P_{\text{ref}}$ | 400 W | practical TDP ceiling |
| $C_{\text{ref}}$ | 3500 RON | baseline friendly budget |

### 3. Metric Transformations

Raw values are transformed into **utility scores** using monotonic functions $f_T(\cdot)$:

$$
u_{T,j} = \frac{f_T(x_{T,j})}{f_T(x_{T,\text{ref}})}.
$$

Typical choices:

- VRAM:  $f_M(x) = \ln(1+x)$  
- Bandwidth: $f_B(x) = \sqrt{x}$  
- Power (LiB): $f_P(x) = \ln\!\left(1 + \frac{1}{x}\right)$  
- Cost (LiB): $f_C(x) = \ln\!\left(1 + \frac{1}{x}\right)$  

These guarantee:
- HiB metrics grow sublinearly (diminishing returns),
- LiB metrics produce higher utility for smaller values.

All utilities satisfy:  
$$
u_{T,j} > 0.
$$

### 4. Weights

Each metric receives a weight $w_T$ such that:

$$
w_M + w_B + w_P + w_C = 1.
$$

Currently, the **default weights** are chosen for balanced AI workloads:

$$
w_M = 0.25,\quad
w_B = 0.35,\quad
w_P = 0.20,\quad
w_C = 0.20.
$$

Memory and bandwidth have slightly higher importance since they directly affect training batch size and throughput.

The implementation takes the reference values and the weights from [data.py](./src/data.py) through the `GPU_REFERENCE_METRICS` and `GPU_METRICS_DEFAULT_WEIGHTS` variables. These can be adjusted accordingly if the default values do not match the requirements.

## Rationale for Reference Values and Weights

The chosen reference values are intended to reflect **friendly-budget oriented AI workloads**, where typical consumer/prosumer GPUs are targeted:

- $M_{\text{ref}} = 32$ GB: enough VRAM for medium-sized models and reasonable batch sizes.  
- $B_{\text{ref}} = 1000$ GB/s: covers most modern GDDR6/GDDR6X cards.  
- $P_{\text{ref}} = 400$ W: ensures that power-hungry GPUs are penalized without being unrealistic.  
- $C_{\text{ref}} = 3500$ RON: approximates a friendly mid-range budget for acquisition.

The **default weights** reflect the relative importance of each metric in deep-learning tasks:

- $w_M = 0.25$: VRAM is crucial for batch size and model size.  
- $w_B = 0.35$: Memory bandwidth directly affects matrix multiplication throughput (GEMM).  
- $w_P = 0.20$: Power efficiency is relevant for operational cost.  
- $w_C = 0.20$: Price efficiency controls upfront cost trade-offs.

This weighting scheme ensures that **compute-relevant metrics dominate**, while cost and power still influence the final score.

### 5. Aggregator: CES or Geometric Mean

Two mathematically consistent choices exist:

#### **A. Weighted Geometric Mean (ρ = 0 case)**  
Stable, interpretable default:

$$
S_j
= \exp\!\left( \sum_{T} w_T \ln u_{T,j} \right).
$$

#### **B. CES Aggregator (general ρ)**  
$$
S_j(\rho)
= \left( \sum_{T} w_T \, u_{T,j}^{\rho} \right)^{1/\rho}.
$$

Interpretation via elasticity of substitution:

$$
\rho = 0 \Rightarrow \text{geometric mean}, \quad
\rho > 0 \text{ allows compensation}, \quad
\rho < 0 \text{penalizes weak metrics}.
$$

Typical safe values:

$$
\rho \in \{-1.0,\,-0.5,\,0,\,0.5,\,1.0\}.
$$

## Practical Notes

1. **Numerical Stability**:  
   - For the geometric mean (ρ = 0), always compute as  
     $$
     S_j = \exp\Big(\sum_T w_T \ln u_{T,j}\Big)
     $$  
     to avoid floating-point issues with small or large utilities.  
   - Utilities $u_{T,j}$ must satisfy $u_{T,j} > 0$; clip small values to $\varepsilon \approx 10^{-6}$ if necessary.

2. **Choice of ρ in CES Aggregator**:  
   - $\rho = 0$: geometric mean (default, balanced trade-offs).  
   - $\rho > 0$: allows compensation; weak metrics can be offset by strong ones.  
   - $\rho < 0$: penalizes bottlenecks; a very low metric significantly reduces the overall score.  
   - Safe range: $\rho \in \{-1.0, -0.5, 0, 0.5, 1.0\}$.

3. **Interpretation**:  
   - $S_j \in [0,1]$: higher values indicate better overall GPU quality.  
   - The score is **relative** to the chosen references and weights; changing these will shift the scale and ranking.

4. **Extensibility**:  
   - Additional metrics (e.g., FP16 throughput, tensor cores) can be incorporated by defining new $u_{T,j}$ and updating weights $w_T$.  
   - Reference values should be updated as GPU technology evolves.

5. **Use Case Focus**:  
   - Primarily designed for **AI/deep-learning workloads** where VRAM and bandwidth dominate.  
   - Gaming or specialized workloads may require different weights or reference values.


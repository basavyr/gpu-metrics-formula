# GPU Quality Score

A mathematical formalism to compute GPU *quality scores* for objective comparison across many models.

The goal is to convert heterogeneous GPU specifications into a single numeric score in the range **0–1**, where a **higher score** indicates a better trade-off between performance, efficiency, and cost. The methodology is optimized for **AI / deep-learning workloads**, not gaming.

---

## Usage

The implementation is available via [`test_gpu.py`](src/test_gpu.py).  
Example (for an NVIDIA RTX 5080 from a Romanian retail store — **price in RON**):

```bash
python3 test_gpu.py \
    --name RTX_5080 --vram 16 --bandwidth 960 --tdp 360 --cost 6400
```

---

# Formal Methodology

## 1. Metrics

Each GPU \( j \) is characterized by four metrics:

- **VRAM capacity**: \( M_j \) (GB, *higher is better — HiB*)
- **Memory bandwidth**: \( B_j \) (GB/s, *HiB*)
- **Power consumption (TDP)**: \( P_j \) (W, *lower is better — LiB*)
- **Cost**: \( C_j \) (RON or USD, *LiB*)

These four factors correspond to the essential constraints in deep-learning workloads:
batch size, training throughput, energy cost, and budget.

---

## 2. Reference Values

Each metric is normalized using a fixed **reference value**:

\[
M_{\text{ref}},\quad B_{\text{ref}},\quad P_{\text{ref}},\quad C_{\text{ref}}.
\]

These can be set according to current-generation “typical upper bounds”.  
Example:

| Metric | Reference | Meaning |
|--------|-----------|---------|
| \( M_{\text{ref}} \) | 32 GB | reference VRAM |
| \( B_{\text{ref}} \) | 1200 GB/s | upper bandwidth of modern GPUs |
| \( P_{\text{ref}} \) | 500 W | upper TDP of high-end cards |
| \( C_{\text{ref}} \) | 3500 RON | baseline mid-range price |

---

## 3. Metric Transformations

Raw values are transformed into **utility scores** using monotonic functions \( f_T(\cdot) \):

\[
u_{T,j} \;=\; \frac{f_T(x_{T,j})}{f_T(x_{T,\text{ref}})}.
\]

Typical choices:

- VRAM:  \( f_M(x) = \ln(1+x) \)
- Bandwidth: \( f_B(x) = \sqrt{x} \)
- Power (LiB): \( f_P(x) = \ln\!\left(1 + \frac{1}{x}\right) \)
- Cost (LiB): \( f_C(x) = \ln\!\left(1 + \frac{1}{x}\right) \)

These guarantee:
- HiB metrics grow sublinearly (diminishing returns),
- LiB metrics produce higher utility for smaller values.

All utilities satisfy:  
\[
u_{T,j} > 0.
\]

---

## 4. Weights

Each metric receives a weight \( w_T \) such that:

\[
w_M + w_B + w_P + w_C = 1.
\]

Example (deep-learning oriented):

\[
w_M = 0.30,\quad
w_B = 0.25,\quad
w_P = 0.20,\quad
w_C = 0.25.
\]

VRAM and bandwidth dominate because they limit batch size and GEMM throughput.

---

## 5. Aggregator: CES or Geometric Mean

Two mathematically consistent choices exist:

### **A. Weighted Geometric Mean (ρ = 0 case)**  
Stable, interpretable default:

\[
S_j
= \exp\!\left( \sum_{T} w_T \ln u_{T,j} \right).
\]

### **B. CES Aggregator (general ρ)**  
\[
S_j(\rho)
= \left( \sum_{T} w_T \, u_{T,j}^{\rho} \right)^{1/\rho}.
\]

Interpretation via elasticity of substitution:
\[
\rho = 0 \Rightarrow \text{geometric mean}, \quad
\rho > 0 \text{ allows compensation}, \quad
\rho < 0 \text{penalizes weak metrics}.
\]

Typical safe values:
\[
\rho \in \{-1.0,\,-0.5,\,0,\,0.5,\,1.0\}.
\]

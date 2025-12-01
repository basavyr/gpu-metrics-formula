from typing import List
from dataclasses import dataclass
import logging

from math import sqrt, log, pow
import matplotlib.pyplot as plt

from data import GPU_METRICS_DEFAULT_WEIGHTS, GPU_REFERENCE_METRICS

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


@dataclass
class GPUMetric:
    name: str
    vram: float
    bandwidth: float
    tdp: float
    cost_ron: float


class GPUData:
    RTX_5070TI = GPUMetric(name="RTX_5070TI", vram=16,
                           bandwidth=896, tdp=305, cost_ron=4500)
    RX_9070XT = GPUMetric(name="RX_9070XT", vram=16,
                          bandwidth=650, tdp=310, cost_ron=3500)
    RX_7900XTX = GPUMetric(name="RX_7900XTX", vram=24,
                           bandwidth=960, tdp=400, cost_ron=5000)
    RTX_5080 = GPUMetric(name="RTX_5080", vram=16,
                         bandwidth=960, tdp=360, cost_ron=6400)
    RTX_5070 = GPUMetric(name="RTX_5070", vram=12,
                         bandwidth=670, tdp=250, cost_ron=3300)


class GPUMetricTransform:
    @staticmethod
    def f_M_j(gpu_vram: float):
        return log(1.0 + gpu_vram)

    @staticmethod
    def f_B_j(gpu_bandwidth: float):
        return sqrt(gpu_bandwidth)

    @staticmethod
    def f_P_j(gpu_tdp: float):
        return log(1.0 + (1.0 / gpu_tdp))

    @staticmethod
    def f_C_j(gpu_cost: float):
        return log(1.0 + (1.0 / gpu_cost))

    @staticmethod
    def u_M_j(gpu_metrics: GPUMetric):
        f_M_j = GPUMetricTransform.f_M_j(gpu_metrics.vram)
        f_M_ref = GPUMetricTransform.f_M_j(GPU_REFERENCE_METRICS['vram'])
        return f_M_j/f_M_ref

    @staticmethod
    def u_B_j(gpu_metrics: GPUMetric):
        f_B_j = GPUMetricTransform.f_B_j(gpu_metrics.bandwidth)
        f_B_ref = GPUMetricTransform.f_B_j(GPU_REFERENCE_METRICS['bandwidth'])
        return f_B_j/f_B_ref

    @staticmethod
    def u_P_j(gpu_metrics: GPUMetric):
        f_P_j = GPUMetricTransform.f_P_j(gpu_metrics.tdp)
        f_P_ref = GPUMetricTransform.f_P_j(GPU_REFERENCE_METRICS['tdp'])
        return f_P_j/f_P_ref

    @staticmethod
    def u_C_j(gpu_metrics: GPUMetric):
        f_C_j = GPUMetricTransform.f_C_j(gpu_metrics.cost_ron)
        f_C_ref = GPUMetricTransform.f_C_j(GPU_REFERENCE_METRICS['cost_ron'])
        return f_C_j/f_C_ref


def plot_gpu_scores(gpu_list: List[GPUMetric]):
    import numpy as np
    bar_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    names =[]
    scores = []
    for gpu in gpu_list:
        names.append(gpu.name)
        scores.append(compute_gpu_score(gpu))
        bar_colors.append(np.random.choice(bar_colors, 1).item())
    plt.bar(names, scores, color=bar_colors)
    plt.ylabel("Quality Score")
    plt.savefig("gpu-scores.png", dpi=300)
    plt.close()


def compute_gpu_score(gpu: GPUMetric, rho: float = -100, metric_weights: dict = GPU_METRICS_DEFAULT_WEIGHTS) -> float:
    """
    Apply the Weighted Constant Elasticity of Substitution (CES) aggregator, with the formula:
    S_j(\rho) = um_k w_j u_{k, j}^\rho\right)^{1/\rho}
    where
    sum_kw_k = 1
    """
    s_vram = metric_weights["vram"]*pow(GPUMetricTransform.u_M_j(gpu), rho)
    s_bandwidth = metric_weights["bandwidth"] * \
        pow(GPUMetricTransform.u_B_j(gpu), rho)
    s_tdp = metric_weights["tdp"]*pow(GPUMetricTransform.u_P_j(gpu), rho)
    s_cost = metric_weights["cost_ron"]*pow(GPUMetricTransform.u_C_j(gpu), rho)
    s = round(pow(s_vram+s_bandwidth+s_tdp+s_cost, 1.0/rho), 3)

    logger.info(f'GPU: {gpu.name} | Score: {s}')
    return s


def main():
    gpus = [GPUData.RTX_5070TI,
            GPUData.RX_9070XT,
            GPUData.RX_7900XTX,
            GPUData.RTX_5080,
            GPUData.RTX_5070]
    # scores = [compute_gpu_score(gpu) for gpu in gpus]
    plot_gpu_scores(gpus)


if __name__ == "__main__":
    main()

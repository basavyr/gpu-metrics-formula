import argparse


from gpu import compute_gpu_score, GPUMetric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Compute a GPU quality score",
                                     description="A mathematical formalism based on the Constant of Elasticity Substitution")

    parser.add_argument("--name", required=True, type=str,
                        help="GPU series (e.g., RTX_5070TI, RX_9700XT, etc.)")
    parser.add_argument("--vram", required=True, type=float,
                        help="The total VRAM of the GPU (in GB)")
    parser.add_argument("--bandwidth", required=True, type=float,
                        help="The max theoretical memory bandwidth of the GPU (in GBps)")
    parser.add_argument("--tdp", required=True, type=float,
                        help="The total power required by the GPU (in W)")
    parser.add_argument("--cost", required=True, type=float,
                        help="The GPU price (in USD or RON)")

    args = parser.parse_args()

    gpu = GPUMetric(args.name, args.vram, args.bandwidth, args.tdp, args.cost)
    compute_gpu_score(gpu)

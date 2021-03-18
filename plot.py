"""plot.py
Python script for plotting performance experiment results
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# color palette
PAL = "muted"

# Set Matplotlib to use pgfplots as a backend
mpl.use("pgf")

# Matplotlib config for generating LaTeX
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8]{inputenc}\DeclareUnicodeCharacter{2212}{-}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{cmbright}",
            ]
        ),
    }
)

# Read in the results data
df = pd.read_csv(
    "output/random.txt",
    sep=" ",
    names=["kernel", "length", "count", "microseconds"],
)

df["count"] = df["count"] ** 2

df["flops"] = df["length"] ** 2 * df["count"] * 18

df["gflops"] = df["flops"] / df["microseconds"] / 1000

multi_kernels = {
    "softdtw_cuda_naive_multi": "naive",
    "softdtw_cuda_stencil_multi": "stencil",
    "softdtw_cuda_diagonal_multi": "diagonal",
    # "soft_dtw_tiled_multi": "tiled",
}

df_multi = (
    df[df.kernel.isin(multi_kernels) & (df.length == 100)]
    .groupby(["kernel", "length", "count"])[["gflops", "microseconds"]]
    .mean()
    .reset_index()
)

df_multi.kernel = df_multi.kernel.apply(lambda x: multi_kernels.get(x))

plot_multi = sns.lineplot(
    data=df_multi,
    x="count",
    y="gflops",
    style="kernel",
    hue="kernel",
    markers=True,
    dashes=False,
    ci=None,
    palette=PAL,
)

plot_multi.set_xlabel("Pairwise DTW calculations")
plot_multi.set_ylabel("GFLOP/s")
plot_multi.set_ylim(0)

# plt.savefig("fig/plot_multi.png")
plt.savefig("fig/plot_multi.pgf")
plt.clf()

# Naive kernel by Sakoe-Chiba bandwidth
# look up multiplier for flops since Sakoe-Chiba results in fewer FLOPs
bw_pct = {
    100: 1.0,
    80: 0.962,
    60: 0.844,
    40: 0.646,
    20: 0.368,
}

df_naive_bw = (
    df[
        df.kernel.str.startswith("softdtw_cuda_naive_multi")
        & (df.length == 100)
    ]
    .groupby(["kernel", "length", "count"])[["gflops", "microseconds"]]
    .mean()
    .reset_index()
)

df_naive_bw["bandwidth"] = (
    pd.to_numeric(df_naive_bw.kernel.str[-2:], errors="coerce")
    .fillna(100)
    .astype(int)
)

df_naive_bw["bw_pct"] = df_naive_bw["bandwidth"].apply(
    lambda x: bw_pct.get(x, 100)
)

df_naive_bw["gflops"] = df_naive_bw["gflops"] * df_naive_bw["bw_pct"]

plot_naive_bw = sns.lineplot(
    data=df_naive_bw,
    x="count",
    y="gflops",
    style="bandwidth",
    hue="bandwidth",
    markers=True,
    dashes=False,
    ci=None,
    palette=PAL,
)

plot_naive_bw.set_xlabel("Pairwise DTW calculations")
plot_naive_bw.set_ylabel("GFLOP/s")
plot_naive_bw.set_ylim(0)

plt.savefig("fig/plot_naive_multi_bw.pgf")
plt.clf()

# CPU sequential program
df_cpu = pd.read_csv(
    "output/random_cpu.txt",
    sep=" ",
    names=["kernel", "length", "count", "microseconds"],
)

df_cpu["count"] = df_cpu["count"] ** 2

df_cpu["flops"] = df_cpu["length"] ** 2 * df_cpu["count"] * 18

df_cpu["gflops"] = df_cpu["flops"] / df_cpu["microseconds"] / 1000

df_cpu_agg = (
    df_cpu[(df_cpu.kernel == "softdtw") & (df_cpu.length == 100)]
    .groupby(["kernel", "length", "count"])[["gflops", "microseconds"]]
    .mean()
    .reset_index()
)

df_cpu_agg.kernel = "cpu"

plot_cpu = sns.lineplot(
    data=df_cpu_agg,
    x="count",
    y="gflops",
    style="kernel",
    hue="kernel",
    markers=True,
    dashes=False,
    ci=None,
    palette=PAL,
)

plot_cpu.set_xlabel("Pairwise DTW calculations")
plot_cpu.set_ylabel("GFLOP/s")
plot_cpu.set_ylim(0)

# plt.savefig("fig/plot_multi.png")
plt.savefig("fig/plot_cpu.pgf")
plt.clf()

cpu_gpu_kernels = {"softdtw": "CPU", "softdtw_cuda_naive_multi": "CUDA"}

# CPU vs GPU naive
df_cpu_gpu = pd.concat([df, df_cpu])
df_cpu_gpu = (
    df_cpu_gpu[
        df_cpu_gpu.kernel.isin(cpu_gpu_kernels) & (df_cpu_gpu.length == 100)
    ]
    .groupby(["kernel", "length", "count"])[["gflops", "microseconds"]]
    .mean()
    .reset_index()
)

df_cpu_gpu.kernel = df_cpu_gpu.kernel.apply(lambda x: cpu_gpu_kernels.get(x))

plot_cpu_gpu = sns.lineplot(
    data=df_cpu_gpu,
    x="count",
    y="gflops",
    style="kernel",
    hue="kernel",
    markers=True,
    dashes=False,
    ci=None,
    palette=PAL,
)

plot_cpu_gpu.set_xlabel("Pairwise DTW calculations")
plot_cpu_gpu.set_ylabel("GFLOP/s")
plot_cpu_gpu.set_ylim(0)

# plt.savefig("fig/plot_multi.png")
plt.savefig("fig/plot_cpu_gpu.pgf")
plt.clf()

# Euclidean Distance kernel plot
# TODO

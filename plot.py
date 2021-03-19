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

# Profiling data

prof_kernels = {
    "softdtw_naive_kernel_multi": "1 naive",
    "softdtw_stencil": "3 stencil",
    "softdtw_diagonal_kernel_multi": "2 diagonal",
    # "soft_dtw_tiled_multi": "tiled",
}

metrics = {
    "Achieved Occupancy": "Occupancy",
    "Registers Per Thread": "Registers / Thread",
    "L1/TEX Hit Rate": "L1 Cache Hit",
    "L2 Hit Rate": "L2 Cache Hit",
    "SM Busy": "SM Busy",
    "Mem Busy": "Mem Busy",
}

df_prof = pd.read_csv("output/ncu_100_2.csv")

df_prof["Kernel"] = df_prof["Kernel Name"].str.split("(", n=1, expand=True)[0]

df_prof = df_prof[
    (df_prof["Kernel"].isin(prof_kernels))
    & (df_prof["Metric Name"].isin(metrics))
]

df_prof["Kernel"] = df_prof["Kernel"].apply(lambda x: prof_kernels[x])
df_prof["Metric"] = df_prof["Metric Name"].apply(lambda x: metrics[x])

df_prof["Metric Value"] = df_prof["Metric Value"].astype(float)
df_prof_sum = (
    df_prof.groupby(["Kernel", "Metric"])["Metric Value"]
    .mean()
    .round(2)
    .reset_index()
    .pivot_table(index="Kernel", columns=["Metric"])
    .reset_index()
)

df_prof_sum.to_latex("fig/prof_table.tex", index=False)

# ECG perf

df_ecg = pd.read_csv(
    "output/ecg.txt",
    sep=" ",
    names=["kernel", "length", "count", "microseconds"],
)


df_ecg["gflops"] = (
    ((df_ecg["length"] ** 2) * (df_ecg["count"] ** 2) * 18)
    / df_ecg["microseconds"]
    / 1000
)

ecg_kernels = {
    "convert_diagonal_multi": "convert diagonal",
    "softdtw_cuda_diagonal_multi": "diagonal",
    "softdtw_cuda_naive_multi": "naive",
    "softdtw_cuda_naive_multi_bw_20": "naive bandwidth 20",
    "softdtw_cuda_naive_multi_bw_40": "naive bandwidth 40",
    "softdtw_cuda_naive_multi_bw_60": "naive bandwidth 60",
    "softdtw_cuda_naive_multi_bw_80": "naive bandwidth 80",
    "softdtw_cuda_stencil_multi": "stencil",
    "softdtw_cuda_stencil_multi_20": "stencil bandwidth 20",
    "softdtw_cuda_stencil_multi_40": "stencil bandwidth 40",
    "softdtw_cuda_stencil_multi_60": "stencil bandwidth 60",
    "softdtw_cuda_stencil_multi_80": "stencil bandwidth 80",
    "sq_euclid_dist_multi": "squared euclidean distance",
}

df_ecg = (
    df_ecg.groupby(["kernel"])[["microseconds", "gflops"]]
    .mean()
    .round()
    .astype(int)
    .reset_index()
)

df_ecg["kernel"] = df_ecg["kernel"].apply(lambda x: ecg_kernels[x])

df_ecg.to_latex("fig/ecg_kernels.tex", index=False)

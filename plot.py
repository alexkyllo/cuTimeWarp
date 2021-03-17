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

plt.savefig("fig/plot_naive_multi_bw.pgf")
plt.clf()

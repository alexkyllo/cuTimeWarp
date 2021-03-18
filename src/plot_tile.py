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
#mpl.use("pgf")

# Matplotlib config for generating LaTeX


# Read in the results data
df = pd.read_csv(
    "output/random_tiles.csv",
    sep=",",
    names=["kernel", "length", "gflops" ],
)



multi_kernels = {
    "soft_dtw_tiled": "tiled",
    #"softdtw_cuda_stencil_multi": "stencil",
    #"softdtw_cuda_diagonal_multi": "diagonal",
    # "soft_dtw_tiled_multi": "tiled",
}

df_multi = (
    df[df.kernel.isin(multi_kernels)]
    .groupby(["kernel", "length"])[["gflops"]]
    .mean()
    .reset_index()
)

df_multi.kernel = df_multi.kernel.apply(lambda x: multi_kernels.get(x))

plot_multi = sns.lineplot(
    data=df_multi,
    x="length",
    y="gflops",
    style="kernel",
    hue="kernel",
    markers=True,
    dashes=False,
    ci=None,
    palette=PAL,
)

plot_multi.set_xlabel("lenght(millions)")
plot_multi.set_ylabel("GFLOP/s")
plot_multi.set_ylim(0)

# plt.savefig("fig/plot_multi.png")
plt.savefig("fig/plot_tiled.png")
plt.clf()


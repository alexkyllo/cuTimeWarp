"""plot.py
Python script for plotting performance experiment results
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

df_naive = (
    df[(df.kernel == "softdtw_cuda_naive_multi") & (df.length == 100)]
    .groupby(["length", "count"])["microseconds"]
    .mean()
    .reset_index()
)

plot_naive = sns.lineplot(
    data=df_naive,
    x="count",
    y="microseconds",
    markers=True,
    ci=None,
)

plt.savefig("fig/plot_naive.pgf")

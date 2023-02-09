from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from os import path as op


# This script is designed to receive unified errors csv and depict boxplot errors of each network

repo = "/home/sasha/Documents/BachelorsProject/Repo/progress_tracking/detection/corners_nn_c_jcd/errors/"
file_path = repo+"total_errors.csv"
network_name = "C-JCD"
frame = pd.read_csv(file_path, sep=",", index_col=0, header=0)

sets = ["test", "train"]
f, ax = plt.subplots(1, 2, figsize=(10, 5))
for idx_s, s in enumerate(sets):
    best_vals = [10, 10, 10, 10]  # max, 1, 2 ,3 quartiles
    subdf = frame[frame["SET"] == s][["MEAN", "SHAPE"]]
    data = {}
    for size in pd.unique(subdf["SHAPE"]):
        vals = subdf[subdf["SHAPE"]==size]["MEAN"].values
        data[str(size)+"x"+str(size)] = vals
        if max(vals) < best_vals[0]: best_vals[0] = max(vals)
        if np.quantile(vals, .25) < best_vals[1]: best_vals[1] = np.quantile(vals, .25)
        if np.quantile(vals, .5) < best_vals[2]: best_vals[2] = np.quantile(vals, .5)
        if np.quantile(vals, .75) < best_vals[3]: best_vals[3] = np.quantile(vals, .75)
    a = ax[idx_s]
    a.boxplot(data.values())
    a.set_xticklabels(data.keys())
    a.set_xlabel("Image size in pixels")
    a.set_title(f"{s} set")
    a.set_ylim(0, 0.3)
    # a.set_yscale("log")

    # plot minimal references
    a.hlines(min(subdf["MEAN"].values), 0, 5, color="g", ls="--", alpha=0.5,
             label="best values of min, max, 1-3 quantiles")
    for v in best_vals: a.hlines(v, 0, 5, color="g", ls="--", alpha=0.5)
    a.legend()
ax[0].set_ylabel("Distribution of normalised error")
plt.suptitle(f"{network_name} average prediction errors depending on image size")
plt.tight_layout()
plt.savefig(repo+"jcd_c_boxplot.png", dpi=300)


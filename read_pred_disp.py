import numpy as np
import os
from matplotlib import pyplot as plt

config_path = "/home/zcy/tmp/repcv_bb/models/weights_10"
file_path = os.path.join(config_path, "disps_eigen_split.npy")

preds = np.load(file_path)

for p in preds:
    plt.imshow(p)
    plt.show()
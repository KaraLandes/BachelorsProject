import glob
from pathlib import Path
import os
from Repo.detection.corners_nn.network import CornerDetector
from Repo.detection.corners_nn.train_evaluate import TrainCorner
from Repo.detection.corners_nn.dataset import CornerRealBillSet, CornerBillOnBackGroundSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image
import numpy as np
import cv2

bills = sorted(glob.glob(os.path.join("processed_data", "faster_processed_2", "realbills", "unseen", "[r]*.jpg"), recursive=False))
masks = sorted(glob.glob(os.path.join("processed_data", "faster_processed_2", "realbills", "unseen", "[m]*.jpg"), recursive=False))
rotate_neg = ["real_spar_2_007", "real_spar_2_008", "real_spar_2_009", "real_spar_2_010",
              "real_spar_2_011", "real_spar_2_012", "real_spar_2_013", "real_spar_2_014",
              "real_spar_2_036", "real_spar_2_037", "real_spar_2_038", "real_spar_2_039",
              "real_spar_2_040", "real_spar_2_041", "real_spar_2_043", "real_spar_2_044",
              "real_spar_2_045", "real_billa_007", "real_spar_3_038", "real_spar_3_039",
              "real_spar_3_040", "real_spar_3_041", "real_spar_3_042", "real_spar_4_002",
              "real_spar_4_003", "real_spar_4_004"]
for b, m in tqdm(zip(bills, masks)):
    bill = Image.open(b).convert('RGB')
    mask = Image.open(m).convert('L')
    dims = (500, 500)
    bill.thumbnail(size=dims, resample=Image.ANTIALIAS)
    mask.thumbnail(size=dims, resample=Image.ANTIALIAS)


    if mask.width != bill.width and np.isin(b.split("/")[-1][:-4], rotate_neg):
        mask = mask.rotate(-90, expand=True)
    elif mask.width != bill.width:
        mask = mask.rotate(90, expand=True)
    elif np.isin(b.split("/")[-1][:-4], rotate_neg):
        mask = mask.rotate(180, expand=True)

    bill = np.array(bill)
    mask = np.array(mask)

    corners = cv2.goodFeaturesToTrack(mask, 4, 0.01, 5)
    corners = [[c[0][0], c[0][1]] for c in corners]
    f, ax = plt.subplots(3, 1)
    ax[0].imshow(bill[:,:,::-1])
    mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    ax[1].imshow(bill * mask / 255)
    ax[2].imshow(mask)
    for c in corners: ax[2].scatter(c[0], c[1], c="r", s=50)
    f.suptitle(f"{b[25:-4]}")

    f.savefig(os.path.join("real_bills_results", "chekup", "unseen", f"{b.split('/')[-1][:-4]}.png"), dpi=200)
    plt.clf()

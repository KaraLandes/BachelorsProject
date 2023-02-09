import glob
import os
from os import path as op

import cv2
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image, ImageFilter
from detection.dummy_cnn.dataset import BaseBillOnBackGroundSet
from tqdm import tqdm
from sewar.full_ref import sam as sim_measure
from itertools import combinations, product
import time
from matplotlib import pyplot as plt
from multiprocessing import Pool
import pandas as pd


repo = Path(os.getcwd())

im_dir_gen = os.path.join(repo, "processed_data", "genbills")
im_dir_real = os.path.join(repo, "processed_data", "realbills")
im_dir_unseen = os.path.join(repo, "processed_data", "realbills", "unseen")


def resize(list_of_images, size):
    outp = []
    for im in tqdm(list_of_images):
        copy = im.copy()
        copy.thumbnail(size=(size, size), resample=Image.ANTIALIAS)
        if copy.width > copy.height:
            copy = copy.rotate(90, fillcolor=(0,), expand=True)
        outp.append(copy)
    return outp

def combs_self(list_of_images):
    return np.array(list(combinations(range(len(list_of_images)), r=2))).astype(int)

def combs_between(list_of_images1, list_of_images2):
    return np.array(list(product(range(len(list_of_images1)), range(len(list_of_images2))))).astype(int)

def simil(pair): # subfunction to put in parallel loop
    im_1, im_2 = pair
    m = ""
    if im_1.width != im_2.width or im_1.height != im_2.height:
        m = f"crop happened\n im1 dims = {im_1.width},{im_1.height},\n im2 dims = {im_2.width},{im_2.height}"
        min_w = min(im_1.width, im_2.width)
        min_h = min(im_1.height, im_2.height)
        im_1 = im_1.crop((1, 1, min_w-1, min_h-1))
        im_2 = im_2.crop((1, 1, min_w-1, min_h-1))
        m+= f"\n crop dims = 1to{min_w-1}, 1to{min_h-1}"
        m+= f"\n final dims = {im_1.width},{im_1.height}"

    try:
        score = sim_measure(np.array(im_1), np.array(im_2))
    except Exception as e:
        score = 0.5
        print(e)
        print(m)
    return score
def similarity(list_of_images1, list_of_images2, combs):
    similarity_score = 0
    list_of_images1 = [list_of_images1[idx] for idx in combs[:,0]]
    list_of_images2 = [list_of_images2[idx] for idx in combs[:,1]]
    with Pool(12) as pool:
        for score in tqdm(pool.imap(simil, zip(list_of_images1, list_of_images2)), total=len(list_of_images1)):
            similarity_score += score
        pool.close()
    similarity_score /= len(combs)
    return similarity_score

def edgin(image): #task function to put in Pool loop
    corners = cv2.goodFeaturesToTrack(np.array(image.convert("L")), int(1e+6), 1e-6, 1e-6)
    return len(corners)
def edginess(list_of_images):
    score = 0
    with Pool(12) as pool:
        for corners in tqdm(pool.imap(edgin, list_of_images), total=len(list_of_images)):
            score += corners
    score /= len(list_of_images)
    return score


# This script is meant do discover which size for training corner_cnn is the best
generated_images = BaseBillOnBackGroundSet(image_dir=im_dir_gen)
loader = DataLoader(dataset=generated_images,
                    batch_size=1,
                    num_workers=12,
                    shuffle=True)
temp = []
for im, _ in tqdm(loader, total=200):
    im = im[0].numpy()
    where_0 = np.sum(im, axis=2) > 0
    for row, element in enumerate(where_0):
        if np.all(element == 0): break

    for col, element in enumerate(where_0.T):
        if np.all(element == 0): break

    im = im[:row, :col, :]
    try:
        temp.append(Image.fromarray(im))
    except:
        print("Error occured")

    if len(temp) == 200: break

generated_images = temp

real_images = glob.glob(op.join(im_dir_real, "*.jpg"), recursive=False)
real_images = [Image.open(file) for file in real_images if not "mask" in file]#[:8]

test_images = glob.glob(op.join(im_dir_unseen, "*.jpg"), recursive=False)
test_images = [Image.open(file) for file in test_images if not "mask" in file]#[:8]

sizes = np.geomspace(1000, 10, 100).astype(int)
scores = {'sim_gen': [],
          'sim_real': [],
          'sim_test': [],
          'sim_gen_vs_real': [],
          'sim_gen_vs_test': [],
          'sim_test_vs_real': [],
          "edg_gen": [],
          "edg_real": [],
          "edg_test": []}
print("#" * 100)
print()
for size in sizes:
    images_of_size = {"gen": [], "real": [], "test": []}

    print(f"Resizing {size}")
    images_of_size['gen'] = resize(generated_images, size)
    images_of_size['real'] = resize(real_images, size)
    images_of_size['test'] = resize(test_images, size)
    time.sleep(2)

    print(f"\nCollect similarity inside every set {size}")
    for k in images_of_size.keys():
        sim = similarity(list_of_images1=images_of_size[k],
                         list_of_images2=images_of_size[k],
                         combs=combs_self(images_of_size[k]))
        scores[f'sim_{k}'].append(sim)
    time.sleep(2)

    print(f"\nCollect similarity inbetween sets {size}")
    for k_pair in [("gen", "real"), ("gen", "test"), ("test", "real")]:
        sim = similarity(list_of_images1=images_of_size[k_pair[0]],
                         list_of_images2=images_of_size[k_pair[1]],
                         combs=combs_between(list_of_images1=images_of_size[k_pair[0]],
                                             list_of_images2=images_of_size[k_pair[1]]))

        scores[f'sim_{k_pair[0]}_vs_{k_pair[1]}'].append(sim)
    time.sleep(2)

    print(f"\nCollect edginess of every set {size}")
    for k in images_of_size.keys():
        edg = edginess(list_of_images=images_of_size[k])
        scores[f'edg_{k}'].append(edg)
    time.sleep(2)

    # plotting current results
    num_el = len(scores["sim_gen"])
    f, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    ax[0].set_title("Dissimilarity of images within each set")
    ax[0].set_xlabel("Size of image")
    ax[0].plot(sizes[:num_el][::-1], scores["sim_gen"][::-1], label="generated images", c="red")
    ax[0].plot(sizes[:num_el][::-1], scores["sim_real"][::-1], label="real images", c="blue")
    ax[0].plot(sizes[:num_el][::-1], scores["sim_test"][::-1], label="test images", c="blue", ls=":")

    ax[1].set_title("Dissimilarity of images between sets")
    ax[1].set_xlabel("Size of image")
    ax[1].plot(sizes[:num_el][::-1], scores["sim_gen_vs_real"][::-1], label="generated vs real images", c="blue")
    ax[1].plot(sizes[:num_el][::-1], scores["sim_gen_vs_test"][::-1], label="generated vs test images", c="blue", ls=":")
    ax[1].plot(sizes[:num_el][::-1], scores["sim_test_vs_real"][::-1], label="real vs test images", c="green")

    ax[2].set_title("Number of corners detected of images within each set")
    ax[2].set_xlabel("Size of image")
    ax[2].plot(sizes[:num_el][::-1], scores["edg_gen"][::-1], label="generated images", c="red")
    ax[2].plot(sizes[:num_el][::-1], scores["edg_real"][::-1], label="real images", c="blue")
    ax[2].plot(sizes[:num_el][::-1], scores["edg_test"][::-1], label="test images", c="blue", ls=":")
    ax[2].set_yscale('log')

    for a in ax:
        a.legend()
        a.grid(axis="x", which="both")
        a.invert_xaxis()
        a.set_xscale('log')

    plt.tight_layout()
    plt.savefig("/home/sasha/Documents/BachelorsProject/Repo/real_bills_results/comp_sizes/0_stats.png", dpi=150)
    plt.close("all")

    # save examples of images
    images_of_size['gen'][0].save(f"/home/sasha/Documents/BachelorsProject/Repo/real_bills_results/comp_sizes/generated_{size}.png")
    images_of_size['real'][0].save(f"/home/sasha/Documents/BachelorsProject/Repo/real_bills_results/comp_sizes/real_{size}.png")
    images_of_size['test'][0].save(f"/home/sasha/Documents/BachelorsProject/Repo/real_bills_results/comp_sizes/test_{size}.png")

    #save scores
    frame = pd.DataFrame(scores)
    frame.set_index(sizes[:num_el], inplace=True)
    frame.to_csv(f"/home/sasha/Documents/BachelorsProject/Repo/real_bills_results/comp_sizes/0_scores.csv", sep=";")
    print("#" * 100)


#############################################################################################################
# STEP 0
# Imports and custom functions
#############################################################################################################
from PIL import Image
import cv2

import numpy as np
import os
import math
import re
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from pandas.plotting import table
import pandas as pd

from detection.dummy_cnn.dataset import BaseRealBillSet

import pytesseract
from pytesseract import Output

# pytesseract.pytesseract.tesseract_cmd = os.path.abspath("/usr/bin/tesseract")


def form_purchase(text_list):
    purchases = {}
    purchases["economy"] = 0

    price_pattern = re.compile('[0-9]+\.[0-9]+ [A-Z]+')
    bad_price_pattern = re.compile('[0-9][0-9] [A-Z]')
    discount_pattern = re.compile('-[0-9]+\.[0-9]+')

    stop_words = ["ZAHLUNG", "SUMME", "BEZAHLT"]
    prev_price_id = 0
    for t, text in enumerate(text_list):
        price_match = price_pattern.match(text)
        bad_match = bad_price_pattern.match(text)
        if not price_match is None or not bad_match is None:
            if not price_match is None:
                price = float(text.split(" ")[0])
            elif not bad_match is None:
                price = float(text.split(" ")[0])/100
            if prev_price_id == 0:
                product = " ".join(text_list[t-2])
            else:
                product = " ".join(text_list[prev_price_id+1:t])

            try:
                discount_match = discount_pattern.match(text_list[t + 4])
                if not discount_match is None:
                    discount = float(text_list[t + 4])
                    price += discount
                    purchases["economy"] += (discount * (-1))
            except: pass

            purchases[product] = price
            prev_price_id = t
            continue

        for word in stop_words:
            if word in text:
                return purchases

    return purchases

def detect_angle(arr):
    img_edges = cv2.Canny(arr, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []

    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    return np.mean(angles)


#############################################################################################################
# STEP 1
# Prepare
#############################################################################################################

repo = Path(os.getcwd())

im_dir_gen = os.path.join(repo, "processed_data", "genbills")
im_dir_real = os.path.join(repo, "processed_data", "realbills")
im_dir_unseen = os.path.join(repo, "processed_data", "realbills", "unseen")

im_dir_real_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills")
im_dir_unseen_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills", "unseen")

UNSEEN = 0
if UNSEEN:
    im_dir = im_dir_unseen
else:
    im_dir = im_dir_real


dataset = BaseRealBillSet(image_dir=im_dir, output_shape=(4000, 4000), coefficient=1)

#############################################################################################################
# STEP 2
# Mask original Images, extract bill only
# Preprocess image according to tesseract tutorials
# Define angle of text rotation
#############################################################################################################

for idx in tqdm(range(dataset.__len__())):
    item = dataset.__getitem__(idx)
    image, mask = item[0], item[1]["masks"]

    image = np.array(Image.fromarray(image).convert("L")).astype(np.uint8)

    mask_in = np.zeros(mask.shape)
    mask_in[mask == 0] = 1

    masked_bill = np.zeros(image.shape)
    masked_bill[mask_in == 1] = image[mask_in == 1]
    masked_bill = masked_bill.astype(np.uint8)

    masked_bill = cv2.adaptiveThreshold(src=masked_bill,
                                        maxValue=255,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=211,
                                        C=25)

    masked_bill = Image.fromarray(masked_bill)

    try: angle = detect_angle(np.array(mask))
    except: angle = detect_angle(np.array(masked_bill))
    descewed_bill = masked_bill.rotate(angle=-angle, expand=True)

    text_for_prices = pytesseract.image_to_string(descewed_bill, output_type=Output.DICT, lang="deu",
                                                  config="--psm 11")

    text_for_prices = text_for_prices['text'].split("\n")
    purchase = form_purchase(text_for_prices)

    if len(purchase.keys()) == 1:
        descewed_bill = descewed_bill.rotate(angle=180, expand=True)
        text_for_prices = pytesseract.image_to_string(descewed_bill, output_type=Output.DICT, lang="deu",
                                                      config="--psm 11")

        text_for_prices = text_for_prices['text'].split("\n")
        purchase = form_purchase(text_for_prices)

    #  checkup stage
    frame = pd.DataFrame.from_dict(purchase, orient="index", columns=["Sum"])
    f, a = plt.subplots(1, 2, figsize=(18, 9))
    a[0].imshow(np.array(descewed_bill), cmap="binary_r")
    a[1].table(cellText=frame.values,
               colWidths=[0.25]*len(frame.columns),
               rowLabels=frame.index,
               colLabels=frame.columns,
               cellLoc='center',
               rowLoc='center',
               loc='center right')
    for ax in a:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis('off')
    f.savefig(os.path.join(repo, "real_bills_results", "tesseract", f"{UNSEEN}_{idx:03d}.png"),
              dpi=600)
    plt.tight_layout()
    plt.close()
    b = 1

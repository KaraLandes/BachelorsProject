import os
import glob
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
from data_collection_and_preprocessing.scrapper import SparScrapper
from data_collection_and_preprocessing.generateimages import SparImageGenerator, PennyImageGenerator, BillaImageGenerator
from image_segmentation.data import BillSet, collate_fn
from image_segmentation.networks import BaseNet
from image_segmentation.train_evaluate import run_epoch, depict


# This file is created to prepare and preprocess all  data
# HYPERPARAMETERS, WHICH CODE TO RUN
SCRAP = False  # do scrapping from websites
GENIM = False  # generate bills images
SEGMENT = True  # do image segmentation


repo = Path(os.getcwd())
########################################################################################################################
# Stage 1
# Within this stage I scrap information about products from online shops.
# Everything is saved in csv file.
########################################################################################################################
def shorten(title):
    """
    Takes full title of product and creates a shortened version. Keeping 3 words, 9 letters max
    """
    title = title.split(" ")
    title = [word[:10].upper()+"." if len(word)>9 else word.upper() for word in title[:3]]
    title = " ".join(title)
    return title

if SCRAP:
    path = os.path.join(repo, "processed_data", "scrapped", "spar.csv")
    spar = SparScrapper(start_url="https://www.interspar.at/shop/lebensmittel/",
                        saving_file=os.path.abspath(path))

    # spar.scrap(cat_range=range(0,16), starting_page=1)

    spar_products = pd.read_csv(path, sep=';', index_col=0)
    spar_products.reset_index(inplace=True, drop=True)
    spar_products.drop_duplicates(inplace=True)
    spar_products['short_title']=spar_products['full_title'].apply(shorten)
    spar_products.to_csv(os.path.join(repo, "processed_data", "scrapped","spar_no_dubs.csv"),sep=';')

########################################################################################################################
# Stage 2
# Having primary data from website I am generating images which are very similar to real
# supermarket bills.
########################################################################################################################
if GENIM:
    #SPAR BILLS
    numer_of_bills = int(2000/4)
    savedir = os.path.join(repo, "processed_data", "genbills", "spar")
    data = os.path.join(repo, "processed_data", "scrapped", "spar_no_dubs.csv")
    font_dir = os.path.join(repo, "materials_for_preprocessing", "fonts", "MerchantCopy")
    font = "Merchant Copy"

    for clazz, l in zip([1,2,3,4],['spar.png','spar_express.jpg','eurospar.png','interspar.png']):
        print(l,"="*50)
        logo = os.path.join(repo, "materials_for_preprocessing", "logos", l)
        im_gen_MC = SparImageGenerator(saving_path=savedir, data_path=data, logo_path=logo,
                                       font_path=font_dir, font_name=font, s_name='spar')
        pool = Pool(10)
        f = partial(im_gen_MC.generate_bills,clazz=clazz)  # my function is with 2 args, a way to initialise it
        for _ in tqdm(pool.imap_unordered(f, range(numer_of_bills)), total=numer_of_bills): pass
        pool.close()
        pool.join()

    #PENNY BILLS
    numer_of_bills = 2000
    savedir = os.path.join(repo, "processed_data", "genbills", "penny")

    for clazz, l in zip([5],['penny.png']):
        print(l, "=" * 50)
        logo = os.path.join(repo, "materials_for_preprocessing", "logos", l)
        im_gen_MC = PennyImageGenerator(saving_path=savedir, data_path=data, logo_path=logo,
                                        font_path=font_dir, font_name=font, s_name='penny')
        pool = Pool(10)
        f = partial(im_gen_MC.generate_bills, clazz=clazz)  # my function is with 2 args, a way to initialise it
        for _ in tqdm(pool.imap_unordered(f, range(numer_of_bills)), total=numer_of_bills): pass
        pool.close()
        pool.join()

    #BILLA BILLS
    numer_of_bills = int(2000 / 4)
    savedir = os.path.join(repo, "processed_data", "genbills", "billa")

    for clazz, l in zip([6,7,8,9], ['billa.jpg','billaplus.jpg','billaplus2.png','billaplus3.jpg']):
        print(l, "=" * 50)
        logo = os.path.join(repo, "materials_for_preprocessing", "logos", l)
        im_gen_MC = BillaImageGenerator(saving_path=savedir, data_path=data, logo_path=logo,
                                        font_path=font_dir, font_name=font, s_name='billa')
        pool = Pool(10)
        f = partial(im_gen_MC.generate_bills, clazz=clazz)  # my function is with 2 args, a way to initialise it
        for _ in tqdm(pool.imap_unordered(f, range(numer_of_bills)), total=numer_of_bills): pass
        pool.close()
        pool.join()

########################################################################################################################
# Stage 3
# In this stage I use generated bills to train a CNN which separated header
# with supermarket logo and purchased goods from the whole image.
########################################################################################################################
if SEGMENT:
    imdir = os.path.join(repo, "processed_data", "genbills")
    images = sorted(glob.glob(os.path.join(imdir, "**", "*.jpg"), recursive=True))
    np.random.seed(0)
    indices = np.random.permutation(len(images))

    test_share, valid_share = 0.15, 0.15#0.99, 0.005, #
    test_share, valid_share = int(test_share*len(images)), int(valid_share*len(images))
    train_share = len(images) - test_share - valid_share

    train_ids = indices[:train_share]
    test_ids = indices[train_share:(train_share+test_share)]
    valid_ids = indices[(train_share+test_share):]

    train_set = BillSet(imdir, train_ids, coefficient=1.5)
    test_set = BillSet(imdir, test_ids, seed=0)
    valid_set = BillSet(imdir, valid_ids, seed=1)

    batch_size = 1
    workers = 10
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              collate_fn=collate_fn,
                              num_workers=workers,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=batch_size,
                              collate_fn=collate_fn,
                              num_workers=workers,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             collate_fn=collate_fn,
                             num_workers=workers,
                             shuffle=False)

    writer = SummaryWriter(log_dir=os.path.join(repo, "progress_tracking", "tensorboard"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_base = True
    # Baseline Network
    if train_base:
        basenet = BaseNet(n_hidden_layers=7)
        basenet.to(device)
        train_losses, valid_losses = [], []
        for epoch in range(1, 10+1):
            print(f"\nEpoch {epoch}:")
            time.sleep(1)
            train_results = run_epoch(loader=train_loader, network=basenet, writer=writer,
                                      optimizer=Adam, learningrate=1e-3, weight_decay=1e-5, device=device)
            valid_results = run_epoch(loader=valid_loader, network=basenet, writer=writer,
                                      optimizer=Adam, learningrate=1e-3, weight_decay=1e-5,
                                      optimize=False, device=device)
            mean_tr_loss, mean_val_loss = np.mean(train_results), np.mean(valid_results)
            train_losses.append(mean_tr_loss)
            valid_losses.append(mean_val_loss)
            torch.save(basenet.state_dict(),
                       os.path.join(repo, "progress_tracking", "image_segmentation", "models", f"basenet_epoch{epoch}.pt"))
            writer.add_scalars("BaseNet_Loss",
                               {
                                   "Training": mean_tr_loss,
                                   "Validation": mean_val_loss
                               },
                               global_step=epoch)
            print("Train Loss\t\t", "{:.2f}".format(mean_tr_loss))
            print("Validation Loss\t", "{:.2f}".format(mean_val_loss))
            depict(loader=train_loader, network=basenet, name_convention=f"train_epoch{epoch}_", num=5, writer=None,
                   path=os.path.join(repo, "progress_tracking", "image_segmentation", "visualization"), device=device)
            depict(loader=train_loader, network=basenet, name_convention=f"valid_epoch{epoch}_", num=5, writer=writer,
                   path=os.path.join(repo, "progress_tracking", "image_segmentation", "visualization"), device=device)
            writer.flush()
        np.save(os.path.join(repo, "progress_tracking", "image_segmentation", "models", f"basenet_trainlosses.npy"),
                train_losses)
        np.save(os.path.join(repo, "progress_tracking", "image_segmentation", "models", f"basenet_validlosses.npy"),
                valid_losses)

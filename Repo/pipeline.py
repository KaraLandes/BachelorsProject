import os

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from torch.optim import Adam, SGD
from data_collection_and_preprocessing.scrapper import SparScrapper
from data_collection_and_preprocessing.generateimages import SparImageGenerator, PennyImageGenerator, BillaImageGenerator
from image_segmentation.data import BillSet, BoxedBillSet
from image_segmentation.networks import BaseNet, MRCNN
from image_segmentation.train_evaluate import TrainBase, TrainMask


# This file is created to prepare and preprocess all  data
# HYPERPARAMETERS, WHICH CODE TO RUN
SCRAP = False  # do scrapping from websites
GENIM = False  # generate bills images
DETECT = True  # detect bills on images
SEGMENT = False  # do image segmentation


repo = Path(os.getcwd())
# repo = repo.parent.absolute()
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
# In this stage I use generated bills to train a CNN which returns cleared bill image only
########################################################################################################################
if DETECT:
    imdir = os.path.join(repo, "processed_data", "genbills")

    # Baseline Network (just CNN)
    basenet_de = BaseNet(n_hidden_layers=2, n_kernels=3)

    train_base = TrainBase(im_dir=imdir, network=basenet)
    train_base.set_datasets(valid_share=.15, test_share=.15, dataset_type=BillSet, coefficient=2)
    train_base.set_writer(
        log_dir=os.path.join(repo, "progress_tracking", "image_segmentation", "basenet", "tensorboard"))
    train_base.set_loaders(batch_size=2)
    train_base.set_device()
    save_model_path = os.path.join(repo, "progress_tracking", "image_segmentation", "basenet", "models", "basenet_")
    save_images_path = os.path.join(repo, "progress_tracking", "image_segmentation", 'basenet', "visualization")
    train_base.train(optimiser=Adam(basenet.parameters(), lr=1e-1, weight_decay=1e-5),
                     save_model_path=save_model_path, epochs=50,
                     save_images_path=save_images_path)

########################################################################################################################
# Stage 4
# In this stage I use generated bills to train a CNN which separated header
# with supermarket logo and purchased goods from the whole image.
########################################################################################################################
if SEGMENT:
    imdir = os.path.join(repo, "processed_data", "genbills")

    # Baseline Network (dummy CNN)
    basenet = BaseNet(n_hidden_layers=2, n_kernels=3)

    train_base = TrainBase(im_dir=imdir, network=basenet)
    train_base.set_datasets(valid_share=.15, test_share=.15, dataset_type=BillSet, coefficient=2)
    train_base.set_writer(log_dir=os.path.join(repo, "progress_tracking", "image_segmentation", "basenet", "tensorboard"))
    train_base.set_loaders(batch_size=2)
    train_base.set_device()
    save_model_path = os.path.join(repo, "progress_tracking", "image_segmentation", "basenet", "models",  "basenet_")
    save_images_path = os.path.join(repo, "progress_tracking", "image_segmentation", 'basenet', "visualization")
    train_base.train(optimiser=Adam(basenet.parameters(), lr=1e-1, weight_decay=1e-5),
                     save_model_path=save_model_path, epochs=50,
                     save_images_path=save_images_path)

    # Mask Regioned Network (Mask R-CNN)
    mrcnnet = MRCNN(num_classes=3).get_model() #3 classes - background, logo, content
    parameters = [p for p in mrcnnet.parameters() if p.requires_grad]

    train_mrcnn = TrainMask(im_dir=imdir, network=mrcnnet)
    train_mrcnn.set_datasets(valid_share=0.15, test_share=0.15, dataset_type=BoxedBillSet, coefficient=2)
    train_mrcnn.set_writer(log_dir=os.path.join(repo, "progress_tracking", "image_segmentation", "maskrcnn", "tensorboard"))
    train_mrcnn.set_loaders(workers=1)
    train_mrcnn.set_device()
    save_model_path = os.path.join(repo, "progress_tracking", "image_segmentation", "maskrcnn", "models", "maskrcnn_")
    save_images_path = os.path.join(repo, "progress_tracking", "image_segmentation", "maskrcnn", "visualization")
    train_mrcnn.train(optimiser=Adam(parameters, lr=1e-3, weight_decay=1e-5),
                      save_model_path=save_model_path,
                      save_images_path=save_images_path)
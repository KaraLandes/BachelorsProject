import os

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool

from data_collection_and_preprocessing.scrapper import SparScrapper
from data_collection_and_preprocessing.generateimages import SparImageGenerator, PennyImageGenerator, BillaImageGenerator


# This file is created to prepare and preprocess all  data
# HYPERPARAMETERS, WHICH CODE TO RUN
SCRAP = False  # do scrapping from websites
GENBILLS = True  # generate bills images


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
    spar_products.to_csv(os.path.join(repo, "processed_data", "scrapped", "spar_no_dubs.csv"), sep=';')

########################################################################################################################
# Stage 2
# Having primary data from website I am generating images which are very similar to real
# supermarket bills.
########################################################################################################################
if GENBILLS:
    num = 500
    #SPAR BILLS
    numer_of_bills = int(num/4)
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
    numer_of_bills = num
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
    numer_of_bills = int(num / 4)
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

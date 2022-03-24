import os
import pandas as pd
from pathlib import Path
from scrapper import SparScrapper
from generateimages import SparImageGenerator

# This file is created to prepare and preprocess all  data
# HYPERPARAMETERS, WHICH CODE TO RUN
SCRAP = True #do scrapping from websites
GENIM = True #generate bills images


current_dir = Path(os.getcwd())
repo = current_dir.parent.absolute()
########################################################################################################################
# Stage 1
# Within this stage I scrap information about products from online shops.
# Everything is saved in csv file.
########################################################################################################################
""" Takes full title of product and creates a shortened version. Keeping 3 words, 9 letters max"""
def shorten(title):
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
    numer_of_bills = 1
    savedir = os.path.join(repo, "processed_data", "genbills")
    logo = os.path.join(repo, "materials_for_preprocessing", "logos", "sparlogo.png")
    data = os.path.join(repo, "processed_data", "scrapped", "spar_no_dubs.csv")
    font_dir = os.path.join(repo, "materials_for_preprocessing", "fonts", "MerchantCopy")
    font = "Merchant Copy"
    im_gen_MC = SparImageGenerator(saving_path=savedir, data_path=data, logo_path=logo,
                                   font_path=font_dir, font_name=font)

    im_gen_MC.generate_bills(n=numer_of_bills)

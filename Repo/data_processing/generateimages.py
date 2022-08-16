import os
import string
import numpy as np
import pandas as pd
from PIL import Image
from numpy import random as rnd
from matplotlib import font_manager
from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageGenerator():
    def __init__(self, saving_path: str, data_path: str, logo_path: str, font_path: str, font_name: str, s_name: str):
        self.directory = saving_path
        self.datadf = pd.read_csv(data_path, sep=';', index_col=0)
        self.logo = logo_path
        self.font_path = font_path
        self.font_name = font_name
        self.s_name = s_name

    def generate_bills(self, n: int, clazz: int) -> None:
        """Method which allows to generate an image of supermarket bill
        Image is saved in self.directory, alongside with dataframe describing its content
        @:param n Integer describing unique number for naming
        @:param clazz Integer used to distinguish between branches of the same market
        @:param name String for unified naming"""

        sample = self._random_basket(seed=(clazz * (10 ** 5) + n))  # select some random goods and quantities
        figure, mask = self._generate_image(sample=sample, seed=(clazz * (10 ** 5) + n))  # generate image and data

        # save temporarily, open with pillow, rotate
        figure.savefig(os.path.join(self.directory, f"temp{n}.jpg"))
        figure = Image.open(os.path.join(self.directory, f"temp{n}.jpg")).convert('L')
        figure = figure.rotate(180)

        # save finally
        figure.save(os.path.join(self.directory, f"{self.s_name}_{str(clazz)}{str(n).zfill(4)}.jpg"))
        sample.to_csv(os.path.join(os.path.join(self.directory, f"{self.s_name}_{str(clazz)}{str(n).zfill(4)}.csv")),
                      sep=';')
        np.save(os.path.join(os.path.join(self.directory, f"{self.s_name}_{str(clazz)}{str(n).zfill(4)}.npy")), mask)

        # clean up
        os.remove(os.path.join(self.directory, f"temp{n}.jpg"))
        del figure
        del sample

    def _random_basket(self, seed: int) -> pd.DataFrame:
        """Hidden method for inside use only.
        Generates a random basket of goods
        @:return Products pd.DataFrame with such columns: 'full_title', 'title', 'specification', 'description',
                                                         'price', 'cat1', 'cat2', 'cat3', 'quantity', 'discount',
                                                         'fullvalue', 'finalvalue'"""

        rnd.seed(seed)
        # STEP1 -- generate random number of unique items, sample random items
        threshold = 0.85
        if rnd.normal() > threshold:
            num_items = rnd.randint(15, 40)
        else:
            num_items = rnd.randint(1, 15)
        sample_ids = rnd.choice(a=self.datadf.index, size=num_items, replace=False)
        sample = self.datadf.loc[sample_ids].copy(deep=True)

        # STEP 2 -- for each product in random basket generate quantity, discount and value

        # small functions for fast columns initialisation
        def quantity(x):
            threshold = rnd.normal(0, 1)
            if threshold < 0.6:
                return 1
            if threshold >= 0.6 and threshold < 0.85:
                return rnd.randint(2, 4)
            else:
                return rnd.randint(4, 20)

        def discount(x):
            probability = rnd.rand()
            if probability < 0.95:
                return 0
            else:
                return rnd.choice(a=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], size=1)[0] / 100

        # columns introduction
        sample['quantity'] = None
        sample['quantity'] = sample['quantity'].apply(quantity)
        sample['discount'] = None
        sample['discount'] = sample['discount'].apply(discount)

        sample['fullvalue'] = sample['price'] * sample['quantity']
        sample['finalvalue'] = sample['fullvalue'] - (sample['fullvalue'] * sample['discount'])

        return sample

    def _random_word(self, length: tuple, seed: int, capital=True, alluper=False, alllower=False) -> str:
        """A hidden method to generate word-like strings
       @:param length A tuple of integers controlling th length of word
       @:param capital Boolean, defining if the first letter is uppercase
       @:return String"""

        rnd.seed(seed)
        lowercase = list(string.ascii_lowercase)
        uppercase = list(string.ascii_uppercase)
        s = ''
        if capital:
            s += rnd.choice(uppercase, 1)[0]
            s += ''.join(rnd.choice(lowercase, rnd.randint(length[0], length[1])))
        elif alluper:
            s += ''.join(rnd.choice(uppercase, rnd.randint(length[0], length[1])))
        elif alllower:
            s += ''.join(rnd.choice(lowercase, rnd.randint(length[0], length[1])))
        return s

    def _create_mask(self, yo1: float, yo2: float, yo3: float, xol:float, logo_w: float, logo_h: float,
                     width: int, height: int, cm2inch: float, DPI: int) -> np.array:
        """
        This method creates a mask
        :param yo1: y offset of logo
        :param yo2: y offset of text2
        :param yo2: y offset of text3
        :param xol: x offset of logo
        :param logo_w: width of logo in cm
        :param logo_h: height of logo in cm
        :param width: width of image
        :param height: height of image
        :return: mask with 0 everywhere, 100 on logo, 200 on contents
        """

        height = int(height * cm2inch * DPI)
        width = int(width * cm2inch * DPI)
        mask = np.zeros(shape=(height, width))

        # logo mask
        logo_start_y = np.round(yo1, 0).astype(int)
        logo_end_y = np.round(logo_start_y + logo_h * cm2inch * DPI, 0).astype(int)
        logo_start_x = np.round(xol, 0).astype(int)
        logo_end_x = np.round(logo_start_x + logo_w * cm2inch * DPI, 0).astype(int)
        mask[logo_start_y:logo_end_y, logo_start_x:logo_end_x] = 100

        # content mask
        content_start_y = np.round(yo2, 0).astype(int)
        content_end_y = np.round(yo3 - 0.255 * cm2inch * DPI, 0).astype(int)
        conten_start_x = np.round(0.5 * cm2inch * DPI, 0).astype(int)
        mask[content_start_y:content_end_y, conten_start_x:-conten_start_x] = 200

        mask = mask.astype(np.uint8)
        return mask

    def _generate_image(self, sample, seed: int):
        """This Method is rewritten for every supermarket
        @raise ValueError if called from this class"""

        raise ValueError("To use method please create an instance of specific supermarket.")


class SparImageGenerator(ImageGenerator):
    pass

    def _generate_image(self, sample: pd.DataFrame, seed: int) -> (plt.Figure, np.array):
        """Hidden method for inside use only.
        Generates an image of bill
        @:param sample: pandas DataFrame which has all needed data inside
        @:return tuple: matplotlib figure object, image of bill and np.array of needed areas"""

        # 0 -- preventive cleanup
        try:
            plt.close('all')
            plt.cla()
            plt.clf()
        except:
            pass
        rnd.seed(seed)

        # 1 -- generate fake header
        street = self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 30)) + "." + str(rnd.randint(1, 100))
        city = str(rnd.randint(1000, 9999)) + " " + self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 15))
        tel = "Tel.: 0" + str(rnd.randint(100, 999)) + " " + str(rnd.randint(100000, 999999))
        date = f"Ihr Einkauf am {str(rnd.randint(1, 32)).zfill(2)}.{str(rnd.randint(1, 13)).zfill(2)}.{rnd.randint(2021, 3000)} " \
               f"um {str(rnd.randint(0, 24)).zfill(2)}:{str(rnd.randint(1, 60)).zfill(2)} Uhr"
        separator_type1 = "-" * 45

        # 2 -- generate basket
        nl = "\n"
        summ = "{:.2f}".format(np.sum(sample['finalvalue']))
        eur = ' ' * 39 + 'EUR'
        basket = ''
        for i, row in sample.iterrows():
            basket += row['short_title']
            char_in_line = 44 - len(row['short_title'])
            if row['quantity'] > 1:  # adding multiplication row, calculating characters left
                basket += nl
                calculation = "     " + "{:.0f}".format(row['quantity']) + "   x   " + "{:.2f}".format(row['price'])
                char_in_line = 44 - len(calculation)
                basket += calculation

            # adding price and final letter, taking characters in account to assign spaces
            price_and_letter = "{:.2f}".format(row['fullvalue']) + " " + self._random_word(
                seed=seed + rnd.randint(0, 100),
                length=(1, 2), capital=False,
                alluper=True)
            char_in_line -= len(price_and_letter)
            basket += " " * char_in_line + price_and_letter + nl

            # if there was a discount i add a line with subtracted sum
            if row['discount'] > 0:
                message = "   " + "Aktionsersparnis"
                char_in_line = 44 - len(message)
                basket += message

                discounted = "{:.2f}".format(row['finalvalue'] - row['fullvalue'])
                char_in_line -= (len(discounted) + 2)
                basket += " " * char_in_line + discounted + nl

        summe = "SUMME    :"
        char_in_line = 44 - len(summe) - len(summ) - 2
        summe += " " * char_in_line + summ + nl

        # if ersparnis is 0, then text it corresponds to is ""
        # else ->I introduce a new line
        ersparnis = ""
        if np.sum(sample['discount']) > 0:
            ersparnis = " Ihre Ersparnis heute: " + "{:.2f}".format(
                sum(sample['fullvalue']) - sum(sample['finalvalue'])) + nl
        separator_type2 = "=" * 45

        # 3 -- generate fake footer
        zahlung = 'Zahlung ' + self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 20)) + '\t' * 8 + summ
        empty = ' ' * 45
        bezahlt = 'BEZAHLT ' + self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 8)) + \
                  self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 10)) + \
                  '\t' * 6 + summ
        hash = "#" * 12 + str(rnd.randint(100, 999)) + '\t' * 6 + "0000"
        num = str(
            rnd.randint(10000000, 99999999)) + f"   {rnd.randint(1, 32)}.{rnd.randint(1, 13)}.{rnd.randint(2021, 3000)}" \
                                               f"/{rnd.randint(0, 24)}:{rnd.randint(1, 60)}:{rnd.randint(1, 60)}" + \
              "   " + str(rnd.randint(10000, 99999))
        num2 = "GEN.NR.:" + str(rnd.randint(10000, 99999))
        num3 = str(rnd.randint(100000000, 999999999))
        num4 = "\t" * 3 + self._random_word(seed=seed + rnd.randint(0, 100), capital=False, alluper=True,
                                            length=(31, 33))
        someword = self._random_word(seed=seed + rnd.randint(0, 100), length=(9, 12), capital=False, alluper=True)

        # 4 -- add font to matplotlib and use it
        font_files = font_manager.findSystemFonts(fontpaths=self.font_path)
        for font_file in font_files: font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = self.font_name

        # 5 -- text to figure, adjusting bill length to text
        DPI = 300
        cm2inch = 1 / 2.54  # centimeters in inches

        text1 = street + nl + city + nl + tel + nl + nl + date + nl + separator_type1
        text2 = eur + nl + basket + separator_type1 + nl + summe + ersparnis + separator_type2 + nl
        text3 = zahlung + nl + empty + nl + bezahlt + nl + hash + nl + num + nl + num2 + nl + num3 + nl + num4 + nl + someword

        text = text1 + text2 + text3
        new_lines = text.count("\n") + 1
        height = (1 + new_lines) * 0.255 + 3  # cm!
        height = height if height > 14 else 14
        width = 8

        bill = plt.figure(figsize=(width * cm2inch, height * cm2inch), dpi=DPI)
        plt.axis('off')

        x_offset1 = 4 * cm2inch * DPI  # 4 cm -> 1/2.54 * DPI
        y_offset1 = 2.5 * cm2inch * DPI
        plt.annotate(text=text1, xy=(x_offset1, y_offset1), ha='center', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        x_offset2 = 0.8 * cm2inch * DPI  # 1 cm -> 1/2.54 * DPI
        y_offset2 = 4 * cm2inch * DPI
        plt.annotate(text=text2, xy=(x_offset2, y_offset2), ha='left', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        x_offset3 = 0.8 * cm2inch * DPI  # 1 cm -> 1/2.54 * DPI
        y_offset3 = 4 * cm2inch * DPI + (text2.count("\n") + 1) * 0.255 * cm2inch * DPI
        plt.annotate(text=text3, xy=(x_offset3, y_offset3), ha='left', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        # 6 -- place spar logo
        logo = Image.open(self.logo).convert('L')
        logo = logo.rotate(180)
        logo_max_height = int(0.6 * cm2inch * DPI)  # 0.6cm
        logo.thumbnail(size=(500, logo_max_height), resample=Image.ANTIALIAS)
        logo_arr = np.array(logo).astype(np.float)
        logo_width = logo.size[0] / (cm2inch * DPI)

        x_offset = ((8 - logo_width) / 2) * cm2inch * DPI  # 2.2 cm -> 1/2.54 * DPI
        y_offset = 1.8 * cm2inch * DPI
        bill.figimage(logo_arr, origin='upper', xo=x_offset, yo=y_offset, cmap='binary_r')

        # 7 -- create a mask of needed areas
        mask = self._create_mask(yo1=y_offset, yo2=y_offset2, yo3=y_offset3, logo_h=logo.size[1]/(cm2inch * DPI),
                                 xol=x_offset, logo_w=logo_width, width=width, height=height,
                                 cm2inch=cm2inch, DPI=DPI)

        return bill, mask


class PennyImageGenerator(ImageGenerator):
    pass

    def _generate_image(self, sample: pd.DataFrame, seed: int) -> (plt.Figure, np.array):
        """Hidden method for inside use only.
        Generates an image of bill
        @:param sample: pandas DataFrame which has all needed data inside
        @:return tuple: matplotlib figure object, image of bill and np.array of needed areas"""

        # 0 -- preventive cleanup
        try:
            plt.close('all')
            plt.cla()
            plt.clf()
        except:
            pass
        rnd.seed(seed)

        # 1 -- generate fake header
        nl = "\n"
        message = nl + self._random_word(seed=seed + rnd.randint(0, 100), length=(20, 40), alluper=1, capital=0) + \
                  nl + self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 30), alluper=1, capital=0) + \
                  nl + self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 30), alluper=1, capital=0)
        datum = "Datum: " + f"{str(rnd.randint(1, 32)).zfill(2)}.{str(rnd.randint(1, 13)).zfill(2)}.{rnd.randint(2021, 3000)}" + \
                " " * 20 + \
                f"Zeit: {str(rnd.randint(0, 24)).zfill(2)}:{str(rnd.randint(1, 60)).zfill(2)}"

        # 2 -- generate basket
        summ = "{:.2f}".format(np.sum(sample['finalvalue']))
        basket = ''
        for i, row in sample.iterrows():
            if row['quantity'] > 1:  # adding multiplication row, calculating characters left
                calculation = "     " + "{:.0f}".format(row['quantity']) + " x   " + "{:.2f}".format(row['price'])
                basket += (calculation + nl)
            basket += row['short_title']
            char_in_line = 48 - len(
                row['short_title']) - 13 - 1  # 11 == spaces and digits after letter, 1 stands for letter
            basket += " " * char_in_line + self._random_word(seed=+rnd.randint(0, 100), length=(1, 2),
                                                             capital=False, alluper=True)  # adding letter

            price = "{:.2f}".format(row['fullvalue'])  # adding price
            char_in_line = 13 - len(price)
            basket += " " * char_in_line + price + nl

            # if there was a discount i add a line with subtracted sum
            if row['discount'] > 0:
                sale = "AKTIONSNACHLASS"
                char_in_line = 48 - len(sale) - 13 - 1  # 11 == spaces and digits after letter, 1 stands for letter
                basket += sale
                basket += " " * char_in_line + self._random_word(seed=seed + rnd.randint(0, 100), length=(1, 2),
                                                                 capital=False, alluper=True)  # adding letter

                discounted = "{:.2f}".format(row['finalvalue'] - row['fullvalue'])
                char_in_line = 13 - len(discounted)
                basket += " " * char_in_line + discounted + nl

        separator_type1 = "-" * 48
        summe = "Summe"
        char_in_line = 48 - len(summe) - 14 - 3  # 12 == spaces and digits after EUR, 3 stands for EUR
        summe += " " * char_in_line + "EUR"
        char_in_line = 14 - len(str(summ))
        summe += " " * char_in_line + str(summ)
        separator_type2 = "=" * 48

        # 3 -- generate fake footer
        if rnd.normal() >= 0.5:
            bezahlt = "B E Z A H L T" + (48 - 13 - 10) * " " + str(rnd.randint(10000000, 99999999)) + nl
            gegeben = "Gegeben   KK " + self._random_word(seed=seed + rnd.randint(0, 100), length=(4, 13),
                                                          capital=True)
            char_in_line = 48 - len(gegeben) - 5  # 5 stands for summ
            gegeben += " " * char_in_line + str(rnd.randint(10, 99)) + "." + str(rnd.randint(10, 99)) + nl
            art = self._random_word(seed=seed + rnd.randint(0, 100), length=(4, 13),
                                    capital=False, alluper=True)
            char_in_line = 48 - len(art) - 6  # 6 for some number
            art += " " * char_in_line + str(rnd.randint(100000, 999999)) + nl

            art += "XXXX " * 4 + str(rnd.randint(1000, 9999)) + "      (" + str(rnd.randint(1, 9)) + \
                   ")" + " " * 10 + f"{str(rnd.randint(1, 13)).zfill(2)}/{str(rnd.randint(1, 32)).zfill(2)}" + nl
            art += self._random_word(seed=seed + rnd.randint(0, 100), length=(15, 16),
                                     capital=False, alluper=True)
            art += " " * 17 + "Beleg Nr.:" + str(rnd.randint(100000, 999999)) + nl
            art += self._random_word(seed=seed + rnd.randint(0, 100), length=(15, 16),
                                     capital=False, alluper=True)
            art += " " * 25 + self._random_word(seed=seed + rnd.randint(0, 100), length=(3, 4),
                                                capital=False, alluper=True) + nl
            art += str(rnd.randint(10000000, 99999999)) + " " + str(rnd.randint(100000, 999999)) + nl
            art += self._random_word(seed=seed + rnd.randint(0, 100), length=(40, 42),
                                     capital=False, alluper=True) + nl
        else:
            gegeben = "Gegeben   Bar"
            bezahlt = ""
            char_in_line = 48 - len(gegeben) - 5  # 5 stands for given money
            gegeben += " " * char_in_line + str(rnd.randint(10, 99)) + "." + str(rnd.randint(10, 99))
            art = "Restgeld"
            char_in_line = 48 - len(art) - 14 - 3  # 12 == spaces and digits after EUR, 3 stands for EUR
            art += " " * char_in_line + "EUR"
            art += " " * 9 + str(rnd.randint(10, 99)) + "." + str(rnd.randint(10, 99))

        # 4 -- add font to matplotlib and use it
        font_files = font_manager.findSystemFonts(fontpaths=self.font_path)
        for font_file in font_files: font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = self.font_name

        # 5 -- text to figure, adjusting bill length to text
        DPI = 300
        cm2inch = 1 / 2.54  # centimeters in inches

        text1 = message + nl + datum + nl + nl
        text2 = basket + separator_type1 + nl + summe + nl + separator_type2 + nl
        text3 = gegeben + nl + bezahlt + art

        text = text1 + text2 + text3
        new_lines = text.count("\n") + 1
        height = (1 + new_lines) * 0.255 + 5  # cm!
        height = height if height > 14 else 14
        width = 8

        bill = plt.figure(figsize=(width * cm2inch, height * cm2inch), dpi=DPI)
        plt.axis('off')

        x_offset1 = 4 * cm2inch * DPI  # 4 cm -> 1/2.54 * DPI
        y_offset1 = 4.5 * cm2inch * DPI
        plt.annotate(text=text1, xy=(x_offset1, y_offset1), ha='center', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        x_offset2 = 0.7 * cm2inch * DPI  # 1 cm -> 1/2.54 * DPI
        y_offset2 = (4.5 + 6 * 0.255) * cm2inch * DPI  # 4 lines, 2 empty  == 6
        plt.annotate(text=text2, xy=(x_offset2, y_offset2), ha='left', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        x_offset3 = 0.7 * cm2inch * DPI  # 1 cm -> 1/2.54 * DPI
        y_offset3 = (4.5 + (text2.count("\n") + 1 + 6) * 0.255) * cm2inch * DPI
        plt.annotate(text=text3, xy=(x_offset3, y_offset3), ha='left', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        # 6 -- place spar logo
        logo = Image.open(self.logo).convert('L')
        logo = logo.rotate(180)
        logo_max_height = int(2.8 * cm2inch * DPI)  # 2.8cm
        logo.thumbnail(size=(logo_max_height, logo_max_height), resample=Image.ANTIALIAS)
        logo_arr = np.array(logo).astype(np.float)
        logo_width = logo.size[0] / (cm2inch * DPI)

        x_offset = ((8 - logo_width) / 2) * cm2inch * DPI  # 2.2 cm -> 1/2.54 * DPI
        y_offset = 2.3 * cm2inch * DPI
        bill.figimage(logo_arr, origin='upper', xo=x_offset, yo=y_offset, cmap='binary_r')

        # 7 -- create a mask of needed areas
        mask = self._create_mask(yo1=y_offset, yo2=y_offset2, yo3=y_offset3, logo_h=logo.size[1] / (cm2inch * DPI),
                                 xol=x_offset, logo_w=logo_width, width=width, height=height,
                                 cm2inch=cm2inch, DPI=DPI)
        return bill, mask


class BillaImageGenerator(ImageGenerator):
    pass

    def _generate_image(self, sample: pd.DataFrame, seed: int) -> (plt.Figure, np.array):
        """Hidden method for inside use only.
        Generates an image of bill
        @:param sample: pandas DataFrame which has all needed data inside
        @:return tuple: matplotlib figure object, image of bill and np.array of needed areas"""

        # 0 -- preventive cleanup
        try:
            plt.close('all')
            plt.cla()
            plt.clf()
        except:
            pass
        rnd.seed(seed)

        # 1 -- generate fake header
        nl = "\n"
        billag = "Billa AG"
        city = str(rnd.randint(1000, 9999)) + " " + self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 15))
        street = self._random_word(seed=seed + rnd.randint(0, 100), length=(5, 30)) + "." + str(rnd.randint(1, 100))
        tel = "TEL: 0" + str(rnd.randint(100, 999)) + "-" + str(rnd.randint(100000, 999999))
        atu = self._random_word(seed=seed + rnd.randint(0, 100),
                                length=(3, 4), capital=False, alluper=True) + str(rnd.randint(10000000, 99999999))

        datum = "Datum: " + f"{str(rnd.randint(1, 32)).zfill(2)}.{str(rnd.randint(1, 13)).zfill(2)}.{rnd.randint(2021, 3000)}" + \
                " " * 20 + \
                f"Zeit: {str(rnd.randint(0, 24)).zfill(2)}:{str(rnd.randint(1, 60)).zfill(2)}"
        separator_type1 = "-" * 48

        # 2 -- generate basket
        summ = "{:.2f}".format(np.sum(sample['finalvalue']))
        basket = ''
        for i, row in sample.iterrows():
            if row['quantity'] > 1:  # adding multiplication row, calculating characters left
                calculation = "     " + "{:.0f}".format(row['quantity']) + " x   " + "{:.2f}".format(row['price'])
                basket += (calculation + nl)
            basket += row['short_title']
            char_in_line = 48 - len(
                row['short_title']) - 13 - 1  # 11 == spaces and digits after letter, 1 stands for letter
            basket += " " * char_in_line + self._random_word(seed=+rnd.randint(0, 100), length=(1, 2),
                                                             capital=False, alluper=True)  # adding letter

            price = "{:.2f}".format(row['fullvalue'])  # adding price
            char_in_line = 13 - len(price)
            basket += " " * char_in_line + price + nl

            # if there was a discount i add a line with subtracted sum
            if row['discount'] > 0:
                sale = "AKTIONSNACHLASS"
                char_in_line = 48 - len(sale) - 13 - 1  # 11 == spaces and digits after letter, 1 stands for letter
                basket += sale
                basket += " " * char_in_line + self._random_word(seed=seed + rnd.randint(0, 100), length=(1, 2),
                                                                 capital=False, alluper=True)  # adding letter

                discounted = "{:.2f}".format(row['finalvalue'] - row['fullvalue'])
                char_in_line = 13 - len(discounted)
                basket += " " * char_in_line + discounted + nl

        summe = "Summe"
        char_in_line = 48 - len(summe) - 14 - 3  # 12 == spaces and digits after EUR, 3 stands for EUR
        summe += " " * char_in_line + "EUR"
        char_in_line = 14 - len(str(summ))
        summe += " " * char_in_line + str(summ)
        separator_type2 = "=" * 48

        # 3 -- generate fake footer
        if rnd.normal() >= 0.5:
            bezahlt = "B E Z A H L T" + (48 - 13 - 10) * " " + str(rnd.randint(10000000, 99999999)) + nl
            gegeben = "Gegeben   KK " + self._random_word(seed=seed + rnd.randint(0, 100), length=(4, 13),
                                                          capital=True)
            char_in_line = 48 - len(gegeben) - 5  # 5 stands for summ
            gegeben += " " * char_in_line + str(rnd.randint(10, 99)) + "." + str(rnd.randint(10, 99)) + nl
            art = self._random_word(seed=seed + rnd.randint(0, 100), length=(4, 13),
                                    capital=False, alluper=True)
            char_in_line = 48 - len(art) - 6  # 6 for some number
            art += " " * char_in_line + str(rnd.randint(100000, 999999)) + nl

            art += "XXXX " * 4 + str(rnd.randint(1000, 9999)) + "      (" + str(rnd.randint(1, 9)) + \
                   ")" + " " * 10 + f"{str(rnd.randint(1, 13)).zfill(2)}/{str(rnd.randint(1, 32)).zfill(2)}" + nl
            art += self._random_word(seed=seed + rnd.randint(0, 100), length=(15, 16),
                                     capital=False, alluper=True)
            art += " " * 17 + "Beleg Nr.:" + str(rnd.randint(100000, 999999)) + nl
            art += self._random_word(seed=seed + rnd.randint(0, 100), length=(15, 16),
                                     capital=False, alluper=True)
            art += " " * 25 + self._random_word(seed=seed + rnd.randint(0, 100), length=(3, 4),
                                                capital=False, alluper=True) + nl
            art += str(rnd.randint(10000000, 99999999)) + " " + str(rnd.randint(100000, 999999)) + nl
            art += self._random_word(seed=seed + rnd.randint(0, 100), length=(40, 42),
                                     capital=False, alluper=True) + nl
        else:
            gegeben = "Gegeben   Bar"
            bezahlt = ""
            char_in_line = 48 - len(gegeben) - 5  # 5 stands for given money
            gegeben += " " * char_in_line + str(rnd.randint(10, 99)) + "." + str(rnd.randint(10, 99))
            art = "Restgeld"
            char_in_line = 48 - len(art) - 14 - 3  # 12 == spaces and digits after EUR, 3 stands for EUR
            art += " " * char_in_line + "EUR"
            art += " " * 9 + str(rnd.randint(10, 99)) + "." + str(rnd.randint(10, 99))

        # 4 -- add font to matplotlib and use it
        font_files = font_manager.findSystemFonts(fontpaths=self.font_path)
        for font_file in font_files: font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = self.font_name

        # 5 -- text to figure, adjusting bill length to text
        DPI = 300
        cm2inch = 1 / 2.54  # centimeters in inches

        text1 = billag + nl + city + nl + street + nl + tel + nl + atu + nl + separator_type1 + nl + datum + nl
        text2 = basket + separator_type1 + nl + summe + nl + separator_type2 + nl
        text3 = gegeben + nl + bezahlt + art

        text = text1 + text2 + text3
        new_lines = text.count("\n") + 1
        height = (1 + new_lines) * 0.255 + 7  # cm!
        height = height if height > 14 else 14
        width = 8

        bill = plt.figure(figsize=(width * cm2inch, height * cm2inch), dpi=DPI)
        plt.axis('off')

        x_offset1 = 4 * cm2inch * DPI  # 4 cm -> 1/2.54 * DPI
        y_offset1 = 4.8 * cm2inch * DPI
        plt.annotate(text=text1, xy=(x_offset1, y_offset1), ha='center', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        x_offset2 = 0.7 * cm2inch * DPI  # 1 cm -> 1/2.54 * DPI
        y_offset2 = (4.8 + 8 * 0.255) * cm2inch * DPI  # 4 lines, 2 empty  == 6
        plt.annotate(text=text2, xy=(x_offset2, y_offset2), ha='left', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        x_offset3 = 0.7 * cm2inch * DPI  # 1 cm -> 1/2.54 * DPI
        y_offset3 = (4.8 + (text2.count("\n") + 1 + 8) * 0.255) * cm2inch * DPI
        plt.annotate(text=text3, xy=(x_offset3, y_offset3), ha='left', fontsize=11,
                     xycoords='figure pixels', rotation=180)

        # 6 -- place spar logo
        logo = Image.open(self.logo).convert('L')
        logo = logo.rotate(180)
        logo_max_height = int(1.5 * cm2inch * DPI)  # 2.8cm
        logo.thumbnail(size=(500, logo_max_height), resample=Image.ANTIALIAS)
        logo_arr = np.array(logo).astype(np.float)
        logo_width = logo.size[0] / (cm2inch * DPI)

        x_offset = ((8 - logo_width) / 2) * cm2inch * DPI  # 2.2 cm -> 1/2.54 * DPI
        y_offset = 2.6 * cm2inch * DPI
        bill.figimage(logo_arr, origin='upper', xo=x_offset, yo=y_offset, cmap='binary_r')

        # 7 -- create a mask of needed areas
        mask = self._create_mask(yo1=y_offset, yo2=y_offset2, yo3=y_offset3, logo_h=logo.size[1] / (cm2inch * DPI),
                                 xol=x_offset, logo_w=logo_width, width=width, height=height,
                                 cm2inch=cm2inch, DPI=DPI)
        return bill, mask

import string
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mtplb
from numpy import random as rnd
from matplotlib import font_manager
from matplotlib import pyplot as plt

class SparImageGenerator():
    def __init__(self, saving_path:str, data_path:str, logo_path:str, font_path:str, font_name:str):
        self.directory = saving_path
        self.datadf = pd.read_csv(data_path, sep=';', index_col=0)
        self.logo = logo_path
        self.font_path = font_path
        self.font_name = font_name

    """Method which allows to generate n images of supermarket bills
    Images are saved in self.directory, alongside with dataframes describing their content"""
    def generate_bills(self, n=1) -> None:
        for attempt in range(n):
            sample = self._random_busket()#select some random goods and quantities
            figure = self._generate_image(sample=sample)#generate image and data
            #generate unique name
            #save

    """Hidden method for inside use only.
    Generates a random busket of goods
    :return Products pd.DataFrame with such columns: 'full_title', 'title', 'specification', 'description', 
                                                     'price', 'cat1', 'cat2', 'cat3', 'quantity', 'discount',
                                                     'fullvalue', 'finalvalue'"""
    def _random_busket(self) -> pd.DataFrame:
        #STEP1 -- generate random number of unique items, sample random items
        num_items = rnd.randint(1, 20)
        sample_ids = rnd.choice(a=self.datadf.index, size=num_items, replace=False)
        sample = self.datadf.loc[sample_ids].copy(deep=True)

        #STEP 2 -- for each product in random busket generate quantity, discount and value

        #small functions for fast columns initialisation
        def quantity(x):
            threshold = rnd.normal(0,1)
            if threshold<0.6:
                return 1
            if threshold>=0.6 and threshold<0.85:
                return rnd.randint(2,4)
            else:
                return rnd.randint(4,20)

        def discount(x):
            probability = rnd.rand()
            if probability < 0.95:
                return 0
            else:
                return rnd.choice(a=[5,10,15,20,25,30,35,40,45,50],size=1)[0]/100

        #columns introduction
        sample['quantity'] = None
        sample['quantity'] = sample['quantity'].apply(quantity)
        sample['discount'] = None
        sample['discount'] = sample['discount'].apply(discount)

        sample['fullvalue'] = sample['price']*sample['quantity']
        sample['finalvalue'] = sample['fullvalue']-(sample['fullvalue']*sample['discount'])

        return sample

    """Hidden method for inside use only.
    Generates an image of bill
    @:param sample: pandas DataFrame which has all needed data inside
    @:return matplotlib figure object, image of bill"""
    def _generate_image(self, sample:pd.DataFrame) -> plt.Figure:
        DPI = 100
        cm2inch = 1 / 2.54  # centimeters in inches
        cm2pixel = DPI/2.54



        #1 -- generate fake header
        street = self._random_word(length=(5,30))+"."+str(rnd.randint(1,100))
        city = str(rnd.randint(1000,9999))+" "+self._random_word(length=(5,15))
        tel = "Tel.: 0"+str(rnd.randint(100,999))+" "+str(rnd.randint(100000,999999))
        date = f"Ihr Einkauf am {rnd.randint(1,32)}.{rnd.randint(1,13)}.{rnd.randint(2021,3000)} " \
               f"um {rnd.randint(0,24)}:{rnd.randint(1,60)} Uhr"
        separator_type1 = "-"*50

        #2 -- generate basket
        nl = "\n"
        summ = "{:.2f}".format(np.sum(sample['finalvalue']))
        eur = ' '*44 + 'EUR'
        basket = ''
        for i,row in sample.iterrows():
            basket+=row['short_title']
            char_in_line= 49 - len(row['short_title'])
            if row['quantity']>1: #adding multiplication row, calculating characters left
                basket+=nl
                calculation = "     "+"{:.0f}".format(row['quantity'])+"   x   "+"{:.2f}".format(row['price'])
                char_in_line = 49 - len(calculation)
                basket+=calculation

            # adding price and final letter, taking characters in account to assign spaces
            price_and_letter = "{:.2f}".format(row['fullvalue'])+" "+self._random_word(length=(1,2),capital=False,alluper=True)
            char_in_line -= len(price_and_letter)
            basket+= " "*char_in_line + price_and_letter + nl

            #if there was a discount i add a line with subtracted sum
            if row['discount']>0:
                message = "   " + "Aktionsersparnis"
                char_in_line = 49 - len(message)
                basket += message

                discounted = "{:.2f}".format(row['finalvalue']-row['fullvalue'])
                char_in_line -= (len(discounted)+2)
                basket += " "*char_in_line + discounted + nl

        summe = "SUMME    :"
        char_in_line = 49 - len(summe)-len(summ)-2
        summe+= " "*char_in_line + summ + nl

        #if ersparnis is 0, then text it corresponds to is ""
        #else ->I introduce a new line
        ersparnis = ""
        if np.sum(sample['discount'])>0:
            ersparnis = " Ihre Ersparnis heute: "+"{:.2f}".format(sum(sample['fullvalue'])-sum(sample['finalvalue']))+nl
        separator_type2 = "="*50

        #3 -- generate fake footer
        zahlung = 'Zahlung '+ self._random_word(length=(5,30)) + '\t'*8 + summ
        bezahlt = 'BEZAHLT '+ self._random_word(length=(5,8)) + self._random_word(length=(5,10))\
                   + '\t'*6 + summ
        hash = "#"*12 + str(rnd.randint(100,999))+ '\t'*6 + "0000"
        num = str(rnd.randint(10000000,99999999)) + f"   {rnd.randint(1,32)}.{rnd.randint(1,13)}.{rnd.randint(2021,3000)}" \
                                                    f"/{rnd.randint(0,24)}:{rnd.randint(1,60)}:{rnd.randint(1,60)}" +\
                                                    "   " + str(rnd.randint(10000,99999))
        num2 = "GEN.NR.:" + str(rnd.randint(10000,99999))
        num3 = str(rnd.randint(100000000,999999999))
        num4 = "\t"*3 + self._random_word(capital=False, alluper=True, length=(31,33))
        someword = self._random_word(length=(9,12), capital=False, alluper=True)

        #4 -- add font to matplotlib and use it
        font_files = font_manager.findSystemFonts(fontpaths=self.font_path)
        for font_file in font_files: font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = self.font_name

        # 5 -- text to figure, adjusting bill length to text
        text1 = street + nl + city + nl + tel + nl + nl + date + nl + separator_type1 + nl
        text2 = eur + nl + basket + separator_type1 + nl + summe + ersparnis + separator_type2 + nl
        text2 += zahlung + nl + nl + bezahlt + nl + hash + nl + num + nl + num2 + nl + num3 + nl + num4 + nl + someword

        text = text1 + text2
        new_lines = text.count("\n")
        height = new_lines * 0.3 + 2.5

        bill = plt.figure(figsize=(8 * cm2inch, height * cm2inch), dpi=DPI)
        bill.text(s=text1, x=0.5, y=(height-3.6) * cm2pixel, ha='center', fontsize=10)
        bill.text(s=text2, x=0.1, y=0.2, ha='left', fontsize=10)

        #6 -- place spar logo
        logo = Image.open(self.logo).convert('L')
        logo = logo.resize((int(4 * cm2pixel), int(0.5 * cm2pixel)))
        logo_arr = np.array(logo).astype(np.float)
        bill.figimage(logo_arr, origin='upper', xo=2.2 * cm2pixel, yo=(height-1.8) * cm2pixel)

        return bill

    """A hidden method to generate word-like strings
    @:param length A tuple of integers controlling th length of word
    @:param capital Boolean, defining if the first letter is uppercase
    @:return String"""
    def _random_word(self,length:tuple, capital=True, alluper=False, alllower=False) -> str:
        lowercase = list(string.ascii_lowercase)
        uppercase = list(string.ascii_uppercase)
        s = ''
        if capital:
            s += rnd.choice(uppercase,1)[0]
            s += ''.join(rnd.choice(lowercase,rnd.randint(length[0],length[1])))
        elif alluper:
            s += ''.join(rnd.choice(uppercase, rnd.randint(length[0], length[1])))
        elif alllower:
            s += ''.join(rnd.choice(lowercase, rnd.randint(length[0], length[1])))
        return s
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from selenium import webdriver

"""General Class created to scrap goods from online shops
Should be extended."""
class Scrapper():
    def __init__(self, start_url:str, saving_file:str):
        self.homepage = start_url
        self.file = saving_file
        self.driver_path = os.path.abspath(os.path.join("../materials_for_preproseccing/firefox", "geckodriver"))
        self.driver = None

    """Send driver to homepage.
        https://www.interspar.at/shop/lebensmittel/"""
    def to_homepage(self) -> None:
        self.driver.get(self.homepage)
        self.random_sleep()

    """Receiving a dictionary of products from an ÜBERSICHT page it writes a df, 
        concatenates with already existing csv and updates the csv file.
        :param: dict with following keys: 'category', 'price', 'summary', 'title'
                           and values as lists with the same length = number of objects in overview"""
    def write_to_file(self, products: dict) -> None:
        new_df = pd.DataFrame(products)
        try:
            csv_df = pd.read_csv(self.file, sep=";", index_col=0)
            full_df = pd.concat([new_df, csv_df])
        except:
            full_df = new_df
        full_df.to_csv(self.file, sep=';')

    """Function enables different idling time for
         marionette driver to behave less suspicious"""
    def random_sleep(self, threshold=1) -> None:
        long = np.random.random()  # make a long or short delay
        if long > threshold:
            second = np.random.randint(7, 20)
        else:
            second = 1
        time.sleep(second)

"""Class which is specifically created for scrappint Spar. Inherits from a general class"""
class SparScrapper(Scrapper):
    pass

    """
    Method establishes driver, goes to spar website and accepts cookies.
    Nothing is returned. Property self.driver is updated
    """
    def establish_driver(self) -> None:
        try: self.driver.quit() # close an old one
        except: pass

        self.driver = webdriver.Firefox(executable_path=self.driver_path)
        self.random_sleep(threshold=0)
        self.driver.get(self.homepage)
        self.random_sleep(threshold=0)
        attempt = -1
        # accept cookies
        while attempt < 5:
            try:
                cook = "/html/body/div[2]/div/div[2]/span[2]/a"
                self.driver.find_element(by='xpath', value=cook).click()
                self.random_sleep(threshold=0)
                break
            except:
                attempt += 1
                self.driver.quit()
                self.driver = webdriver.Firefox(executable_path=self.driver_path)
                self.random_sleep(threshold=0)
        if attempt==5:
            print("Failed to go to homepage.")

    """ 
    Method opens an overview page of the category by defined id.
    @:param: cat_id is an integer from 0 to 15 incl.
    """
    def open_category(self,cat_id) -> None:
        self.to_homepage()
        self.random_sleep()
        all_cat = "/html/body/div[8]/div[2]/div[5]/div/div[1]/a"
        self.driver.find_element(by='xpath', value=all_cat).click() # show all categories

        category = [item for item in self.driver.find_elements(by="class name", value="flyout-categories__item")
                         if item.get_attribute('data-level')=='1'][cat_id] # select one with correspondent id
        category.click() # open category /or/ open the full page (depends if ÜBERSICHT available)
        self.random_sleep()
        try: # last categories in web-site have no ÜBERSICHT section
            ubersicht = [item for item in self.driver.find_elements(by="class name", value="ellipsisText")
                     if "ÜBERSICHT" in item.text][0] # collect corresponding overview page
            print(f"{ubersicht.text} is opened {'='*60}")
            ubersicht.click() # open
        except:
            pass
        self.random_sleep()

    """While being inside overall view of some category it collects all products
    from all pages.
    :return: dict with following keys: 'category', 'price', 'summary', 'title'
                       and values as lists with the same length = number of objects in overview"""
    def collect_products(self, start_page = 1) -> None:
        holder = []
        page = 1
        while True: #looping over pages
            if page<start_page: #in case I want to specify where to start
                self.driver.find_element(by='class name', value='next').click()
                self.random_sleep()
                print(f"\tPage {page} is skipped.")
                page += 1
                continue
            print(f"\tPage {page}:")
            all_products = self.driver.find_elements(by='class name', value='productBox')
            for product in tqdm(all_products):
                try:
                    main_title, title = product.find_elements(by='class name', value='productTitle')
                    compound_title = main_title.text+" "+title.text
                    title = title.text
                    specifications = product.find_element(by='class name', value='productSummary').get_attribute("title")
                    integer_price = float(product.find_element(by='class name', value='priceInteger').text)
                    decimal_price = float(product.find_element(by='class name', value='priceDecimal').text) / 100
                    price = integer_price + decimal_price

                    #open product page in a new tab to collect categories path and description
                    product_url = self.homepage[:-1]+product.get_attribute('data-url')
                    self.driver.execute_script(f"window.open('{product_url}', 'new_window')")
                    self.driver.switch_to.window(self.driver.window_handles[1]) # switch to the tab
                    # self.random_sleep()
                    three_last_categories = []
                    for _ in range(5): # some glitching here, adding more response time allowance
                        try:
                            path_container = self.driver.find_element(by='class name', value='breadcrumbContainer')
                            three_last_categories = [elem.text for elem in
                                                     path_container.find_elements(by="tag name", value="li")
                                                     if elem.get_attribute('class') != 'separator'][-4:-1]
                            break
                        except: self.random_sleep()

                    if len(three_last_categories)<3: # in case a path to product is short
                        diff = 3-len(three_last_categories)
                        pad = ['No info']*diff
                        three_last_categories = three_last_categories+pad

                    try: description = self.driver.find_element(by='class name', value='baseInformation').text
                    except: description = 'No description' #some products have no description

                    row = [compound_title, title, specifications, description, price]
                    row += three_last_categories
                    holder.append(row)

                    self.driver.close()# close product tab
                    self.driver.switch_to.window(self.driver.window_handles[0])  # switch to UBERSICHT tab
                    # self.random_sleep( )
                except ValueError: pass# some advertisement is embedded as "productBox", no data extraction here

            # when page is done, I save results and print report
            keys = ['full_title', 'title', 'specification', 'description', 'price', 'cat1', 'cat2', "cat3"]
            dictionary = dict(zip(keys, np.transpose(holder)))
            self.write_to_file(dictionary)
            holder = []
            print(f"\tPage {page} is done.\n")

            try: # click next page button
                self.driver.find_element(by='class name', value='next').click()
                self.random_sleep()
                page += 1
            except: break# all pageas are exausted

    """The whole route of retrieving data on all possible
    items from InterSpar online shop
    @:param: cat_range Is an iterable filled with integers, each correspondent to a category id
    @:param: starting_page Is an integer, used in case if I want to scrap only 1 category from a specific page."""
    def scrap(self, cat_range=range(1,16), starting_page=0) -> None:
        self.establish_driver()
        # there are 16 categories
        # but I skip regional products
        for cat_id in cat_range:
            self.open_category(cat_id=cat_id)
            self.collect_products(start_page=starting_page)
            starting_page = 0 #reset for next category
        self.driver.quit()
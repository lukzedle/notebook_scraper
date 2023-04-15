import requests
from bs4 import BeautifulSoup
import csv
import time
from random import uniform
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split


class Notebook_Scraper:
    """
    A class representing a Notebook_Scraper object.

    Methods:
    - __init__(self, url: str): Initializes a new Notebook_Scraper object with the given url and full_details_list list.
    - scrape(self): Get notebook details from self.url. Returns a list of dicts with notebook details.
    - save_data(self, data: list): Saves the list of dicts with notebooks details to a file.
    - open_data(self, data_csv_name_file)->list: Opens a csv file with dicts with notebooks details. Returns a list of dicts with notebooks details.
    """

    def __init__(self, url):
        self.url = url
        self.full_details_list = []

    def scrape(self) -> list:
        """
        Get notebook details from self.url.

        Returns:
          List of dicts with notebook details.
        """
        response = requests.get(self.url)
        content = response.content
        soup = BeautifulSoup(content, 'html.parser')

        self.last_page = int(soup.find_all('li', {
            'class': 'flex justify-center items-center w-8 h-8 rounded border border-gray-mercury cursor-pointer'})[
                                 -1].get_text())

        for page in range(1, self.last_page + 1):
            url = f'https://www.komputronik.pl/category/5022/laptopy.html?showBuyActiveOnly=0&p={page}'

            time.sleep(uniform(1.5, 2.5))
            response = requests.get(url)
            content = response.content

            soup = BeautifulSoup(content, 'html.parser')

            # for item in soup.find('div', {'class': 'tests-product-entry'}):
            #   print(item.find('div',{'class': 'py-1'}))
            counter = 0
            for item in soup.find_all('div', {'class': 'tests-product-entry'}):
                counter += 1
                notebook_dict = {}
                try:
                    price = item.find('div', {'class': 'text-3xl font-bold leading-8'}).get_text().replace('zł',
                                                                                                           '').replace(
                        ' ', '').replace('\\xa', '').strip()

                except:
                    price = np.nan
                try:
                    price_no_sale = item.find('span', {'class': 'line-through'}).get_text().replace('zł',
                                                                                                    '').replace(
                        ' ', '').replace('\\xa', '').strip()
                except:
                    price_no_sale = item.find('div', {'class': 'text-3xl font-bold leading-8'}).get_text().replace('zł',
                                                                                                                   '').replace(
                        ' ', '').replace('\\xa', '').strip()

                notebook_dict['price'] = float(price.replace(u'\xa0', '').replace(',', '.'))
                notebook_dict['price_no_sale'] = float(price_no_sale.replace(u'\xa0', '').replace(',', '.'))
                print(price, price_no_sale, counter)

                for detail in item.find_all('div', {'class': 'py-1'}):
                    if detail.find('span') == None:
                        pass
                    else:
                        try:
                            key_lap = detail.find('span').text.strip()
                        except:
                            continue
                        try:
                            value_lap = detail.find(class_='font-semibold').text.strip()
                        except:
                            value_lap = np.nan
                        notebook_dict[key_lap] = value_lap
                try:
                    try:
                        cpu = item.find('p', {'ng-if': '!$ctrl.showOnDesktop',
                                              'class': 'wrap-text mt-2 text-xs text-gray-gravel'}).text.split('|')[
                            0].split('LCD')[0]
                    except:
                        cpu = item.find('p', {'ng-if': '!$ctrl.showOnDesktop',
                                              'class': 'wrap-text mt-2 text-xs text-gray-gravel'}).text.split('|')[0]

                except:
                    cpu = np.nan
                notebook_dict['CPU:'] = cpu
                print(notebook_dict['CPU:'])

                try:
                    brand = item.find('h2', {
                        'class': 'font-headline text-lg font-bold leading-6 line-clamp-3 md:text-xl md:leading-8'}).find(
                        'a').text.strip().split(' ')[0]

                except:
                    brand = np.nan
                notebook_dict['Marka:'] = brand
                print(notebook_dict['Marka:'])

                self.full_details_list.append(notebook_dict)
        return self.full_details_list

    def save_data(self, data: list, file_name: str):
        """
          Saves the list of dicts with notebooks details to a file.

          Args:
              data: a list of dicts.

              file_name: a name of a file to be saved. Can pass directory with it.

        """
        with open(file_name, 'w', encoding='UTF8', newline='') as handler:
            writer = csv.writer(handler)
            for item in data:
                writer.writerow([item])

    def open_data(self, data_csv_name_file) -> list:
        """
           Opens a csv file with laptops details dicts.

          Args:
              data_csv_name_file: a file name with a file extension. If needed pass directory with it.

          Returns:
              List of dicts with notebook details.
        """
        self.full_details_list = []
        with open(f'{data_csv_name_file}', 'r') as handler:
            reader = csv.reader(handler)
            for it in reader:
                res = eval(it[0].replace("'", '"'))
                self.full_details_list.append(res)
        return self.full_details_list

def get_gpu_scores(graphic_cards: pd.Series) -> dict:
    """
    Scrape GPU benchmark scores, check the similarity between the scraped GPU and the ones passed in the argument.

    Args:
        graphic_cards: a pd.Series with graphic cards from scraped laptop data.

    Returns:
        matched_gpus: a dict where a key is a name of a graphics card present in the laptops dataframe and a value is a matched benchmark score.
    """

    url = 'https://browser.geekbench.com/opencl-benchmarks'

    response = requests.get(url)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')

    gpu_scores = {}
    gpus = []
    scores = []
    for item in soup.find_all('table', {'class': 'table benchmark-chart-table'}):
        for gpu in soup.find_all('td', {'class': 'name'}):
            gpus.append(gpu.text.strip())
        for score in soup.find_all('td', {'class': 'score'}):
            scores.append(int(score.text))

    for i in range(0, len(gpus)):
        gpu_scores[gpus[i]] = scores[i]

    print(gpu_scores)

    list_of_gpus = graphic_cards.unique().tolist()
    print(type(list_of_gpus[0]), list_of_gpus[0])

    # find the best matching GPU name in the dictionary for each GPU name in the list
    matches = {}
    matched_gpus = {}
    for gpu_name in list_of_gpus:
        best_match = None
        best_score = 0
        for name in gpu_scores.keys():
            score = fuzz.ratio(name, str(gpu_name))
            if score > best_score:
                best_match = gpu_name
                best_score = score
                best_value = gpu_scores[name]
        matches[gpu_name] = best_match
        matched_gpus[best_match] = best_value

    return matched_gpus

def separate_train_test(df, X_list: list, y_list:list):
    """
    Divides data into training and test sets
    Args:
        df: dataframe with data to be divided
        X_list: a list of independent variables that will divide 
        y_list: a list of dependent variables that will divide 

    Returns:

    """
    X = df[X_list]
    y = df[y_list]
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42) # zwraca tupla z czterema elementami 
    return X_train, X_test, y_train, y_test
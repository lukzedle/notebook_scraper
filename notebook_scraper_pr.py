import requests
from bs4 import BeautifulSoup
import csv
import time
from random import uniform
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

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

def data_modification(X: pd.DataFrame, y: pd.DataFrame):
    """
    Data processing of the transferred dataframe
    Args:
        X: a df of independent values
        y: a df of dependent values
    Returns:
        X: a df of a processed independent values
        y: a df of a processed dependent value
    """

    X['System operacyjny:'] = X['System operacyjny:'].map(
        lambda x: 'Windows' if x.lower().startswith('windows') else x)
    X['Wielkość pamięci RAM:'] = X['Wielkość pamięci RAM:'].str.split(' ').map(lambda x: x[0]).astype(
        int)
    most_common_ssd = X['Pojemność dysku SSD:'].value_counts().head().index[0]
    X['Pojemność dysku SSD:'].fillna(most_common_ssd, inplace=True)
    X['Pojemność dysku SSD:'] = X['Pojemność dysku SSD:'].apply(lambda x: x.split(' ')[0]).astype(int)
    ssd_threshold = 4000
    price_threshold = np.percentile(y, 75)
    merged_train = pd.concat([X, y], axis=1)
    filtered_df = merged_train[
        (merged_train['Pojemność dysku SSD:'] > ssd_threshold) & (merged_train['price'] < price_threshold)].index
    X.drop(index=filtered_df, axis=0, inplace=True)
    y.drop(index=filtered_df, axis=0, inplace=True)
    most_common = X['Rozdzielczość:'].value_counts().index[0]
    X['Rozdzielczość:'] = X['Rozdzielczość:'].fillna(most_common)
    X['Rozdzielczość:'] = X['Rozdzielczość:'].map(
        lambda x: int(str(x).split('x')[0]) * int(str(x).split('x')[1].strip()[:4]))
    top_cat = X['Karta graficzna:'].value_counts()[0]
    X['Karta graficzna:'] = X['Karta graficzna:'].fillna(top_cat)
    X['Pojemność dysku SSD 2:'] = np.where(X['Pojemność dysku SSD 2:'].isnull(), False, True)
    X['Podświetlana klawiatura:'] = np.where(X['Podświetlana klawiatura:'] == 'tak', True, False)
    most_common_type = X['Rodzaj laptopa:'].value_counts().head().index[0]
    X['Rodzaj laptopa:'].fillna(most_common_type, inplace=True)
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
    list_of_gpus = X['Karta graficzna:'].unique().tolist()
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
    X['GPU benchmark:'] = X['Karta graficzna:'].map(matched_gpus)
    url = 'https://browser.geekbench.com/processor-benchmarks'
    response = requests.get(url)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    cpu_scores = {}
    cpus = []
    scores = []
    single_core = soup.find('div', {'id': 'single-core'})
    # print(single_score)
    for item in single_core.find('table', {'class': 'table benchmark-chart-table', 'id': 'pc'}).find_all('a'):
        cpus.append(item.text.strip())
    for score in single_core.find('table', {'class': 'table benchmark-chart-table', 'id': 'pc'}).find_all('td', {
        'class': 'score'}):
        scores.append(int(score.text))
    url = 'https://browser.geekbench.com/mac-benchmarks'
    response = requests.get(url)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    single_core = soup.find('div', {'id': 'single-core'})
    for item in single_core.find('table', {'class': 'table benchmark-chart-table', 'id': 'mac'}).find_all('div', {
        'class': 'description'}):
        name = item.text.strip().split('@')[0].strip()
        cores = item.text.strip().split('GHz')[1].strip()
        full_name = name + ' ' + cores
        cpus.append(full_name)
    for score in single_core.find('table', {'class': 'table benchmark-chart-table', 'id': 'mac'}).find_all('td', {
        'class': 'score'}):
        scores.append(int(score.text))
    for i in range(0, len(cpus)):
        cpu_scores[cpus[i]] = scores[i]
    list_of_cpus = X['CPU:'].unique().tolist()
    matches = {}
    matched_cpus = {}
    for cpu_name in list_of_cpus:
        best_match = None
        best_score = 0
        for name in cpu_scores.keys():
            score = fuzz.ratio(name, str(cpu_name))
            if score > best_score:
                best_match = cpu_name
                best_score = score
                best_value = cpu_scores[name]
        matches[cpu_name] = best_match
        matched_cpus[best_match] = best_value
    X['CPU benchmark:'] = X['CPU:'].map(matched_cpus)
    X['CPU benchmark:'].fillna(X['CPU benchmark:'].mean(), inplace=True)
    # grouped_data = X.groupby(['System operacyjny:'])
    # mean_data = grouped_data.mean()
    # X['CPU benchmark:'] = np.where(X['CPU benchmark:'].isnull(),
    #                                         float(mean_data.loc[
    #                                         X[X['CPU benchmark:'].isnull()]['System operacyjny:'], 'CPU benchmark:'].values[0]),
    #                                         X['CPU benchmark:'])
    X.drop('Karta graficzna:', axis=1, inplace=True)
    X.drop('CPU:', axis=1, inplace=True)
    return X, y


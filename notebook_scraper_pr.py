import requests
from bs4 import BeautifulSoup
import csv
import time
from random import uniform
import numpy as np


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
                    cpu = item.find('p', {'ng-if': '!$ctrl.showOnDesktop', 'class': 'wrap-text mt-2 text-xs text-gray-gravel'}).text.split('|')[0].split('LCD')[0]
                  except:
                    cpu = item.find('p', {'ng-if': '!$ctrl.showOnDesktop', 'class': 'wrap-text mt-2 text-xs text-gray-gravel'}).text.split('|')[0]

                except:
                  cpu = np.nan
                notebook_dict['CPU:'] = cpu
                print(notebook_dict['CPU:'])
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
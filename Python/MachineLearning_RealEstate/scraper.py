#!/usr/bin/env python
import re  # tworzenie wyrażeń regularnych
import random  # zwracanie losowej zmiennej
import unicodedata  # dostęp do bazy danych znaków Unicode (UCD)
from bs4 import BeautifulSoup  # wyodrębnianie danych z HTML-a
import requests  # wysyłanie żądań HTTP, zwraca m.in. dane odpowiedzi (treść, kodowanie, status itd.)
import pandas as pd  # przetwarzanie danych
import time  # obsługa zadań związanych z czasem

from numpy.random import randint


def random_user_agent():  # zdefiniowanie jak będziemy widoczni przez serwer
    user_agent_strings = [
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.72 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 "
        "Safari/600.1.25",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",
        "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 "
        "Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.1.17 (KHTML, like Gecko) Version/7.1 "
        "Safari/537.85.10",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
        "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.104 Safari/537.36"
    ]
    return random.choice(user_agent_strings)


def create_data_frame(table):
    df = pd.DataFrame()
    df = df.append(table, ignore_index=False)
    df.rename(columns={0: 'nazwa', 1: 'dzielnica', 2: 'cena', 3: 'metry', 4: 'pietro', 5: 'pokoje', 6: 'rok_budowy',
                       7: 'link', 8: 'data_oferty', 9: 'mieszkanie', 10: 'dom'}, inplace=True)
    return df


def save_data_frame(df):  # Zmienić przed oddaniem
    today_date = time.strftime("%Y_%m_%d_%H:%M")
    df.to_csv(f'data/real_estate_data_{today_date}.csv', index=False, encoding='utf-8', mode='a')
    return df


def find_name(data):  # nazwa mieszkania
    name = data.find(id=re.compile(r'tertiary-name_\d'))
    name = name.get_text() if name else ""
    return name


def find_district(data):  # nazwa dzielnicy/lokalizacji
    district = data.find(id=re.compile(r'tertiary-province_\d'))
    district = district.get_text() if district else ""
    district = district.replace("Katowice", "")
    return district


def find_price(data):  # cena
    price = data.find(id=re.compile(r'primary-display_\d'))
    price = price.get_text() if price else ""
    price = re.search(r'\d+\s\d{3}\s\d{3}|\d{3}\s\d{3}', price)  # zmiana 2 pierwszych wyrazen na d+
    price = price.group(0) if price else ""
    price = unicodedata.normalize("NFKD", price)
    price = price.replace(" ", "")
    return price


def find_metres(data):  # metry
    metres = data.find(id=re.compile(r'area-display_\d'))
    metres = metres.get_text() if metres else ""
    metres = re.findall(r'\d+-\d{2}|-|\d+', metres)
    metres = '.'.join(metres)
    return metres


def find_floor(data):  # piętra
    floor = data.find_all(lambda tag: tag.name == 'td')
    floor = floor[0].get_text() if floor else ""
    floor = floor.replace("parter", "0")
    floor = re.search(r'\d+', floor)
    floor = floor[0] if floor else ""
    return floor


def find_rooms(data):
    rooms = data.find_all(lambda tag: tag.name == 'td')
    rooms = rooms[1].get_text() if rooms else ""
    rooms = re.search(r'\d+', rooms)
    rooms = rooms.group() if rooms else ""
    return rooms


def find_year(data):
    year = data.find_all(lambda tag: tag.name == 'td')
    year = year[2].get_text() if year else ""
    year = re.search(r'\d{4}', year)
    year = year.group() if year else ""
    return year


def find_link(data):
    link = data.find(href=True)
    link = link.get('href') if link else ""
    return link


def find_total_number_of_offers(building, page):
    soup = get_soup(building, page)
    offers = soup.find('span', attrs={'id': 'boxOfCounter'})
    try:
        offers = offers.get_text()
    except Exception as exception:
        print(exception)
        print("Serwer zablokował Twoje IP")
        raise

    offers = re.findall(r'\d+|\d{3}\s\d{3}', offers)
    offers = ''.join(offers)
    print(offers)
    offers = int(offers)
    return offers


def get_link(building, page):
    url = 'https://katowice.nieruchomosci-online.pl/szukaj.html?3,' + building + ',sprzedaz,,Katowice:19572&p=' + str(
        page)
    return url


def get_soup(building, page):
    headers = {'User-Agent': random_user_agent()}
    url = get_link(building, page)
    print(url)
    req = requests.get(url, headers=headers)
    soup = BeautifulSoup(req.text, "html.parser")
    return soup


def add_offers_to_table(table, building):
    page = 0
    offers = find_total_number_of_offers(building, page)
    while (True):
        page = page + 1
        soup = get_soup(building, page)
        for data in list(soup.find_all(id=re.compile('tile_\d+'))):
            name = find_name(data)  # Nazwa mieszkania
            district = find_district(data)  # Dzielnica
            price = find_price(data)  # Cena
            metres = find_metres(data)  # Metry
            floor = find_floor(data)  # Piętro
            rooms = find_rooms(data)  # Liczba pokoi
            year = find_year(data)  # Data budowy
            link = find_link(data)  # Link do oferty
            if link == "":
                continue
            date = time.strftime("%Y/%m/%d")  # Data dodania
            flat = 1 if building == 'mieszkanie' else 0  # Rodzaj budynku
            house = 1 if building == 'dom' else 0  # Rodzaj budynku
            table.append([name, district, price, metres, floor, rooms, year, link, date, flat, house])
            offers = offers - 1
            if offers <= 0:
                break
        # time.sleep(3)
        time.sleep(randint(2, 6))
        if offers <= 0:
            break
        print(offers)
    return table


def main():
    table = []
    add_offers_to_table(table, building='mieszkanie')
    add_offers_to_table(table, building='dom')
    df = create_data_frame(table)
    save_data_frame(df)


if __name__ == '__main__':
    main()

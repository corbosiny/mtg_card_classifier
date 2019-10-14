#!/usr/bin/env python3
import csv, os, sys, time

DEFAULT_CARD_SEARCH = ['Sphinx\'s Insight', 'Warrior', 'Worm', 'Artful Takedown', 'Barrier of Bones', 'Assassin\'s Trophy']
USE_API = True

if USE_API:
    import requests as rq
    import json
else:
    import xlrd
    CATALOG_NAME                = '../data/Magic_Card_Catalog.xlsx'
    PRICE_SHEET_NAME            = 'price_sheet'
    NAME_COLUMN           = 0
    RARITY_COLUMN         = 1
    PRICE_COLUMN          = 2
    SET_COLUMN            = 3

COLLECTION_FILE_NAME        = '../data/Magic_Card_Collection.csv'
FILE_HEADERS          = ['NAME', 'RARITY', 'PRICE', 'SET']


class CardCataloger():

    def __init__(self):
        if not USE_API:
            self.workbook = xlrd.open_workbook(CATALOG_NAME)
            self.priceSheet = self.workbook.sheet_by_name(PRICE_SHEET_NAME)

    def logCards(self, names):
        print('Logging cards..')
        self.cards = []
        for name in names:
            print('Finding card statistics {0}..'.format(name))
            try:
                self.addCardStats(name)
            except Exception as e:
                print('Could not find card information on card {0}'.format(name))
        if not os.path.exists(COLLECTION_FILE_NAME):
            self.initCollectionFile()
            
        self.addCardsToCollection(self.cards)

    def addCardStats(self, name):
        if USE_API:
            card_request = rq.get("https://api.scryfall.com/cards/search?q=%21%22{0}%22".format(name))
            time.sleep(0.05) # Scryfall asks for 50ms between requests
            for ea_card_info in card_request.json()['data']:
                self.cards.append([name, ea_card_info['rarity'], ea_card_info['prices']['usd'], ea_card_info['set']])
                # Maybe rarity should be single letter instead of full word?
            return    
        else:
            for rowIndex in range(self.priceSheet.nrows):
                row = self.priceSheet.row(rowIndex)
                rowName = row[NAME_COLUMN].value.lower()
                if rowName == name.lower():
                    print('Entry for card found..')
                    rarity = row[RARITY_COLUMN].value
                    print('Rarity: {0}'.format(rarity))
                    price = row[PRICE_COLUMN].value
                    print('Price: {0}'.format(price))
                    setOrigin = row[SET_COLUMN].value
                    print('Set: {0}'.format(setOrigin))
                    self.cards.append([name, rarity, price, setOrigin])
                    return

##    def getCardStats(self):
##        pass # Enter code to connect to web server

    def initCollectionFile(self):
        with open(COLLECTION_FILE_NAME, 'w+', newline= '') as file:
            writer = csv.writer(file, delimiter= ',')
            writer.writerow(FILE_HEADERS)

    def addCardsToCollection(self, cards): # TODO: check for duplicates
        with open(COLLECTION_FILE_NAME, 'a', newline = '') as collection:
            writer = csv.writer(collection, delimiter= ',')
            for card in cards:
                writer.writerow(card)


if __name__ == "__main__":
    logger = CardCataloger()
    cards = DEFAULT_CARD_SEARCH
    if len(sys.argv) > 1:
        cards = sys.argv[1:]
    logger.logCards(cards)

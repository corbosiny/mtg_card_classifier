#!/usr/bin/env python3
import csv, os, sys, time

DEFAULT_CARD_SEARCH = ['Sphinx\'s Insight', 'Warrior', 'Worm', 'Artful Takedown', 'Barrier of Bones', 'Assassin\'s Trophy']

import requests as rq
import json

COLLECTION_FILE_NAME        = '../data/collection/Magic_Card_Collection.csv'
IMAGES_DIR_NAME             = '../data/collection/'

CARD_INFO_TO_GATHER   = ['rarity', 'prices']
FILE_HEADERS          = ['NAME', 'RARITY', 'PRICE', 'SET']


class CardCataloger():

    def __init__(self):
        pass

    def logCards(self, names, getImage= False):
        print('Logging cards..')
        cards = []
        for name in names:
            print('Finding card statistics {0}..'.format(name))
            try:
                cardData = self.getCardStats(name, getImage)
                print('Name: {0}, Rarity: {1}, Price: {2}, Set: {3}'.format(name, cardData[1], cardData[2], cardData[3]))
                cards.append(cardData)
            except Exception as e:
                print('Could not find card information on card {0}'.format(name))
        if not os.path.exists(COLLECTION_FILE_NAME):
            self.initCollectionFile()
            
        self.addCardsToCollection(cards)

    def getCardStats(self, name, getImage= False):
        card_request = rq.get("https://api.scryfall.com/cards/search?q=%21%22{0}%22&order=released&unique=art".format(name))
        time.sleep(0.05) # Scryfall asks for 50ms between requests
        try:
            cardData = card_request.json()['data'][0]
            if getImage: self.writeImageToCatalog(cardData)
            cardData = [cardData['name'], cardData['rarity'], cardData['prices']['usd'], cardData['set']]
            return cardData
        except:
            print("Issue finding card by that name online..")
            return None
        
    def writeImageToCatalog(self, cardData):
        card_image_request = rq.get(cardData['image_uris']['normal'], stream=True) # Maybe use small instead of normal if it can handle it?
        if card_image_request.status_code == 200:
            with open(IMAGES_DIR_NAME + cardData['name'] + '.jpg', 'wb') as f:
                for chunk in card_image_request.iter_content(1024):
                    f.write(chunk)


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
    logger.logCards(cards, True)

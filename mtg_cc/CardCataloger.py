import xlrd, csv, os

CATALOG_NAME                = '../data/Magic_Card_Catalog.xlsx'
COLLECTION_FILE_NAME        = '../data/Magic_Card_Collection.csv'
PRICE_SHEET_NAME            = 'price_sheet' 

FILE_HEADERS          = ['NAME', 'RARITY', 'PRICE', 'SET']
NAME_COLUMN           = 0
RARITY_COLUMN         = 1
PRICE_COLUMN          = 2
SET_COLUMN            = 3

class CardCataloger():

    def __init__(self):
        self.workbook = xlrd.open_workbook(CATALOG_NAME)
        self.priceSheet = self.workbook.sheet_by_name(PRICE_SHEET_NAME)
    
    def logCards(self, names):
        print('Logging cards..')
        cards = []
        for name in names:
            print('Finding card statistics {0}..'.format(name))
            try:
                rarity, price, cardSet = self.getCardStats(name)
                cards.append([name, rarity, price, cardSet])
            except Exception as e:
                print('Could not find card information on card {0}'.format(name))
        if not os.path.exists(COLLECTION_FILE_NAME):
            self.initCollectionFile()
            
        self.addCardsToCollection(cards)

    def getCardStats(self, name):
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
                return rarity, price, setOrigin

##    def getCardStats(self):
##        pass # Enter code to connect to web server

    def initCollectionFile(self):
        with open(COLLECTION_FILE_NAME, 'w+', newline= '') as file:
            writer = csv.writer(file, delimiter= ',')
            writer.writerow(FILE_HEADERS)

    def addCardsToCollection(self, cards):
        with open(COLLECTION_FILE_NAME, 'a', newline = '') as collection:
            writer = csv.writer(collection, delimiter= ',')
            for card in cards:
                writer.writerow(card)
                
            
    
if __name__ == "__main__":
    logger = CardCataloger()
    cards = ['Artful Takedown', 'Barrier of Bones', 'Assassin\'s Trophy']
    logger.logCards(cards)

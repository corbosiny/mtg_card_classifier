### Python library imports
import cv2, os, sys
import numpy as np

### Custom library imports
from CardCataloger import CardCataloger


VIDEO_INPUT = 0 # What camera we will pull video input from

### Defines for edge detection parameters 
THRESHOLDING_CUTOFF = 130
ESTIMATION_ACCURACY = .04

### Min and max area values for possible cards detected
CARD_MAX_AREA_THRESHOLD = 75000
CARD_MIN_AREA_THRESHOLD = 18000

### Min and max aspect ratios a cards dimensions could take on
MIN_ASPECT_RATIO = 1.0
MAX_ASPECT_RATIO = 2.3

# Defines for the appearance of the contour boxes
CONTOUR_COLOR     = (0, 255, 0)
CONTOUR_THICKNESS = 2

### Surf algorithm parameters
FLANN_INDEX_KDTREE = 0
INDEX_PARAMS  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
SEARCH_PARAMS = dict(checks = 50)
MIN_MATCH_COUNT = 15   # Need a minimum of 15 key point matches to count as a possible match

### Input command keys
QUIT_KEY    = 'q'
COMPARE_KEY = 'c'
CATALOG_KEY = 'l'


class CardClassifier():

    def __init__(self):
        self.cataloger = CardCataloger()

    def startVideo(self):
        self.cap = cv2.VideoCapture(VIDEO_INPUT)
        while True:
            ret, frame = self.cap.read()
            if ret == False: self.quit()

            cardContours = self.detectCardContours(frame)
            self.drawCardContoursOnFrame(frame, cardContours)
            cv2.imshow('frame', frame)
            
            self.checkForKeyPress()


    def detectCardContours(self, frame):
        contours = self.detectAllContours(frame)
        cardContours = self.extractOnlyCardContours(contours)
        return cardContours

    def detectAllContours(self, frame):
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholdedImg = cv2.threshold(grayImg, THRESHOLDING_CUTOFF, 255, cv2.THRESH_BINARY)            # 255 means anyone above the cutoff is pushed up to be white
        _, contours, heirarchy = cv2.findContours(thresholdedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_TREE returns full family heirarchy of contours, heirarchy could be later used to cutt off internal art or text box contours
                                                                                                           # should test RETR_EXTERNAL here as this only returns outer contours not surrounded by others
                                                                                                           # cv2.CHAIN_APPROX_SIMPLE tells it to only store the verticies in the contour instead of all points, saving memory
        return contours

    def extractOnlyCardContours(self, contours):
        cards = []
        for contour in contours:        
            arcLength = cv2.arcLength(contour, True)                                         # True here and in the next line specifys to only find closed contours
            verticies = cv2.approxPolyDP(contour, ESTIMATION_ACCURACY * arcLength, True)     # Estimation accuracy is how lenient we let the approximator be in finding the bounding rectangle
            if self.isCardContour(contour, verticies):
                cards.append(contour)
        return cards

    def isCardContour(self, contour, verticies):
        startX, startY, width, height = cv2.boundingRect(verticies)
        aspectRatio = width / float(height)
        return (cv2.contourArea(contour) > CARD_MIN_AREA_THRESHOLD and cv2.contourArea(contour) < CARD_MAX_AREA_THRESHOLD and len(verticies) == 4)
    
    def drawCardContoursOnFrame(self, frame, cards):
        cv2.drawContours(frame, cards, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)               # -1 means to draw all card contours
        self.isolatedCardImages = [self.isolateCardImage(cardContour, frame) for cardContour in cards] 
            
    def checkForKeyPress(self):
        keyVal = cv2.waitKey(1)

        if keyVal == ord(QUIT_KEY):
            self.quit()
        elif keyVal == ord(COMPARE_KEY):
            self.compare()
        elif keyVal == ord(CATALOG_KEY):
            self.catalogue()


    def quit(self):
        print('Releasing camera..')
        self.cap.release()
        print('Destroying all windows..')
        cv2.destroyAllWindows()
        print('Exiting..')
        sys.exit('Exiting program')


    def compare(self):
        print('Comparing detected cards to database..')
        matches = []
        for i, elem in enumerate(self.isolatedCardImages):
            image, maxWidth, maxHeight = elem
            matches.append(self.findImageMatch(image, maxWidth, maxHeight))
        print('Finished comparing cards')
        matches = [match for match in matches if match[0] is not None]
        return matches

    def findImageMatch(self, image, maxWidth, maxHeight):
        surf = cv2.xfeatures2d.SURF_create()
        testKP, testDES = surf.detectAndCompute(image, None)
        testImages = [im for im in os.listdir(os.getcwd()) if '.png' in im or '.jpg' in im]
        highestMatches = 0
        bestCard, name = None, None
        bestGood, bestKP, = None, None
        
        for testIMG in testImages:
            img = cv2.imread(testIMG, 0)
            kp, des = surf.detectAndCompute(img, None)
            flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)
            matches = flann.knnMatch(testDES,des,k=2)

            # store all the good matches as per Lowe's ratio test.
            good = [m for m,n in matches if m.distance < 0.7*n.distance]
            if len(good) >= MIN_MATCH_COUNT and len(good) > highestMatches:
                highestMatches = len(good)
                bestCard = img
                name = testIMG.split('.')[0]
                bestKP = kp
                bestGood = good
                print("Enough matches are found in %s - %d/%d" % (testIMG,len(good),MIN_MATCH_COUNT))

        if bestCard is not None: self.displayMatch(name, image, bestCard, testKP, bestKP, bestGood)
        else: print("No good matches")
        
        return name, bestCard

    def displayMatch(self, name, image, bestCard, testKP, kp, good):
        print('Displaying best match for {}..'.format(name))
        src_pts = np.float32([ testKP[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w, _ = image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
        img3 = cv2.drawMatches(image,testKP,bestCard,kp,good,None,**draw_params)
        cv2.imshow('match for {}'.format(name), img3)
    
    def catalogue(self):
        print('Comparing then cataloguing cards..')
        matches = self.compare()
        names = [elem[0] for elem in matches]
        if len(names) == 0:
            print('No recognizable cards to catalogue..')
        else:
            self.cataloger.logCards(names)
            print('Finished writing cards to catalogue')

    def isolateCardImage(self, card, frame):
        # create a min area rectangle from our contour
        _rect = cv2.minAreaRect(card)
        box = cv2.boxPoints(_rect)
        box = np.int0(box)

        # create empty initialized rectangle
        rect = np.zeros((4, 2), dtype = "float32")

        # get top left and bottom right points
        s = box.sum(axis = 1)
        rect[0] = box[np.argmin(s)]
        rect[2] = box[np.argmax(s)]

        # get top right and bottom left points
        diff = np.diff(box, axis = 1)
        rect[1] = box[np.argmin(diff)]
        rect[3] = box[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        frameCopy = frame.copy()
        warped = cv2.warpPerspective(frameCopy, M, (maxWidth, maxHeight))
        return warped, maxWidth, maxHeight
    
if __name__ == "__main__":
    classifier = CardClassifier()
    classifier.startVideo()


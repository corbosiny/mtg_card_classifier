import cv2
import numpy as np

import os, sys

from CardCataloger import CardCataloger

VIDEO_INPUT = 1

SHARPENING_KERNEL = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

SIGMA = .33
DILATION_KERNEL = np.ones((5,5), np.uint8)

CARD_MAX_THRESHOLD = 75000
CARD_MIN_THRESHOLD = 18000

MIN_ASPECT_RATIO = 1.0
MAX_ASPECT_RATIO = 2.3

FLANN_INDEX_KDTREE = 0
INDEX_PARAMS  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
SEARCH_PARAMS = dict(checks = 50)

MIN_MATCH_COUNT = 15

class CardClassifier():

    def __init__(self):
        self.cataloger = CardCataloger()

    def startVideo(self):
        self.cap = cv2.VideoCapture(VIDEO_INPUT)
        while True:
            ret, frame = self.cap.read()
            if ret == False: self.quit()

            cardContours = self.detectCardContours(frame)
            self.drawCardContours(frame, cardContours)
            cv2.imshow('frame', frame)
            
            self.checkForKeyPress()


    def detectCardContours(self, frame):
        contours = self.detectAllContours(frame)
        self.cardContours = self.filterOutCardContours(contours)
        return self.cardContours

    def detectAllContours(self, frame):
        #frame = self.sharpenImage(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #edgedImage = self.findAllEdges(gray)
        #---- apply dilation to help improve edge detection----
        #dilatedImage = cv2.dilate(thresh, DILATION_KERNEL, iterations=1)
        #_, contours, hierarchy = cv2.findContours(dilatedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def sharpenImage(self, img):
        return cv2.filter2D(img, -1, SHARPENING_KERNEL)

    def findAllEdges(self, img):
        imageMedian = np.median(img)
        lowerThreshold = int(max(0, (1.0 - SIGMA) * imageMedian))      
        upperThreshold = int(min(255, (1.0 + SIGMA) * imageMedian))
        #---- apply automatic Canny edge detection using the computed median----
        return cv2.Canny(img, lowerThreshold, upperThreshold)

    def filterOutCardContours(self, contours):
        cards = []
        for contour in contours:
            arcLength = cv2.arcLength(contour, True)
            verticies = cv2.approxPolyDP(contour, 0.04 * arcLength, True)
            x, y, w, h = cv2.boundingRect(verticies)
            ar = w / float(h)
            if cv2.contourArea(contour) > CARD_MIN_THRESHOLD and cv2.contourArea(contour) < CARD_MAX_THRESHOLD and len(verticies) == 4:
                cards.append([contour, ar])
        return cards
    
    def drawCardContours(self, frame, cards):
        self.isolatedCardImages = []
        for card, ar in cards:
            cv2.drawContours(frame, [card], -1, (0, 255, 0), 2)
            self.isolatedCardImages.append(self.isolateCardImage(card, frame, ar))
            
    def checkForKeyPress(self):
        keyVal = cv2.waitKey(1)

        if keyVal == ord('q'):
            self.quit()
        elif keyVal == ord('c'):
            self.compare()
        elif keyVal == ord('l'):
            self.catalogue()


    def quit(self):
        self.cap.release()
        cv2.destroyAllWindows()
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

    def isolateCardImage(self, card, frame, ar):
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


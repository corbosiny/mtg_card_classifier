### Python library imports
import cv2, os, sys
import numpy as np

### Custom library imports
from CardCataloger import CardCataloger


VIDEO_INPUT = 0 # What camera we will pull video input from

### Defines for edge detection parameters 
THRESHOLDING_CUTOFF = 70
ESTIMATION_ACCURACY = .04

### Min and max area values for possible cards detected
CARD_MAX_AREA_THRESHOLD = 75000
CARD_MIN_AREA_THRESHOLD = 19000

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
MIN_MATCH_COUNT = 30   # Need a minimum of 15 key point matches to count as a possible match

### Input command keys
QUIT_KEY           = 'q'
CLEAR_OVERLAY_KEY  = 'o'
COMPARE_KEY        = 'c'
CATALOG_KEY        = 'g'
CLEAR_WINDOW_KEY   = 'l'

### Text Drawing Parameters 
FONT = cv2.FONT_HERSHEY_SIMPLEX 
FONTSCALE = .5
COLOR = (0, 0, 0) 
THICKNESS = 2
LINE_STYLE = cv2.LINE_AA

class textObject():

    def __init__(self, origin, lines):
        self.origin = origin
        self.lines = lines
   

class CardClassifier():

    def __init__(self):
        self.cataloger = CardCataloger()
        self.openWindows = []
        self.textToWrite = []

    def startVideo(self):
        self.cap = cv2.VideoCapture(VIDEO_INPUT)
        while True:
            ret, frame = self.cap.read()
            if ret == False: self.quit()

            cardContours = self.detectCardContours(frame)
            self.drawCardContoursOnFrame(frame, cardContours)
            self.putTextOnFrame(frame)
            cv2.imshow('frame', frame)
            
            self.checkForKeyPress()


    def detectCardContours(self, frame):
        contours = self.detectAllContours(frame)
        cardContours = self.extractOnlyCardContours(contours)
        return cardContours

    def detectAllContours(self, frame):
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        edgesImg = cv2.Canny(grayImg, 100, 200)
        edgesFilteredImg = cv2.dilate(edgesImg, kernel)
        edgesFilteredImg = cv2.morphologyEx(edgesFilteredImg, cv2.MORPH_OPEN, kernel, iterations= 1)
        edgesFilteredImg = cv2.morphologyEx(edgesFilteredImg, cv2.MORPH_CLOSE, kernel, iterations= 1)
        _, contours, heirarchy = cv2.findContours(edgesFilteredImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # RETR_TREE returns full family heirarchy of contours, heirarchy could be later used to cut off internal art or text box contours
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
        return (cv2.contourArea(contour) > CARD_MIN_AREA_THRESHOLD and cv2.contourArea(contour) < CARD_MAX_AREA_THRESHOLD)
    
    def drawCardContoursOnFrame(self, frame, cards):
        cv2.drawContours(frame, cards, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)                 # -1 means to draw all card contours
        self.isolatedCardImages = [self.isolateCardImage(cardContour, frame) for cardContour in cards] 

    def putTextOnFrame(self, frame):
        for textObject in self.textToWrite:
            for i, line in enumerate(textObject.lines):
                x, y = textObject.origin[0], textObject.origin[1] + (i + 1) * 20
                cv2.putText(frame, line, (x, y), FONT, FONTSCALE, COLOR, THICKNESS, LINE_STYLE)
        
    def checkForKeyPress(self):
        keyVal = cv2.waitKey(1)

        if keyVal == ord(QUIT_KEY):
            self.quit()
        elif keyVal == ord(CLEAR_OVERLAY_KEY):
            self.textToWrite = []
        elif keyVal == ord(COMPARE_KEY):
            self.clearAllExtraWindows()
            self.compare(True)
        elif keyVal == ord(CATALOG_KEY):
            self.clearAllExtraWindows()
            self.textToWrite = []
            self.catalogue(False)
        elif keyVal == ord(CLEAR_WINDOW_KEY):
            self.clearAllExtraWindows()


    def quit(self):
        print('Releasing camera..')
        self.cap.release()
        print('Destroying all windows..')
        cv2.destroyAllWindows()
        print('Exiting..')
        sys.exit('Exiting program')

    def clearAllExtraWindows(self):
        for window in self.openWindows:
            cv2.destroyWindow(window)

    def compare(self, showMatches):
        print('Comparing detected cards to database..')
        matches = []
        for i, elem in enumerate(self.isolatedCardImages):
            image, origin, maxWidth, maxHeight = elem
            name, cardImage = self.findImageMatch(image, maxWidth, maxHeight, showMatches)
            matches.append([name, cardImage, origin])
        print('Finished comparing cards')
        matches = [match for match in matches if match[0] is not None]
        return matches
            
    def findImageMatch(self, capturedImage, maxWidth, maxHeight, showPic):
        surf = cv2.xfeatures2d.SURF_create()
        flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)
        testKeyPoints, testDescriptions = surf.detectAndCompute(capturedImage, None)
        testImages = [im for im in os.listdir(os.getcwd()+'/../data/img') if '.png' in im or '.jpg' in im] # Should be able to ignore the filetype
        bestCardImage, cardName, bestKeyPoints, bestSetofMatches = None, None, None, []
        
        for testIMG in testImages:
            img = cv2.imread(testIMG, 0)
            keyPoints, descriptions = surf.detectAndCompute(img, None)
            matches = flann.knnMatch(testDescriptions, descriptions,k=2)
            # store all the good matches as per Lowe's ratio test.
            goodMatches = [m for m,n in matches if m.distance < 0.7*n.distance]

            if len(goodMatches) >= MIN_MATCH_COUNT and len(goodMatches) > len(bestSetofMatches):
                bestCardImage = img
                cardName = testIMG.split('.')[0]
                bestKeyPoints = keyPoints
                bestSetofMatches = goodMatches
                if showPic: print("Enough matches are found in %s - %d/%d" % (testIMG, len(goodMatches), MIN_MATCH_COUNT))

        if bestCardImage is not None and showPic: self.displayMatch(cardName, capturedImage, bestCardImage, testKeyPoints, bestKeyPoints, bestSetofMatches)
        elif showPic: print("No good matches")
        
        return cardName, bestCardImage

    def displayMatch(self, nameOfCard, capturedImage, bestCardImage, testKeyPoints, keyPoints, goodMatches):
        print('Displaying best match for {}..'.format(nameOfCard))
        src_pts = np.float32([ testKeyPoints[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keyPoints[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        draw_params = dict(matchColor= (0,255,0), singlePointColor= None, matchesMask= matchesMask, flags= 2)
        imgOfMatches = cv2.drawMatches(capturedImage, testKeyPoints, bestCardImage, keyPoints, goodMatches, None, **draw_params)
        cv2.imshow('match for {}'.format(nameOfCard), imgOfMatches)
        self.openWindows.append('match for {}'.format(nameOfCard))
    
    def catalogue(self, showPic):
        print('Comparing then cataloguing cards..')
        matches = self.compare(showPic)
        names = [elem[0] for elem in matches]
        print(names)
        if len(names) == 0:
            print('No recognizable cards to catalogue..')
        else:
            for match in matches:
                name, origin = match[0], match[2]
                print(origin)
                card_info = self.cataloger.getCardStats(name)[0]
                text = textObject(origin, ["Name: {}".format(card_info[0]), "Rarity: {}".format(card_info[1]), "Price: {}".format(card_info[2]), "Set: {}".format(card_info[3])])
                self.textToWrite.append(text)
            self.cataloger.logCards(names)
            print('Finished writing cards to catalogue')

    def isolateCardImage(self, cardContour, frame):
        # create a min area rectangle from our contour
        _rect = cv2.minAreaRect(cardContour)
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

        transformationMatrix = cv2.getPerspectiveTransform(rect, dst)
        frameCopy = frame.copy()
        warpedImg = cv2.warpPerspective(frameCopy, transformationMatrix, (maxWidth, maxHeight))
        return warpedImg, [int(num) for num in rect[0]], maxWidth, maxHeight
    
if __name__ == "__main__":
    classifier = CardClassifier()
    classifier.startVideo()


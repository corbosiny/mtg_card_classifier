import csv, os, sys, time
import cv2
import numpy as np
import requests as rq
import json

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

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

### Text Drawing Parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = .5
COLOR = (0, 0, 0)
THICKNESS = 2
LINE_STYLE = cv2.LINE_AA

class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start(0)

    def timerEvent(self):
        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class CardDetectionWidget(QtWidgets.QWidget):
    def __init__(self, cascade_filepath, parent=None):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier(cascade_filepath)
        self.image = QImage()
        self._border = (0, 255, 0)
        self._width = 2
        self.openWindows = []
        self.textToWrite = []

    def image_data_slot(self, image_data):
        if (self.width() > self.height()) != (image_data.shape[1] > image_data.shape[0]):
            # Need to rotate image data, the screen / camera is rotated
            image_data = cv2.rotate(image_data, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cardContours = self.detectCardContours(image_data)
        self.drawCardContoursOnFrame(image_data, cardContours)
        self.putTextOnFrame(image_data)

        self.image = self.get_qimage(image_data)
        self.update()

    def detectCardContours(self, frame):
        contours = self.detectAllContours(frame)
        cardContours = self.extractOnlyCardContours(contours)
        return cardContours

    def putTextOnFrame(self, frame):
        for textObject in self.textToWrite:
            for i, line in enumerate(textObject.lines):
                x, y = textObject.origin[0], textObject.origin[1] + (i + 1) * 20
                cv2.putText(frame, line, (x, y), FONT, FONTSCALE, COLOR, THICKNESS, LINE_STYLE)

    def findImageMatch(self, capturedImage, maxWidth, maxHeight, showPic):
        surf = cv2.xfeatures2d.SURF_create()
        flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)
        testImages = [im for im in os.listdir(os.getcwd()+'/../data/testImages') if '.png' in im or '.jpg' in im] # Should be able to ignore the filetype
        testImages = [os.path.join(os.getcwd()+'/../data/testImages', im) for im in testImages]
        bestCardImage, cardName, bestKeyPoints, bestSetofMatches = None, None, None, []
        testKeyPoints, testDescriptions = surf.detectAndCompute(capturedImage, None)
        for testIMG in testImages:
            img = cv2.imread(testIMG, 0)
            keyPoints, descriptions = surf.detectAndCompute(img, None)
            matches = flann.knnMatch(testDescriptions, descriptions,k=2)
            # store all the good matches as per Lowe's ratio test.
            goodMatches = [m for m,n in matches if m.distance < 0.7*n.distance]

            if len(goodMatches) >= MIN_MATCH_COUNT and len(goodMatches) > len(bestSetofMatches):
                bestCardImage = img
                cardName = testIMG.split('\\')[-1].split('.')[0]
                bestKeyPoints = keyPoints
                bestSetofMatches = goodMatches
                if showPic: print("Enough matches are found in %s - %d/%d" % (testIMG, len(goodMatches), MIN_MATCH_COUNT))

        if bestCardImage is not None and showPic: self.displayMatch(cardName, capturedImage, bestCardImage, testKeyPoints, bestKeyPoints, bestSetofMatches)
        elif showPic: print("No good matches")

        return cardName, bestCardImage

    def compare(self, showMatches):
        print('Comparing detected cards to database..')
        matches = []
        for i, elem in enumerate(self.isolatedCardImages):
            image, origin, maxWidth, maxHeight = elem
            name, cardImage = self.findImageMatch(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), maxWidth, maxHeight, showMatches)
            matches.append([name, cardImage, origin])
        print('Finished comparing cards')
        matches = [match for match in matches if match[0] is not None]
        return matches

    def detectAllContours(self, frame):
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        edgesImg = cv2.Canny(grayImg, 100, 200)
        edgesFilteredImg = cv2.dilate(edgesImg, kernel)
        edgesFilteredImg = cv2.morphologyEx(edgesFilteredImg, cv2.MORPH_OPEN, kernel, iterations= 1)
        edgesFilteredImg = cv2.morphologyEx(edgesFilteredImg, cv2.MORPH_CLOSE, kernel, iterations= 1)
        contours, heirarchy = cv2.findContours(edgesFilteredImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # RETR_TREE returns full family heirarchy of contours, heirarchy could be later used to cut off internal art or text box contours
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

    def get_qimage(self, image):
        height, width, colors = image.shape
        image = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        w = self.width()
        h = self.height()
        cw = self.image.width()
        ch = self.image.height()

        # Keep aspect ratio
        if ch != 0 and cw != 0:
            w = min(cw * h / ch, w)
            h = min(ch * w / cw, h)
            w, h = int(w), int(h)

        painter.drawImage(QtCore.QRect(0, 0, w, h), self.image)
        self.image = QImage()


    def on_click_compare(self):
        print('Compare')

    def on_click_clear(self):
        print('Clear')

    def on_click_catalog(self):
        print('Catalog')

class MainWidget(QtWidgets.QWidget):
    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)
        fp = haarcascade_filepath
        self.card_detection_widget = CardDetectionWidget(fp)

        # 1 is used for frontal camera
        self.record_video = RecordVideo(0)
        self.record_video.image_data.connect(self.card_detection_widget.image_data_slot)

        buttonCompare = QPushButton('Compare', self)
        buttonCompare.setMinimumHeight(100)
        buttonCompare.clicked.connect(self.card_detection_widget.on_click_compare)

        buttonClear = QPushButton('Clear', self)
        buttonClear.setMinimumHeight(100)
        buttonClear.clicked.connect(self.card_detection_widget.on_click_clear)

        buttonCatalog = QPushButton('Catalog', self)
        buttonCatalog.setMinimumHeight(100)
        buttonCatalog.clicked.connect(self.card_detection_widget.on_click_catalog)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.card_detection_widget)
        layout.addWidget(buttonCompare)
        layout.addWidget(buttonClear)
        layout.addWidget(buttonCatalog)
        self.setLayout(layout)


app = QtWidgets.QApplication(sys.argv)
haar_cascade_filepath = cv2.data.haarcascades + '/haarcascade_frontalface_default.xml'
main_window = QtWidgets.QMainWindow()
main_widget = MainWidget(haar_cascade_filepath)
main_window.setCentralWidget(main_widget)
main_window.show()
sys.exit(app.exec_())

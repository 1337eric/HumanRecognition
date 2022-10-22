import tensorflow as tf
import tensorflow_hub
import numpy
import pyautogui
import win32gui
import cv2
import time

detector = tensorflow_hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1")

while True:
    hwnd = win32gui.FindWindow(None, 'Movies & TV') 
    
    rectangle = win32gui.GetWindowRect(hwnd)
    rectangleRegion = rectangle[0], rectangle[1], rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]

    originalScreenShot = numpy.array(pyautogui.screenshot(region=rectangleRegion))
    thisImage = numpy.expand_dims(originalScreenShot, 0)
    imageWidth, imageHeight = thisImage.shape[2], thisImage.shape[1]

    thisImageDetections = detector(thisImage)
    thisImageDetections = {key:value.numpy() for key,value in thisImageDetections.items()}
    detectionBoxes = thisImageDetections['detection_boxes'][0]
    detectionScores = thisImageDetections['detection_scores'][0]
    detectionClasses = thisImageDetections['detection_classes'][0]

    detected_boxes = []
    for index, boundingBox in enumerate(detectionBoxes):
        if detectionClasses[index] == 1 and detectionScores[index] >= 0.40:
            minimumY, minimumX, maximumY, maximumX = tuple(boundingBox)
            boundingLeft, boundingRight, boundingTop, boundingBottom = int(minimumX * imageWidth), int(maximumX * imageWidth), int(minimumY * imageHeight), int(maximumY * imageHeight)
            detected_boxes.append((boundingLeft, boundingRight, boundingTop, boundingBottom))
            cv2.rectangle(originalScreenShot, (boundingLeft, boundingTop), (boundingRight, boundingBottom), (255, 0, 0), 1)

    originalScreenShot = cv2.cvtColor(originalScreenShot, cv2.COLOR_BGR2RGB)
    cv2.imshow("Artificial Intellegence Vision", originalScreenShot)
    cv2.waitKey(1)

    time.sleep(0.05)

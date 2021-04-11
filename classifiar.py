import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps 

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
xtrainScale = xtrain / 255.0
xtestScale = xtest / 255.0
classifier = LogisticRegression(solver='saga', multi_class='multinomial').fit(xtrainScale, ytrain)

def getimage(image):
    imagepil = Image.open(img)
    imagevw = imagepil.convert('L')
    imagevwvsized = imagevw.resize((28, 28), Image.ANTIALIAS)
    pixelprinter = 20
    minimumpixel = np.percentile(imagevwsized, pixelprinters)
    imagescale = np.clip(imagevwsized - minimumpixel, 0, 255)
    maximumpixel = np.max(imagevwsized)
    imagescale = np.asarray(imagescale) / maximumpixel
    testsample = np.array(imagescale).reshape(1, 784)
    testprediction = classifier.predict(testsample)
    return testprediction[0]
    

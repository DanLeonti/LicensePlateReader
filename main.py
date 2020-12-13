import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\Tesseract.exe'

frameWidth = 800
frameHeight = 600
nPlateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
minArea = 300
color = (255,0)
count = 0
path_for_license_plates = os.getcwd() + "Resources/Scanned/NoPlate_0.jpg"
list_license_plates = []
predicted_license_plates = []

cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
contrast = 1.25

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    for(x, y, w, h) in numberPlates:
        area = w*h
        if area >minArea:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI", imgRoi)
            cv2.imshow("Result", img)
            cv2.imwrite("Resources/Scanned/NoPlate_" + str(count) + ".jpg", imgRoi)
            cv2.imshow("Result", img)
            cv2.waitKey(500)
            count += 1

            # Grayscale, Gaussian blur, Otsu's threshold
            gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(gray, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 1)

            # Morph open to remove noise and invert image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            invert = 255 - opening

            # Perform text extraction
            data = pytesseract.image_to_string(invert, lang='eng',
                                               config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
            if len(data) > 7:
                print(data)

            cv2.imshow('thresh', thresh)
            cv2.imshow('opening', opening)
            cv2.imshow('invert', invert)
            ##cv2.waitKey()




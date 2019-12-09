import numpy as np
import imutils
import cv2
from PIL import ImageGrab

def detect_square(c):
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)

	if len(approx) == 4:
		(x, y, w, h) = cv2.boundingRect(approx)
		ar = w / float(h)
		if(ar >= 0.95 and ar <= 1.05):
			return True


fourcc = cv2.VideoWriter_fourcc('X','V','I','D') #you can use other codecs as well.
vid = cv2.VideoWriter('record.avi', fourcc, 8, (640,480))
while(True):
    img = ImageGrab.grab(bbox=(0, 20, 640, 500)) #x, y, w, h
    img_np = np.array(img)
    #frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if(detect_square(c) == True):
        	# compute the center of the contour
        	M = cv2.moments(c)
        	cX = int(M["m10"] / M["m00"])
        	cY = int(M["m01"] / M["m00"])
        	# draw the contour and center of the shape on the image
        	cv2.drawContours(img_np, [c], -1, (0, 255, 0), 2)
        	vid.write(img_np)
        cv2.imshow("frame", img_np)

    key = cv2.waitKey(1)
    if key == 27:
        break    

vid.release()
cv2.destroyAllWindows()
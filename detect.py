import cv2
from image_grab import image_grab
import numpy

food = cv2.imread('food.png')
w_food = food.shape[::-1]
h_food = food.shape[::-1]
while(True):
	scr = image_grab()
	
	res_food = cv2.matchTemplate(scr, food, cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc_food = np.where(res_food >= threshold)

	for pt in zip(*loc_food[::-1]):        
		cv2.rectangle(scr, pt, (pt[0] + w_food, pt[1] + h_food + 9), (50,205,50), 1)
		foodX = pt[0] + w_food
		foodH = pt[1]
        
import cv2 as cv
import numpy as np

img = cv.imread('data/img4.png')
'''
    adım 1 giren resmi kontürle ve rengini köşesine yaz
'''


#while(1):
color = "undefined"
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) #HSV color space geçiş

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)

edges = cv.Canny(blur, 30, 150)

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
res_blue = cv.bitwise_and(img, img, mask = mask_blue)

lower_green = np.array([25, 52, 72])
upper_green = np.array([102, 255, 255])
mask_green = cv.inRange(hsv, lower_green, upper_green)
res_green = cv.bitwise_and(img, img, mask = mask_green)

## Gen lower mask (0-5) and upper mask (175-180) of RED
mask1 = cv.inRange(hsv, (0,50,20), (10,255,255))
mask2 = cv.inRange(hsv, (170,50,20), (180,255,255))

## Merge the mask and crop the red regions
mask_red = cv.bitwise_or(mask1, mask2 )
res_red = cv.bitwise_and(img, img, mask = mask_red)

mask = cv.bitwise_or(mask_blue, mask_green, mask_red) #şimdilik dursun burda kontrol edebiliriz

contours, hierarchy = cv.findContours(mask_red, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for pic, contour in enumerate(contours): 
    area = cv.contourArea(contour) 
    if(area > 300): 
        x, y, w, h = cv.boundingRect(contour) 
        color = "red"
        '''img = cv.rectangle(img, (x, y), 
								(x + w, y + h), 
								(0, 0, 255), 2) 
        
        cv.putText(img, "Red Colour", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))'''

contours, hierarchy = cv.findContours(mask_green, 
									cv.RETR_TREE, 
									cv.CHAIN_APPROX_SIMPLE)

for pic, contour in enumerate(contours): 
    area = cv.contourArea(contour) 
    if(area > 300): 
        x, y, w, h = cv.boundingRect(contour) 

        color = "green"
        '''img = cv.rectangle(img, (x, y), 
								(x + w, y + h), 
								(0, 255, 0), 2) 

        
        cv.putText(img, "Green Colour", (x, y), 
					cv.FONT_HERSHEY_SIMPLEX, 
					1.0, (0, 255, 0))  '''

contours, hierarchy = cv.findContours(mask_blue, 
										cv.RETR_TREE, 
										cv.CHAIN_APPROX_SIMPLE)

for pic, contour in enumerate(contours): 
    area = cv.contourArea(contour) 
    if(area > 300): 
        x, y, w, h = cv.boundingRect(contour) 
        color = "blue"
        '''img = cv.rectangle(img, (x, y), 
								(x + w, y + h), 
								(255, 0, 0), 2) 

        
        cv.putText(img, "Blue Colour", (x, y), 
					cv.FONT_HERSHEY_SIMPLEX, 
					1.0, (255, 0, 0)) '''
    
contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    epsilon = 0.04 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif vertices == 5:
        shape = "Pentagon"
    elif vertices == 6:
        shape = "Altıgen"
    else:
        shape = "Circle"
    
    cv.drawContours(img, [approx], 0, (0, 255, 0), 2)
    cv.putText(img, shape + "\n" + color, (approx[0][0][0], approx[0][0][1]+5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(img, [i], -1, (0, 255, 0), 2)
        cv.circle(img, (cx, cy), 7, (0, 0, 255), -1)
        cv.putText(img, "center", (cx - 20, cy - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    print(f"x: {cx} y: {cy}")

cv.imwrite('outputs/DetectedShapes.png',img)
cv.imshow('image',img)
'''cv.imshow('mask',mask_green)
    cv.imshow('res',res)

k = cv.waitKey(5) & 0xFF
if k == 27:
    break'''

cv.destroyAllWindows()
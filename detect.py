import cv2 as cv
import numpy as np

img = cv.imread('data/img6.png')

#while(1):
color = "undefined"
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

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

masks = {"red" : mask_red, "green" : mask_green, "blue" : mask_blue}
for color, m in masks.items():
    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv.contourArea(contour) < 300: continue
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

        M = cv.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.drawContours(img, [contour], -1, (0, 255, 0), 2)
            cv.circle(img, (cx, cy), 7, (0, 0, 255), -1)
            cv.putText(img, "center", (cx - 20, cy - 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        print(f"x: {cx} y: {cy}")

        cv.drawContours(img, [approx], 0, (0, 255, 0), 2)
        cv.putText(img, shape + "\n" + color, (approx[0][0][0], approx[0][0][1]+5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


cv.imwrite('outputs/DetectedShapes.png',img)
cv.imshow('image',img)
'''cv.imshow('mask',mask_green)
    cv.imshow('res',res)

k = cv.waitKey(5) & 0xFF
if k == 27:
    break'''

cv.destroyAllWindows()
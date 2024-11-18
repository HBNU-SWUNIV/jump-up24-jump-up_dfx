import numpy as np
import cv2

img = cv2.imread('partially.jpg')
img2 = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)

for line in lines:
    rho, theta = line[0]
    cos, sin = np.cos(theta), np.sin(theta)
    cx, cy = rho * cos, rho * sin
    x1, y1 = int(cx + 1000 * (-sin)), int(cy + 1000 * cos)
    x2, y2 = int(cx + 1000 * sin), int(cy + 1000 * (-cos))
    cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('img', img2)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.LineSegmentDetector.detect(img, )
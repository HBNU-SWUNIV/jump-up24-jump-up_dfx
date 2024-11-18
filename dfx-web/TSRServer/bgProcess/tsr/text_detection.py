import cv2
import numpy as np


class TextDetection:
    @staticmethod
    def detect(image: np.ndarray):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 커널을 조정 필요
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy

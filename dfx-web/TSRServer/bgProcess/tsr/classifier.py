import cv2
import numpy as np
from utils import sort_contours, cv2_imshow


class Classifier:
    """
    Table Detection -> Table( point )

    classes : bordered, vertical, horizontal, partially
    """
    def __init__(self):
        pass

    def classify(self, original_image: np.ndarray):
        ver_trigger = True
        hor_trigger = True
        partially_trigger = True
        ver_counter = []
        hor_counter = []

        img_height, img_width, c = original_image.shape
        # print("height : ", img_height, ", width : ", img_width)
        # print("width / 0.15 : ", img_width * 0.85)

        img_v_thresh = int(img_width * 0.85)
        img_h_thresh = int(img_height * 0.85)

        img_vh = self.pre_process(original_image)
        cv2.imwrite("output/img_vh.png", img_vh)
        img_vh_transpose = img_vh.T

        # 가로선 검사
        for i in range(img_height):
            h_line_sum = int(sum(img_vh[i]) / 128)
            if h_line_sum > img_v_thresh:
                hor_trigger = False

        print("hor_trigger : ", hor_trigger)

        # 세로선 검사
        for i in range(img_height):
            v_line_sum = int(sum(img_vh_transpose[i]) / 128)
            if v_line_sum > img_h_thresh:
                ver_trigger = False
        print("ver_trigger : ", ver_trigger)

        result = {
            "vertical": ver_trigger,
            "horizontal": hor_trigger,
            # "partially": partially_trigger,
        }

        return result

    def pre_process(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_height, img_width = img.shape

        thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

        img_bin_inv = 255 - img_bin

        kernel_len_ver = max(10, img_height // 50)
        kernel_len_hor = max(10, img_width // 50)
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (1, kernel_len_ver))  # shape (kernel_len, 1) inverted! xD

        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))  # shape (1,kernel_ken) xD

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Use vertical kernel to detect and save the vertical lines in a jpg
        image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
        # Plot the generated image

        # Use horizontal kernel to detect and save the horizontal lines in a jpg
        image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)

        # Combine horizontal and vertical lines in a new third image, with both having same weight.
        img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        return img_vh


if __name__ == '__main__':
    classifier = Classifier()
    image = cv2.imread('test/00_partially_v.png')
    print(classifier.classify(image))

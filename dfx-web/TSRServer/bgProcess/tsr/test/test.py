import cv2
import numpy as np
from matplotlib import pyplot as plt
from util import sort_contours


def recognize_structure(img, idx, is_logs=False):
    cnt = 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    img_height, img_width = img.shape
    if is_logs:
        cv2.imshow("img_bin", img)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_original.png", img)
        cnt += 1
    thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    if is_logs:
        cv2.imshow("img_bin", img_bin)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_bin.png", img_bin)
        cnt += 1

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    invert = False
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height and (
                w > max(10, img_width / 30) and h > max(10, img_height / 30))):
            invert = True
            img_bin[y:y + h, x:x + w] = 255 - img_bin[y:y + h, x:x + w]
    if is_logs:
        cv2.imshow("img_bin", img_bin)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_bin_inv.png", img_bin)
        cnt += 1

    img_bin = 255 - img_bin if invert else img_bin
    if is_logs:
        cv2.imshow("img_bin", img_bin)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_bin_inv.png", img_bin)
        cnt += 1

    img_bin_inv = 255 - img_bin
    if is_logs:
        cv2.imshow("img_bin_inv", img_bin_inv)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_bin_inv.png", img_bin_inv)
        cnt += 1

    kernel_len_ver = max(10, img_height // 50)
    kernel_len_hor = max(10, img_width // 50)

    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    if is_logs:
        cv2.imshow("image_1", image_1)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img1.png", image_1)
        cnt += 1
    if is_logs:
        cv2.imshow("vertical_lines", vertical_lines)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_vertical_lines.png", vertical_lines)
        cnt += 1

    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)
    if is_logs:
        cv2.imshow("image_2", image_2)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img2.png", image_2)
        cnt += 1
    if is_logs:
        cv2.imshow("horizontal_lines", horizontal_lines)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_horizontal_lines.png", horizontal_lines)
        cnt += 1

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    if is_logs:
        cv2.imshow("img_vh", img_vh)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_vh.png", img_vh)
        cnt += 1

    img_vh = cv2.dilate(img_vh, kernel, iterations=3)
    if is_logs:
        cv2.imshow("img_vh", img_vh)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_vh_dilate.png", img_vh)
        cnt += 1

    thresh, img_vh = (cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY))
    if is_logs:
        cv2.imshow("img_vh", img_vh)
        cv2.waitKey(0)
        cv2.imshow("img_bin", img_bin)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_vh_thresh.png", img_vh)
        cnt += 1

    bitor = cv2.bitwise_or(img_bin, img_vh)
    if is_logs:
        cv2.imshow("bitor", bitor)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_bitor.png", bitor)
        cnt += 1

    img_median = bitor
    cv2.imshow("img_median", img_median)

    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, img_height * 2))
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)
    if is_logs:
        cv2.imshow("vertical_lines", vertical_lines)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_vertical_lines.png", vertical_lines)
        cnt += 1

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width * 2, 3))
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    if is_logs:
        cv2.imshow("horizontal_lines", horizontal_lines)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_horizontal_lines.png", horizontal_lines)
        cnt += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    if is_logs:
        cv2.imshow("img_vh", img_vh)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_vh.png", img_vh)
        cnt += 1
    if is_logs:
        cv2.imshow("~img_vh", ~img_vh)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_vh_~.png", ~img_vh)
        cnt += 1

    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    if is_logs:
        cv2.imshow("img_vh", img_vh)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_vh_erode.png", img_vh)
        cnt += 1

    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)
    if is_logs:
        cv2.imshow("img_vh", img_vh)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_vh_thresh.png", img_vh)
        cnt += 1

    bitxor = cv2.bitwise_xor(img_bin, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    if is_logs:
        cv2.imshow("bitnot", bitnot)
        cv2.waitKey(0)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_bitnot.png", bitnot)
        cnt += 1

    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    mean = np.mean(heights)

    box = []
    image = np.array(0)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])
    if is_logs:
        cv2.imshow("image", image)
        cv2.imwrite(f"logs/{idx}_{cnt}_tsr_img_box.png", image)
        cnt += 1

    row = []
    column = []
    j = 0
    previous = []

    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]

                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    countcol = 0
    index = 0
    for i in range(len(row)):
        current = len(row[i])
        # print("len",len(row[i]))
        if current > countcol:
            countcol = current
            index = i

    center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    center = np.array(center)
    center.sort()

    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    return finalboxes, img_bin


original_img = cv2.imread("../images/color/img.png")
finalboxes, output_img = recognize_structure(original_img, 0, True)

cv2.destroyAllWindows()
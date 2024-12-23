import cv2
from matplotlib import pyplot as plt


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def plt_imshow(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def cv2_imshow(image, title=None):
    if title is None:
        title = "image"
    cv2.imshow(title, image)
    cv2.waitKey(0)
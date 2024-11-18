import cv2


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


def cv2_imshow(image, title=None):
    if title is None:
        title = "image"
    cv2.imshow(title, image)
    cv2.waitKey(0)



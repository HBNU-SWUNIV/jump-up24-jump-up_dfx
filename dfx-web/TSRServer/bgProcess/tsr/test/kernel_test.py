import copy
import kernel
import cv2
import numpy as np
from TSR import TSR
from pprint import pprint
from skimage.util import invert
from skimage.morphology import skeletonize, thin

shape_func = {
    "TShape0": TSR.TShape0,
    "TShape90": TSR.TShape90,
    "TShape180": TSR.TShape180,
    "TShape270": TSR.TShape270,
    "LineShape0": TSR.LineShape0,
    "LineShape90": TSR.LineShape90,
    "LineShape180": TSR.LineShape180,
    "LineShape270": TSR.LineShape270,
    "LShape0": TSR.LShape0,
    "LShape90": TSR.LShape90,
    "LShape180": TSR.LShape180,
    "cross": TSR.cross,
    "LShape270": TSR.LShape270
}

kernels = kernel.kernels
print(len(kernels))
original_image = cv2.imread('bordered_ex.png')

# 이미지 전처리
img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
img_height, img_width = img.shape
print(img_height, img_width)


# thresholding the image to a binary image
# thresh, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
# inverting the image
img_bin = 255 - img_bin

kernel_len_ver = img_height // 50
kernel_len_hor = img_width // 50
print(kernel_len_ver, kernel_len_hor)
# Defining a vertical kernel to detect all vertical lines of image
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))  # shape (kernel_len, 1) inverted! xD

# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))  # shape (1,kernel_ken) xD

# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
# Plot the generated image

# Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
# Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=1)
thresh, binary_image = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("padded_image1", binary_image)

thinning_image = cv2.erode(~binary_image, None, iterations=1)

# binary_image = binary_image // 255
# skeleton = skeletonize(binary_image)
# skeleton_image = (skeleton * 255).astype(np.uint8)

# thinning = thin(binary_image)
# thinning_image = (thinning * 255).astype(np.uint8)


padded_image = np.pad(~thinning_image, 100, constant_values=255)
pd_image_height, pd_image_width = padded_image.shape
result_image = copy.deepcopy(padded_image)
result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 165, 255),
          (128, 0, 128), (203, 192, 255), (42, 42, 165), (128, 128, 128), (128, 0, 0), (0, 0, 0)]

match_list = []


for idx, kernel in enumerate(kernels):
    match_image = (~padded_image // 255)
    result = cv2.matchTemplate(match_image, kernel['kernel'], cv2.TM_CCOEFF_NORMED)

    box_loc = np.where(result >= 0.99)
    for box in zip(*box_loc[::-1]):
        startX, startY = box
        endX, endY = startX + 6, startY + 6
        cv2.rectangle(result_image, (startX, startY), (endX, endY), colors[idx % len(colors)], 1)
        cv2.rectangle(result_image, (startX + 3, startY + 3), (startX + 3, startY + 3), colors[idx % len(colors)], 1)
        cv2.putText(result_image, "shape:{}".format(kernel["shape"]), (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    colors[idx % len(colors)])
        match_list.append({"point": (startX, startY), "class": kernel})

cv2.imshow("padded_image", result_image)
cv2.imwrite("result_image.png", result_image)
new_image = np.zeros((pd_image_height, pd_image_width), dtype=np.uint8)

print(new_image)
for match in match_list:
    shape_func[match["class"]["shape"]]()



# cv2.imwrite("result_image.png", result_image)

cv2.imshow("new_image", new_image)

# bitxor = cv2.bitwise_xor(img, img_vh)
# bitnot = cv2.bitwise_not(bitxor)
# Plotting the generated image
cv2.waitKey(0)
cv2.destroyAllWindows()


import numpy as np
import cv2


class LineGenerator:
    def vertical_lines(self, image: np.ndarray):
        h, w, c = image.shape
        verticals = []
        temp_y = -1
        temp_x = -1
        temp_idx = -1

        bboxs = sorted(self.text_detection(image), key=lambda x: x["x"])

        for x in range(w):
            for idx, bbox in enumerate(bboxs):
                if bbox["x"] == x and temp_y == -1:
                    temp_y = bbox["y"]
                    temp_x = x
                    temp_idx = idx
                    break

                if temp_y != -1 and (bboxs[temp_idx]["w"] + temp_x) == x:
                    verticals.append({"start_x": temp_x, "end_x": x})
                    temp_y = -1
                    temp_x = -1
                    temp_idx = -1
                    break

        # result
        for idx, vertical in enumerate(verticals):
            if idx == 0:
                cv2.line(image, (vertical["start_x"] - 5, 0), (vertical["start_x"] - 5, h), (0, 0, 0))
            if idx == len(verticals) - 1:
                cv2.line(image, (vertical["end_x"] + 5, 0), (vertical["end_x"] + 5, h), (0, 0, 0))
            if idx + 1 < len(verticals):
                middle_x = (verticals[idx + 1]["start_x"] + verticals[idx]["end_x"]) // 2
                cv2.line(image, (middle_x, 0), (middle_x, h), (0, 0, 0))

        return image

    def horizontal_lines(self, image: np.ndarray):
        h, w, c = image.shape
        horizontals = []
        temp_y = -1
        temp_x = -1
        temp_idx = -1

        bboxs = sorted(self.text_detection(image), key=lambda x: x["y"])

        for y in range(h):
            for idx, bbox in enumerate(bboxs):
                if bbox["y"] == y and temp_x == -1:
                    temp_x = bbox["x"]
                    temp_y = y
                    temp_idx = idx
                    break

                if temp_x != -1 and (bboxs[temp_idx]["h"] + temp_y) == y:
                    horizontals.append({"start_y": temp_y, "end_y": y})
                    temp_y = -1
                    temp_x = -1
                    temp_idx = -1
                    break
        # result
        for idx, horizontal in enumerate(horizontals):
            if idx == 0:
                cv2.line(image, (0, horizontal["start_y"] - 5), (w, horizontal["start_y"] - 5), (0, 0, 0))
            if idx == len(horizontals) - 1:
                cv2.line(image, (0, horizontal["end_y"] + 5), (w, horizontal["end_y"] + 5), (0, 0, 0))
            if idx + 1 < len(horizontals):
                middle_y = (horizontals[idx + 1]["start_y"] + horizontals[idx]["end_y"]) // 2
                cv2.line(image, (0, middle_y), (w, middle_y), (0, 0, 0))

        return image

    def bbox_filter(self, contours, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        bboxs = []
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask[y:y+h, x:x+w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
            if r > 0.45 and w > 8 and h > 8:
                bboxs.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

        return bboxs

    def text_detection(self, image: np.ndarray):
        small = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        # using RETR_EXTERNAL instead of RETR_CCOMP
        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return self.bbox_filter(contours, bw.shape)



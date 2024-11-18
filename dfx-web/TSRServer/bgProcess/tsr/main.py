import os
import sqlite3

from text_detection import TextDetection
from line_generator import LineGenerator
from table_extractor import TableExtractor
from classifier import Classifier
from table import Table
from TSR import TSR
from ultralyticsplus import render_result

import kernel
import cv2
import config
import time
import zipfile


extractor = TableExtractor()
classifier = Classifier()
line_generator = LineGenerator()

kernels = kernel.kernels


def table_detection(img_path: str):
    cnt = 0
    dir_path = img_path.split("/")
    file_name = dir_path.pop()
    dir_path = "/".join(dir_path)
    os.makedirs(f"{dir_path}/processing/tables", exist_ok=True)

    result = extractor.extract_table(img_path)
    classes = result[0].boxes.cls
    render = render_result(model=extractor.model, image=img_path, result=result[0])
    original = cv2.imread(img_path)
    for idx, class_ in enumerate(classes):
        point = {
            "x1": int(result[0].boxes[idx].xyxy.tolist()[0][0]),
            "y1": int(result[0].boxes[idx].xyxy.tolist()[0][1]),
            "x2": int(result[0].boxes[idx].xyxy.tolist()[0][2]),
            "y2": int(result[0].boxes[idx].xyxy.tolist()[0][3])
        }
        crop_dict = {
            "x": point["x1"],
            "xw": point["x1"] + (point["x2"]-point["x1"])
        }

        if point["y1"] - 25 > 0:
            crop_dict["y"] = point["y1"]
        else:
            crop_dict["y"] = point["y1"]
        if point["y1"] + (point["y2"]-point["y1"]) + 25 <= original.shape[0]:
            crop_dict["yh"] = point["y1"] + (point["y2"]-point["y1"])
        else:
            crop_dict["yh"] = point["y1"] + (point["y2"]-point["y1"])

        crop = original[crop_dict["y"]:crop_dict["yh"], crop_dict["x"]:crop_dict["xw"]]
        cv2.imwrite(f"{dir_path}/processing/tables/{idx}_{file_name}", crop)
        cnt += 1

    return cnt, dir_path, file_name


if __name__ == "__main__":
    while True:
        conn = sqlite3.connect("../../db.sqlite3")
        cur = conn.cursor()
        cur.execute("SELECT * FROM TSR_tsr WHERE is_complete == 0;")
        result = cur.fetchone()
        if result is not None:
            table_cnt, dir_path, file_name = table_detection(result[4])
            os.makedirs(f"{dir_path}/excels", exist_ok=True)
            save_path = f"{dir_path}/excels"
            for i in range(table_cnt):
                img_path = f"{dir_path}/processing/tables/{i}_{file_name}"
                print(img_path)
                tsr = TSR(img_path, kernels, save_path=save_path, table_cnt=i)
                tsr.run()
                del tsr

            owd = os.getcwd()
            os.chdir(f"{dir_path}")

            zip_file = zipfile.ZipFile(f"{file_name}.zip", "w")
            for file in os.listdir(f"./excels"):
                print(file)
                zip_file.write(os.path.join("./excels", file), compress_type=zipfile.ZIP_DEFLATED)

            zip_file.close()
            os.chdir(owd)

            cur.execute("UPDATE TSR_tsr SET is_complete = ? WHERE id = ?", (True, result[0]))
            conn.commit()

        time.sleep(1)



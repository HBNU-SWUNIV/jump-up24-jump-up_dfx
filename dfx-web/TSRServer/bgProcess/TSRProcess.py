import time
import sqlite3


extractor = TableExtractor()


def table_detection(path):
    result = extractor.extract_table(path)
    classes = result[0].boxes.cls
    render = render_result(model=extractor.model, image=path, result=result[0])
    # render.show()
    render.save("output/result23.png")
    tables = []
    print(result[0].boxes.xyxy.tolist())
    for idx, class_ in enumerate(classes):
        _class = config.classes[int(class_)]
        point = {
            "x1": int(result[0].boxes[idx].xyxy.tolist()[0][0]),
            "y1": int(result[0].boxes[idx].xyxy.tolist()[0][1]),
            "x2": int(result[0].boxes[idx].xyxy.tolist()[0][2]),
            "y2": int(result[0].boxes[idx].xyxy.tolist()[0][3])
        }
        tables.append(Table(_class, point))
    img = cv2.imread(path)

    # 분류기 사용시 사용하는 코드
    for table in tables:
        # 디버그 용
        # cv2.rectangle(img, (table.point["x1"], table.point["y1"]), (table.point["x2"], table.point["y2"]), (255, 0, 0), 5)
        table_img = img[table.point["y1"]:table.point["y2"], table.point["x1"]:table.point["x2"]]
        result_classifier = classifier.classify(table_img)
        print(result_classifier)
        if not result_classifier["vertical"]:
            result_image = line_generator.horizontal_lines(table_img)
            print("vertical")
            cv2.imshow("result", result_image)
        elif not result_classifier["horizontal"]:
            result_image = line_generator.vertical_lines(table_img)
            print("horizontal")
            cv2.imshow("result", result_image)
            cv2.imwrite("output/result233.png", result_image)
        else:
            print("No class found")

# while True:
con = sqlite3.connect('../db.sqlite3')
cur = con.cursor()
cur.execute('SELECT * FROM TSR_tsr WHERE is_complete == 0;')
result = cur.fetchall()

if len(result) > 0:
    print(1)
# print(cur)
from ultralyticsplus import YOLO


class TableExtractor:
    def __init__(self):
        # load model
        # self.model = YOLO('keremberke/yolov8s-table-extraction')
        self.model = YOLO('best.pt')

        # set model parameters
        self.model.overrides['conf'] = 0.25  # NMS confidence threshold
        self.model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 1000  # maximum number of detections per image

    def extract_table(self, file_path):
        print(f'Extracting table from {file_path}')
        return self.model.predict(file_path)

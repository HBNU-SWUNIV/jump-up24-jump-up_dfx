import os

from TSR import TSR

import kernel
import cv2
import time


kernels = kernel.kernels

if __name__ == "__main__":
    debug_log_path = f"./output/logs/TSR_{time.time()}"
    image_path = "test/bordered_ex.png"
    # image_path_rotated = "./bordered_rotate_ex.png"
    image = cv2.imread(image_path)

    tsr = TSR(image, kernels, debug_log_path=debug_log_path)
    tsr.run()
    del tsr
    # 테이블 추출 모델 사용
    # table_detection(image_path)

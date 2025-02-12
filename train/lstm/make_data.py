
import math
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
import torch
import threading
from utils import *
from yolo.utils.general import non_max_suppression
from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load mô hình ONNX
model_path = "weights/model.onnx"
yolo_weights = "weights/last.onnx"
yolo_session = ort.InferenceSession(yolo_weights, providers=["CPUExecutionProvider"])
onnx_session = ort.InferenceSession(model_path)

def detect_yolo(frame):
    in_frame = in_img(frame)
    input_name = yolo_session.get_inputs()[0].name
    detected = yolo_session.run(None, {input_name: in_frame})[0]
    detected = torch.tensor(detected).unsqueeze(0)
    # print('detected.shape',detected.shape)
    detected = non_max_suppression(detected, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    return detected
def rescale(frame, size, x1, y1, x2, y2):
    h, w, i = frame.shape
    x1 = int((x1/size)*w)
    y1 = int((y1/size)*h)
    x2 = int((x2/size)*w)
    y2 = int((y2/size)*h)
    return x1, y1, x2, y2
no_of_frames = 200
start_frame = 10
frame_count = 0 
label = "normal05"
if __name__ == "__main__":
    folder_path = r"C:\Users\OS\Desktop\Act_recognize\train\Standing"
    output_folder = r"C:\Users\OS\Desktop\Act_recognize\datasets\Standing"

    os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục output nếu chưa có

    video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]

    for i, video_file in enumerate(video_files):
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        lm_list = []
        frame_count = 0
        count = 0
        prev_time = time.time()
        size = 192  # Kích thước resize ảnh
        start_frame = 0
        no_of_frames = 100  # Giới hạn số frame
        label = f"sitting_{i}"  # Lấy tên file làm nhãn

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            h, w, _ = frame.shape
            count += 1
            frame1 = cv2.resize(frame, (size, size))
            detected = detect_yolo(frame1)

            if np.any(detected):
                detections = []
                for detect in detected[0]:
                    x1, y1, x2, y2, conf, cls = detect
                    x1, y1, x2, y2 = rescale(frame, size, x1, y1, x2, y2)
                    detections.append([x1, y1, x2, y2, conf])

                detections = np.array(detections)
                bbox = [x1, y1, x2, y2]
                inp = crop_img(frame, bbox)

                # Dự đoán với mô hình ONNX
                keypoints_with_scores = predict_with_onnx(onnx_session, inp)

                for batch_idx in range(keypoints_with_scores.shape[0]):
                    for keypoint_idx in range(keypoints_with_scores.shape[1]):
                        points_inp = keypoints_with_scores[batch_idx, keypoint_idx, :, :2]
                        points_frame, t1, t2 = point_in_frame(frame, inp, bbox, points_inp)
                        frame = draw_skeleton(frame, points_frame, t1, t2)
                        lm = make_landmark_timestep(points_frame, t1, t2)
                        if len(lm)== 32:
                            lm_list.append(lm)

                # Vẽ bounding box
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if frame_count == no_of_frames:
                break

            # Tính FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            img = draw_class_on_image(f"FPS:{fps:.2f} Frame:{count}", frame, (10, 30), (0, 255, 0), 1)
            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Lưu dữ liệu landmark vào file CSV
        df = pd.DataFrame(lm_list)
        output_path = os.path.join(output_folder, f"{label}.txt")
        df.to_csv(output_path, index=False)

        cap.release()
        cv2.destroyAllWindows()

        print(f"✅ Đã xử lý xong {video_file} và lưu kết quả tại {output_path}")

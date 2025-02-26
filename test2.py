import time
import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch
from tracker.tracker2 import BYTETracker
import threading
from utils import *
from general import  non_max_suppression_modified
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
yolo_weights = "weights/yolov8n-pose.onnx"
yolo_session = ort.InferenceSession(yolo_weights)
input_name = yolo_session.get_inputs()[0].name
output_name = yolo_session.get_outputs()[0].name

#lstm model

# Load LSTM model từ ONNX
lstm_model_path = "weights/model_lstm_multiclass.onnx"
lstm_session = ort.InferenceSession(lstm_model_path)


def act_recognize():
    global label, clr, track_keypoints, track_id
    keypoints_array = np.expand_dims(track_keypoints[track_id], axis=0).astype(np.float32)
    
    input_name = lstm_session.get_inputs()[0].name
    output = lstm_session.run(None, {input_name: keypoints_array})[0]
    
    predicted_class = np.argmax(output)
    classes = ["Normal", "Sitting", "Falling"]
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    
    label = classes[predicted_class]
    clr = colors[predicted_class]
    
    track_labels[track_id] = {'label': label, 'clr': clr}
    track_keypoints.get(track_id, {}).clear()

def detect_yolo(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển về RGB
    frame = frame.astype(np.float32) / 255.0  # Chuẩn hóa về [0, 1]
    # Định dạng lại tensor đầu vào (N, C, H, W)
    input_tensor = np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0).astype(np.float32)
    detected = yolo_session.run(None, {input_name: input_tensor})[0]
    detected = non_max_suppression_modified(detected, conf_thres, iou_thres, max_det=max_det)
    return detected

if __name__ == "__main__":  
    url = r"rtsp://admin:cxview2024@192.168.100.50:5554/live"
    video = r"C:\Users\OS\Desktop\ActionProject\videos\50Way.mp4"
    output_video = "output.mp4"
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False)

    cap = cv2.VideoCapture(video)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)  # Tăng buffer size (nếu hỗ trợ)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Giới hạn FPS để giảm tải
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video, fourcc, 15, (frame_width, frame_height))
    frame_counter = 0  # Biến đếm frame
    prev_time = time.time()

    track_keypoints = {}  # Khởi tạo dictionary để lưu keypoints theo track_id
    track_labels = {}    # Khởi tạo dictionary để lưu nhãn và màu theo track_id

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        
        frame_counter += 1  # Tăng biến đếm frame
        
        # Chỉ xử lý YOLO và nhận diện sau mỗi 2 frame
        # if frame_counter % 3 == 0:
        frame1 = cv2.resize(frame, (size, size))
        detected = detect_yolo(frame1)
        if np.any(detected):
            detections = []
            for detect in detected[0]:
                box_score = detect[0:5]
                keypoints = detect[5:]
                x1, y1, x2, y2 = box_rescale(frame, size, box_score[0], box_score[1], box_score[2], box_score[3])
                keypoints_new = []
                for i in range(0, len(keypoints), 3):
                    kp_x, kp_y, vis = keypoints[i:i+3]
                    x, y = kp_rescale(frame, size, kp_x, kp_y)
                    keypoints_new.extend([x, y])
                detection = [x1, y1, x2, y2, box_score[4]] + keypoints_new
                detections.append(detection)
            detections = np.array(detections)
            tracks = tracker.update(detections)
        else:
            tracks = []

        if tracks:
            for track in tracks:
                track_id = track.track_id
                keypoints = track.keypoints
                x1, y1, x2, y2 = map(int, track.tlbr)
                
                if keypoints is not None:
                    lstm_input = []
                    for i in range(0, len(keypoints), 2):
                        kp_x, kp_y = keypoints[i:i+2]
                        x, y = float(kp_x/w), float(kp_y/h)
                        lstm_input.append([x, y])
                        cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (0, 0, 255), -1)
                    frame = draw_skeleton_2(frame, lstm_input)
                    lm = make_landmark_timestep(lstm_input)
                    if track_id not in track_keypoints:
                        track_keypoints[track_id] = []
                    if len(lm) == 32:
                        track_keypoints[track_id].append(lm)

                # Xử lý nhận diện nếu đủ dữ liệu
                if len(track_keypoints.get(track_id, [])) >= 8:
                    thread2 = threading.Thread(target=act_recognize)
                    thread2.start()
            if track_id not in track_labels:
                    track_labels[track_id] = {'label': "Unknown", 'clr': (255, 255, 255)}
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), track_labels[track_id]['clr'], 2)
            frame = cv2.putText(frame, f"{track_labels[track_id]['label']} id:{track_id}",
                                (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, track_labels[track_id]['clr'], 2)

        # Vẽ kết quả lên frame (dù không xử lý mới, vẫn dùng dữ liệu cũ để hiển thị)
        # if 'tracks' in locals():
        #     for track in tracks:
        #         track_id = track.track_id
        #         x1, y1, x2, y2 = map(int, track.tlbr)
                
                    
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        img = draw_class_on_image(f"FPS:{fps:.2f} Frame:{frame_counter}", frame, (10, 30), (0,255,0), 1)
        # out.write(frame)
        
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

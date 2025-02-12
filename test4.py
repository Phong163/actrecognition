import time
import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch
from tracker.tracker import BYTETracker
import threading
from utils import *
from general import non_max_suppression
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
yolo_weights = r"C:\Users\OS\Desktop\ActionProject\weights\last.onnx"
yolo_session = ort.InferenceSession(yolo_weights)

# Load MoveNet model
model_path = "weights/model.onnx"
onnx_session = ort.InferenceSession(model_path)

# Load LSTM model từ ONNX
lstm_model_path = "weights/model_lstm_multiclass.onnx"
lstm_session = ort.InferenceSession(lstm_model_path)

size_data = 32

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
    in_frame = in_img(frame)
    input_name = yolo_session.get_inputs()[0].name
    detected = yolo_session.run(None, {input_name: in_frame})[0]
    detected = torch.tensor(detected).unsqueeze(0)
    detected = non_max_suppression(detected, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    return detected


if __name__ == "__main__":  
    url = r"rtsp://admin:cxview2024@14.232.244.73:5554/live"
    video = r"C:\Users\OS\Desktop\ActionProject\videos\Typa Girls - BLACKPINK - Ciin x SongLinh choreography.mp4"
    output_video = "output.mp4"
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False)
    cap = cv2.VideoCapture(url)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # # Đối tượng ghi video MP4
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
    # out = cv2.VideoWriter(output_video, fourcc, 15, (frame_width, frame_height))
    count = 0
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
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
            tracks = tracker.update(detections)
        else:
            tracks = []

        if tracks:
            for track in tracks:
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.tlbr)

                bbox = [x1, y1, x2, y2]
                # bbox1 = enlarge_bbox(frame, bbox)
                inp = crop_img(frame, bbox)

                keypoints_with_scores = predict_with_onnx(onnx_session, inp)

                for batch_idx in range(keypoints_with_scores.shape[0]):
                    for keypoint_idx in range(keypoints_with_scores.shape[1]):
                        points_inp = keypoints_with_scores[batch_idx, keypoint_idx, :, :2]
                        
                        points_frame, t1, t2 = point_in_frame(frame, inp, bbox, points_inp)
                        frame = draw_skeleton(frame, points_frame, t1, t2)
                        lm = make_landmark_timestep(points_frame, t1, t2)

                        if track_id not in track_keypoints:
                            track_keypoints[track_id] = []
                        if len(lm) == size_data:
                            track_keypoints[track_id].append(lm)

                if len(track_keypoints[track_id]) == 5:
                    thread2 = threading.Thread(target=act_recognize)
                    thread2.start()

                if track_id not in track_labels:
                    track_labels[track_id] = {'label': "Unknown", 'clr': (255, 255, 255)}

                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), track_labels[track_id]['clr'], 2)
                frame = cv2.putText(frame, f"{track_labels[track_id]['label']} id:{track_id}",
                                    (int(bbox[0]), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, track_labels[track_id]['clr'], 2)
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        img = draw_class_on_image(f"FPS:{fps:.2f} Frame:{count}", frame, (10, 30), (0,255,0), 1)
        # out.write(frame)
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

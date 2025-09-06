import cv2
import torch
import numpy as np
from PIL import Image
import pathlib
import sys
from collections import OrderedDict
import time
import warnings
from deep_sort_realtime.deepsort_tracker import DeepSort

# Uyarıları gizle
warnings.filterwarnings("ignore", category=FutureWarning)

# Windows uyumluluğu için (PosixPath hatasına karşı)
pathlib.PosixPath = pathlib.WindowsPath

# 🔧 GEREKLİ YOL: yolov5 klasörünü sys.path'e ekle
sys.path.append(r'C:\Users\donme\Desktop\yolov5')  # yolov5 klasörünü bu konuma klonladıysan

# ✅ YOLOv5 modelini yükle (kendi modelin)
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=r'C:\Users\donme\Desktop\MachingAlg\best.pt', force_reload=True)

# ✅ Kalibrasyon dosyasını oku
fs = cv2.FileStorage(r'C:\Users\donme\Desktop\MachingAlg\stereo_calibration_data.xml', cv2.FILE_STORAGE_READ)
cameraMatrix_left = fs.getNode('cameraMatrix_left').mat()
distCoeffs_left = fs.getNode('distCoeffs_left').mat()
cameraMatrix_right = fs.getNode('cameraMatrix_right').mat()
distCoeffs_right = fs.getNode('distCoeffs_right').mat()
fs.release()

# ✅ İki kamerayı başlat
cap_left = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap_right = cv2.VideoCapture(1, cv2.CAP_MSMF)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Gerçek çözünürlükleri terminalde göster
frame_width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Sol kamera gerçek çözünürlüğü: {frame_width_left}x{frame_height_left}")
print(f"Sağ kamera gerçek çözünürlüğü: {frame_width_right}x{frame_height_right}")

if not cap_left.isOpened():
    print('Sol kamera açılamadı!')
if not cap_right.isOpened():
    print('Sağ kamera açılamadı!')
if not cap_left.isOpened() or not cap_right.isOpened():
    exit()

# ✅ Tracking nesneleri oluştur (daha sıkı parametreler)
tracker_left = DeepSort(max_age=100)  # parametreler ayarlanabilir
tracker_right = DeepSort(max_age=30)  # parametreler ayarlanabilir

def compute_iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def draw_label_with_bg(img, text, topleft, font_scale=0.25, text_color=(0,0,0), bg_color=(0,255,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = topleft
    pad = 1  # çok az padding
    cv2.rectangle(img, (x - pad, y - h - baseline - pad), (x + w + pad, y + baseline + pad), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

while True:
    ret_left, img_left = cap_left.read()
    ret_right, img_right = cap_right.read()
    if not ret_left or not ret_right:
        print('Görüntü alınamadı!')
        break

    # ✅ Undistort işlemi
    img_left_undist = cv2.undistort(img_left, cameraMatrix_left, distCoeffs_left)
    img_right_undist = cv2.undistort(img_right, cameraMatrix_right, distCoeffs_right)

    # ✅ YOLOv5 ile suntaları tespit et
    results_left = model(img_left_undist)
    boxes_left = results_left.xyxy[0].cpu().numpy()
    boxes_left = boxes_left[boxes_left[:, 4] > 0.5]  # 0.5 yerine daha yüksek bir değer deneyebilirsin

    results_right = model(img_right_undist)
    boxes_right = results_right.xyxy[0].cpu().numpy()

    # ✅ Tracking güncelle
    rects_left = [(int(x1), int(y1), int(x2), int(y2)) for (x1, y1, x2, y2, conf, cls) in boxes_left]
    rects_right = [(int(x1), int(y1), int(x2), int(y2)) for (x1, y1, x2, y2, conf, cls) in boxes_right]
    
    detections_left = []
    for box in boxes_left:
        x1, y1, x2, y2, conf, cls = box
        detections_left.append([[float(x1), float(y1), float(x2), float(y2)], float(conf), int(cls)])

    detections_right = []
    for box in boxes_right:
        x1, y1, x2, y2, conf, cls = box
        detections_right.append([[float(x1), float(y1), float(x2), float(y2)], float(conf), int(cls)])

    tracks_left = tracker_left.update_tracks(detections_left, frame=img_left_undist)
    tracks_right = tracker_right.update_tracks(detections_right, frame=img_right_undist)

    # ✅ Sol kamerada accuracy ve ID göster
    sol_centers = []  # (cx, cy, id, conf, cls, bbox)
    for i, box in enumerate(boxes_left):
        x1, y1, x2, y2, conf, cls = box
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        objectID = None
        for track in tracks_left:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            if abs(centroid[0] - ltrb[0]) < 30 and abs(centroid[1] - ltrb[1]) < 30:
                objectID = track_id
                break
        if objectID is not None:
            sol_centers.append((centroid[0], centroid[1], objectID, conf, cls, (x1, y1, x2, y2)))
            cv2.rectangle(img_left_undist, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            draw_label_with_bg(img_left_undist, f"Conf: {conf:.2f}", (int(x1), int(y1)-5), font_scale=0.25)
            draw_label_with_bg(img_left_undist, f"ID: {objectID}", (int(x1), int(y1)+15), font_scale=0.25)
    # ✅ Sağ kamerada accuracy ve sol kameradan ID göster (Tekil eşleşme: one-to-one matching)
    used_sol_ids = set()
    matches = []  # (j, best_sol_idx, score, method)
    for j, box in enumerate(boxes_right):
        x1, y1, x2, y2, conf, cls = box
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        best_score = 0.0
        best_sol_idx = None
        method = None
        for i, (cx, cy, sol_id, sol_conf, sol_cls, (sx1, sy1, sx2, sy2)) in enumerate(sol_centers):
            if int(cls) != int(sol_cls) or sol_id in used_sol_ids:
                continue
            iou = compute_iou((x1, y1, x2, y2), (sx1, sy1, sx2, sy2))
            if iou > 0.1 and iou > best_score:
                best_score = iou
                best_sol_idx = i
                method = 'iou'
        if best_sol_idx is None:
            # IoU ile eşleşme yoksa, merkez yakınlığına bak
            min_dist = 40
            for i, (cx, cy, sol_id, sol_conf, sol_cls, (sx1, sy1, sx2, sy2)) in enumerate(sol_centers):
                if int(cls) != int(sol_cls) or sol_id in used_sol_ids:
                    continue
                dist = np.sqrt((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_score = 1.0 / (dist + 1e-6)
                    best_sol_idx = i
                    method = 'dist'
        matches.append((j, best_sol_idx, best_score, method))
    # Greedy olarak en iyi eşleşmeleri ata
    matches.sort(key=lambda x: x[2], reverse=True)  # skora göre sırala
    right_to_sol = {}
    for j, best_sol_idx, score, method in matches:
        if best_sol_idx is not None:
            sol_id = sol_centers[best_sol_idx][2]
            if sol_id not in used_sol_ids:
                right_to_sol[j] = sol_id
                used_sol_ids.add(sol_id)
    # Kutuları çiz
    for j, box in enumerate(boxes_right):
        x1, y1, x2, y2, conf, cls = box
        objectID = right_to_sol.get(j, None)
        cv2.rectangle(img_right_undist, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        draw_label_with_bg(img_right_undist, f"Conf: {conf:.2f}", (int(x1), int(y1)-5), font_scale=0.25)
        if objectID is not None:
            draw_label_with_bg(img_right_undist, f"ID: {objectID}", (int(x1), int(y1)+15), font_scale=0.25)
        else:
            draw_label_with_bg(img_right_undist, f"ID: -", (int(x1), int(y1)+15), font_scale=0.25)

    # ✅ Sayaç ekle
    id_set = set()
    for track in tracks_left:
        if not track.is_confirmed():
            continue
        id_set.add(track.track_id)
    count_left = len(id_set)
    
    cv2.putText(img_left_undist, f'Sunta Sayisi: {count_left}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    # ✅ Görüntüleri göster
    cv2.imshow('Sol Kamera - ID Gosterimi', img_left_undist)
    cv2.imshow('Sağ Kamera - Doğruluk Oranları', img_right_undist)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

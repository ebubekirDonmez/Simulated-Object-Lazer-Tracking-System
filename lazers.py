import cv2
import torch
import numpy as np
import pathlib
import time
import ctypes
pathlib.PosixPath = pathlib.WindowsPath

# matching.py'den fonksiyonel hale getirilmiş tespit ve eşleştirme fonksiyonu

def preprocess_image(img):
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (H, W, C) -> (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    # Numpy -> Tensor
    img = torch.from_numpy(img)
    # float32 ve normalize
    img = img.float() / 255.0
    # Batch dimension ekle
    img = img.unsqueeze(0)
    # CUDA'ya aktar
    img = img.to('cuda')
    return img

def resize_to_32_multiple(img):
    h, w = img.shape[:2]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    return cv2.resize(img, (new_w, new_h))

def detect_and_match(cap_left, cap_right, model, cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, conf_thresh=0.5):
    """
    Sol ve sağ kameradan görüntü alır, undistort ve YOLOv5 ile tespit yapar,
    epipolar ve boyut benzerliği ile eşleştirme yapar.
    Dönüş: Eşleşmiş köklerin merkezleri ve ID'leri
    """
    ret_left, img_left = cap_left.read()
    ret_right, img_right = cap_right.read()
    if not ret_left or not ret_right:
        return [], [], [], [], None, None

    # Undistort
    img_left_undist = cv2.undistort(img_left, cameraMatrix_left, distCoeffs_left)
    img_right_undist = cv2.undistort(img_right, cameraMatrix_right, distCoeffs_right)
    # 32'nin katı olacak şekilde yeniden boyutlandır
    img_left_undist = resize_to_32_multiple(img_left_undist)
    img_right_undist = resize_to_32_multiple(img_right_undist)

    # YOLOv5 ile tespit
    img_left_tensor = preprocess_image(img_left_undist)
    pred_left = model(img_left_tensor)[0]
    pred_left = non_max_suppression(pred_left, conf_thres=conf_thresh, iou_thres=0.45)[0]
    if pred_left is not None and len(pred_left):
        boxes_left = pred_left.cpu().numpy()
    else:
        boxes_left = np.zeros((0, 6))

    img_right_tensor = preprocess_image(img_right_undist)
    pred_right = model(img_right_tensor)[0]
    pred_right = non_max_suppression(pred_right, conf_thres=conf_thresh, iou_thres=0.45)[0]
    if pred_right is not None and len(pred_right):
        boxes_right = pred_right.cpu().numpy()
    else:
        boxes_right = np.zeros((0, 6))

    # Sol ve sağ için merkezler ve kutular
    sunta_list_left = []
    for box in boxes_left:
        x1, y1, x2, y2, conf, cls = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        sunta_list_left.append((cx, cy, x1, y1, x2, y2, conf, cls))
    sunta_list_right = []
    for box in boxes_right:
        x1, y1, x2, y2, conf, cls = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        sunta_list_right.append((cx, cy, x1, y1, x2, y2, conf, cls))

    # Eşleştirme algoritması (epipolar ve boyut benzerliği)
    Y_TOL = 20
    SIZE_TOL = 0.4
    X_TOL = 100
    matches = []
    used_right = set()
    for i, (cx_l, cy_l, x1_l, y1_l, x2_l, y2_l, conf_l, cls_l) in enumerate(sunta_list_left):
        best_j = None
        best_score = float('inf')
        for j, (cx_r, cy_r, x1_r, y1_r, x2_r, y2_r, conf_r, cls_r) in enumerate(sunta_list_right):
            if j in used_right:
                continue
            if abs(cy_l - cy_r) > Y_TOL:
                continue
            w_l, h_l = x2_l - x1_l, y2_l - y1_l
            w_r, h_r = x2_r - x1_r, y2_r - y1_r
            if abs(w_l - w_r) / max(w_l, w_r) > SIZE_TOL or abs(h_l - h_r) / max(h_l, h_r) > SIZE_TOL:
                continue
            if abs(cx_l - cx_r) > X_TOL:
                continue
            score = abs(cx_l - cx_r) + abs(cy_l - cy_r)
            if score < best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            matches.append((i, best_j))
            used_right.add(best_j)

    # ID atama
    left_ids = [-1] * len(sunta_list_left)
    right_ids = [-1] * len(sunta_list_right)
    current_id = 1
    for i, j in matches:
        left_ids[i] = current_id
        right_ids[j] = current_id
        current_id += 1
    for idx in range(len(sunta_list_left)):
        if left_ids[idx] == -1:
            left_ids[idx] = current_id
            current_id += 1
    for idx in range(len(sunta_list_right)):
        if right_ids[idx] == -1:
            right_ids[idx] = current_id
            current_id += 1

    # Dönüş: kutular, ID'ler, görüntüler (gerekirse)
    return sunta_list_left, left_ids, sunta_list_right, right_ids, img_left_undist, img_right_undist

# Kullanım örneği (ana kodda):
# from lazers import detect_and_match
# sunta_list_left, left_ids, sunta_list_right, right_ids, img_left, img_right = detect_and_match(...)

# Global değişkenler
next_id_left = 1
active_suntas_left = []  # [(id, (cx, cy)), ...]
next_id_right = 1
active_suntas_right = []

def update_ids(current_sunta_centers, active_suntas, next_id):
    new_active = []
    used_prev = set()
    id_map = {}
    for i, (cx, cy) in enumerate(current_sunta_centers):
        min_dist = 30
        best_id = None
        for prev_id, (pcx, pcy) in active_suntas:
            if prev_id in used_prev:
                continue
            dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
            if dist < min_dist:
                min_dist = dist
                best_id = prev_id
        if best_id is not None:
            id_map[i] = best_id
            used_prev.add(best_id)
            new_active.append((best_id, (cx, cy)))
        else:
            id_map[i] = next_id
            new_active.append((next_id, (cx, cy)))
            next_id += 1
    return [id_map[i] for i in range(len(current_sunta_centers))], new_active, next_id

if __name__ == '__main__':
    import torch
    import time

    # Kameraları başlat
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Kameralar açılamadı! Lütfen bağlantıyı ve indexleri kontrol edin.")
        exit()

    # Çözünürlüğü ayarla
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 950)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 544)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 950)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 544)

    # Modeli yükle
    print("CUDA kullanılabilir mi:", torch.cuda.is_available())
    import sys
    sys.path.append(r'C:\Users\donme\Desktop\MachingAlg\yolov5')
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    model = attempt_load(r'C:\Users\donme\Desktop\MachingAlg\best.pt', device='cuda')
    print("Model hangi cihazda:", next(model.parameters()).device)

    # Kalibrasyon dosyasını oku
    fs = cv2.FileStorage(r'C:\Users\donme\Desktop\MachingAlg\stereo_calibration_data1.xml', cv2.FILE_STORAGE_READ)
    cameraMatrix_left = fs.getNode('cameraMatrix_left').mat()
    distCoeffs_left = fs.getNode('distCoeffs_left').mat()
    cameraMatrix_right = fs.getNode('cameraMatrix_right').mat()
    distCoeffs_right = fs.getNode('distCoeffs_right').mat()
    fs.release()

    # Lazer state'leri (her kamera için)
    lazerler_left = [
        {'state': 'idle', 'assigned_center': None, 'start_time': None}
        for _ in range(4)
    ]
    lazerler_right = [
        {'state': 'idle', 'assigned_center': None, 'start_time': None}
        for _ in range(4)
    ]
    # İşlenmiş suntalar merkez koordinatına göre tutulacak (her bölge için set)
    islenmis_suntalar_left = [[] for _ in range(4)]  # yeni hali: liste
    islenmis_suntalar_right = [[] for _ in range(4)]
    CENTER_TOL = 20  # piksel toleransı

    # Ekran çözünürlüğünü al
    user32 = ctypes.windll.user32
    screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        sunta_list_left, _, sunta_list_right, _, img_left, img_right = detect_and_match(
            cap_left, cap_right, model,
            cameraMatrix_left, distCoeffs_left,
            cameraMatrix_right, distCoeffs_right
        )
        current_time = time.time()
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (current_time - prev_time)
            prev_time = current_time
        if img_left is not None:
            h, w, _ = img_left.shape
            cv2.putText(img_left, f"FPS: {fps:.1f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            for i in range(1, 4):
                x = int(w * i / 4)
                cv2.line(img_left, (x, 0), (x, h), (0, 0, 255), 1)
            for i in range(4):
                alanlar = [i]
                if i == 0:
                    alanlar.append(1)
                elif i == 3:
                    alanlar.append(2)
                else:
                    alanlar.extend([i-1, i+1])
                suntalar = []
                for alan in set(alanlar):
                    x_start = int(w * alan / 4)
                    x_end = int(w * (alan + 1) / 4)
                    suntalar.extend([
                        (cx, cy, idx, alan) for idx, (cx, cy, *_ ) in enumerate(sunta_list_left) if x_start <= cx < x_end
                    ])
                suntalar = sorted(suntalar, key=lambda t: t[0])
                id_map = {}
                alan_sayac = {alan:1 for alan in set(alanlar)}
                for cx, cy, idx, alan in suntalar:
                    id_map[(cx, cy, idx)] = alan_sayac[alan]
                    alan_sayac[alan] += 1
                lazer = lazerler_left[i]
                islenmis = islenmis_suntalar_left[i]
                islenmemis_suntalar = []
                for cx, cy, idx, alan in suntalar:
                    local_id = id_map[(cx, cy, idx)]
                    cv2.rectangle(img_left, (int(sunta_list_left[idx][2]), int(sunta_list_left[idx][3])),
                                  (int(sunta_list_left[idx][4]), int(sunta_list_left[idx][5])), (0,255,0), 2)
                    cv2.circle(img_left, (cx, cy), 5, (255,0,0), -1)
                    cv2.putText(img_left, f'ID:{local_id}', (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    # Eğer daha önce işlenmişse tekrar işleme
                    bulundu = False
                    for k, (old_cx, old_cy) in enumerate(islenmis):
                        if np.linalg.norm(np.array([cx, cy]) - np.array([old_cx, old_cy])) < CENTER_TOL:
                            # Eğer yakınsa, eski merkezi güncelle
                            islenmis[k] = (cx, cy)
                            bulundu = True
                            break
                    if bulundu:
                        continue
                    islenmemis_suntalar.append((cx, cy, local_id))
                if lazer['state'] == 'active':
                    kalan = 1.0 - (current_time - lazer['start_time'])
                    if kalan > 0.1:
                        for (cx, cy, local_id) in islenmemis_suntalar:
                            if np.linalg.norm(np.array([cx, cy]) - np.array(lazer['assigned_center'])) < CENTER_TOL:
                                cv2.circle(img_left, (cx, cy), 5, (0,0,255), -1)
                                cv2.putText(img_left, f'ID:{local_id} | {kalan:.1f}', (cx+10, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    else:
                        islenmis.append(tuple(lazer['assigned_center'])) # işlenmişlere ekle
                        lazer['state'] = 'idle'
                        lazer['assigned_center'] = None
                        lazer['start_time'] = None
                if lazer['state'] == 'idle' and islenmemis_suntalar:
                    cx, cy, local_id = islenmemis_suntalar[0]
                    lazer['state'] = 'active'
                    lazer['assigned_center'] = (cx, cy)
                    lazer['start_time'] = current_time
                    islenmis_suntalar_left[i].append((cx, cy))  # işlenmişlere ekle
            cv2.putText(img_left, f'Sunta Sayisi: {len(sunta_list_left)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
            img_left_resized = cv2.resize(img_left, (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Sol Kamera', img_left_resized)
            try:
                cv2.moveWindow('Sol Kamera', 0, 0)
            except cv2.error:
                pass
        if img_right is not None:
            h, w, _ = img_right.shape
            for i in range(1, 4):
                x = int(w * i / 4)
                cv2.line(img_right, (x, 0), (x, h), (0, 0, 255), 1)
            for i in range(4):
                alanlar = [i]
                if i == 0:
                    alanlar.append(1)
                elif i == 3:
                    alanlar.append(2)
                else:
                    alanlar.extend([i-1, i+1])
                suntalar = []
                for alan in set(alanlar):
                    x_start = int(w * alan / 4)
                    x_end = int(w * (alan + 1) / 4)
                    suntalar.extend([
                        (cx, cy, idx, alan) for idx, (cx, cy, *_ ) in enumerate(sunta_list_right) if x_start <= cx < x_end
                    ])
                suntalar = sorted(suntalar, key=lambda t: t[0])
                id_map = {}
                alan_sayac = {alan:1 for alan in set(alanlar)}
                for cx, cy, idx, alan in suntalar:
                    id_map[(cx, cy, idx)] = alan_sayac[alan]
                    alan_sayac[alan] += 1
                lazer = lazerler_right[i]
                islenmis = islenmis_suntalar_right[i]
                islenmemis_suntalar = []
                for cx, cy, idx, alan in suntalar:
                    local_id = id_map[(cx, cy, idx)]
                    cv2.rectangle(img_right, (int(sunta_list_right[idx][2]), int(sunta_list_right[idx][3])),
                                  (int(sunta_list_right[idx][4]), int(sunta_list_right[idx][5])), (0,255,0), 2)
                    cv2.circle(img_right, (cx, cy), 5, (255,0,0), -1)
                    cv2.putText(img_right, f'ID:{local_id}', (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    if any(np.linalg.norm(np.array([cx, cy]) - np.array(center)) < CENTER_TOL for center in islenmis):
                        continue
                    islenmemis_suntalar.append((cx, cy, local_id))
                if lazer['state'] == 'active':
                    kalan = 1.0 - (current_time - lazer['start_time'])
                    if kalan > 0.1:
                        for (cx, cy, local_id) in islenmemis_suntalar:
                            if np.linalg.norm(np.array([cx, cy]) - np.array(lazer['assigned_center'])) < CENTER_TOL:
                                cv2.circle(img_right, (cx, cy), 5, (0,0,255), -1)
                                cv2.putText(img_right, f'ID:{local_id} | {kalan:.1f}', (cx+10, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    else:
                        islenmis.append(tuple(lazer['assigned_center'])) # işlenmişlere ekle
                        lazer['state'] = 'idle'
                        lazer['assigned_center'] = None
                        lazer['start_time'] = None
                if lazer['state'] == 'idle' and islenmemis_suntalar:
                    cx, cy, local_id = islenmemis_suntalar[0]
                    lazer['state'] = 'active'
                    lazer['assigned_center'] = (cx, cy)
                    lazer['start_time'] = current_time
            img_right_resized = cv2.resize(img_right, (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Sağ Kamera', img_right_resized)
            try:
                cv2.moveWindow('Sağ Kamera', 1000, 0)
            except cv2.error:
                pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Çıkış yapılıyor, pencereler kapatılıyor...")
            break
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

# руки: сглаженная поза + приклейка к пикселям оптическим потоком (комплементарный фильтр)
import cv2, json, numpy as np
from scipy.ndimage import gaussian_filter1d
P = np.array(json.load(open('pose_raw.json'))); Ps = gaussian_filter1d(P, 1.5, axis=0)
cap = cv2.VideoCapture('seg.mp4'); prev = None; X = Ps.copy(); i = 0; KEY = [13, 14, 15, 16]
while True:
    ok, fr = cap.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(fr, (540, 960)), cv2.COLOR_BGR2GRAY)
    if prev is not None:
        m = cv2.resize(cv2.imread(f'ma/{i:04d}.png', 0), (540, 960))
        pts = cv2.goodFeaturesToTrack(prev, 800, 0.005, 4, mask=(m > 100).astype(np.uint8) * 255)
        nx, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
        p0 = pts[st[:, 0] == 1, 0] * 2; fl = nx[st[:, 0] == 1, 0] * 2 - p0
        for k in KEY:
            x = X[i - 1, k].copy(); sel = np.linalg.norm(p0 - x, axis=1) < 60
            x = x + (np.median(fl[sel], 0) if sel.sum() >= 5 else Ps[i, k] - Ps[i - 1, k])
            X[i, k] = x + 0.15 * (Ps[i, k] - x)
    prev = g; i += 1
json.dump(X[:, KEY].tolist(), open('arms.json', 'w'))
d = np.diff(X[120:, 13], axis=0); print('jerk', np.abs(np.diff(d, axis=0)).mean(0).round(2))

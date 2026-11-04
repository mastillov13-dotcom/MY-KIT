# Корсет строго по оси: середина силуэта торса, вертикаль в выровненном кадре, зеркальная симметрия, без поворота
import cv2, json, numpy as np
from scipy.ndimage import gaussian_filter1d
W, H = 1080, 1920; TILT = 1.6; T0 = 110
V = np.array(json.load(open('pose_stab.json')))
N = len(V)
ZL = np.cos(np.radians(TILT)) + (H / W) * np.sin(np.radians(TILT)) * 0.98
Mr = cv2.getRotationMatrix2D((W / 2, H / 2), TILT, ZL)[:, :2] / ZL
d = np.linalg.inv(Mr) @ np.array([0.0, 1.0]); d /= np.linalg.norm(d)   # вертикаль финального кадра в исходных координатах
pv = np.array([d[1], -d[0]])                                          # поперёк оси

M = (V[:, 11] + V[:, 12]) / 2; Hm = (V[:, 23] + V[:, 24]) / 2
hw = np.linalg.norm(V[:, 11] - V[:, 12], axis=1) / 2
hh = np.linalg.norm(V[:, 23] - V[:, 24], axis=1) / 2
L = np.linalg.norm(Hm - M, axis=1)

# середина силуэта торса по маске (строки середины торса, где нет рук)
mid = np.full(N, np.nan)
for t in range(T0, N):
    m = np.mean([cv2.imread(f'{dd}/{t:04d}.png', 0) for dd in ('ma', 'mb')], 0) > 120
    xs = []
    for f in np.linspace(0.45, 0.85, 14):
        q = M[t] + (Hm[t] - M[t]) * f; y = int(q[1])
        row = np.where(m[y])[0]
        row = row[np.abs(row - q[0]) < hw[t] * 1.7]
        if len(row) > 10: xs.append((row.min() + row.max()) / 2 - (q[0] - M[t][0]) * 0)
    if xs: mid[t] = np.median(xs) - 0  # x середины на уровне середины торса
ok = ~np.isnan(mid); mid[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], mid[ok])
mid = gaussian_filter1d(mid, 4)
hw_s = gaussian_filter1d(hw, 8); hh_s = gaussian_filter1d(hh, 8); L_s = gaussian_filter1d(L, 8)
My = gaussian_filter1d(M[:, 1], 2.5)

out = V.copy()
for t in range(N):
    # точка оси на высоте середины торса: x из силуэта
    cy = My[t] + L_s[t] * 0.65 * d[1]
    c_mid = np.array([mid[t], cy])
    Mt = c_mid - d * L_s[t] * 0.65
    Ht = Mt + d * L_s[t]
    s0 = 1 if V[t, 11, 0] > V[t, 12, 0] else -1     # какая сторона справа
    out[t, 11] = Mt + pv * hw_s[t] * s0 * np.sign(pv[0]); out[t, 12] = Mt - pv * hw_s[t] * s0 * np.sign(pv[0])
    out[t, 23] = Ht + pv * hh_s[t] * s0 * np.sign(pv[0]); out[t, 24] = Ht - pv * hh_s[t] * s0 * np.sign(pv[0])
json.dump(out.tolist(), open('pose_axis.json', 'w'))
dev = (M[T0:, 0] - (mid[T0:] - d[0] / d[1] * L_s[T0:] * 0.65))
print('axis dir', d.round(4), 'старый центр плеч отклонялся от оси на', np.abs(dev).mean().round(1), 'px (макс', np.abs(dev).max().round(1), ')')

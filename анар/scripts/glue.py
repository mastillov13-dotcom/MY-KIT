# Приклейка корсета к спине: оптический поток по торсу -> подобие (сдвиг, поворот, масштаб) к опорному кадру
import cv2, json, numpy as np
from scipy.ndimage import gaussian_filter1d
A = np.array(json.load(open('pa.json')))[:, :, :2]; B = np.array(json.load(open('pb.json')))[:, :, :2]
P = gaussian_filter1d((A + B) / 2, 3, axis=0)
N = len(P); REF = 165; T0 = 110  # с этого кадра он уже сидит

cap = cv2.VideoCapture('seg.mp4'); G = []; CL = cv2.createCLAHE(3.0, (16, 16))
while True:
    ok, fr = cap.read()
    if not ok: break
    G.append(CL.apply(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))

def torso_mask(p):
    S0, S1, H0, H1 = p[11], p[12], p[23], p[24]
    M = (S0 + S1) / 2; Hm = (H0 + H1) / 2
    poly = [M + (S0 - M) * 0.75 + (Hm - M) * 0.08, M + (S1 - M) * 0.75 + (Hm - M) * 0.08,
            Hm + (H1 - Hm) * 1.1 - (Hm - M) * 0.08, Hm + (H0 - Hm) * 1.1 - (Hm - M) * 0.08]
    m = np.zeros((1920, 1080), np.uint8); cv2.fillPoly(m, [np.int32(poly)], 255)
    hw = np.linalg.norm(S0 - S1) / 2
    cv2.circle(m, tuple(np.int32(M - (Hm - M) * 0.45)), int(hw * 0.55), 255, -1)   # затылок
    cv2.ellipse(m, tuple(np.int32(Hm + (Hm - M) * 0.15)), (int(hw * 1.1), int(hw * 0.35)), 0, 0, 360, 255, -1)  # пояс/шорты
    return m

ref = G[REF]
rp = cv2.goodFeaturesToTrack(ref, 2000, 0.0008, 5, mask=torso_mask(P[REF]), blockSize=5)
T = [None] * N; T[REF] = np.float32([[1, 0, 0], [0, 1, 0]])
def step(t, prevT):
    init = cv2.transform(rp, prevT)
    nx, st, err = cv2.calcOpticalFlowPyrLK(ref, G[t], rp, init.copy(), winSize=(31, 31), maxLevel=4,
                                           flags=cv2.OPTFLOW_USE_INITIAL_FLOW, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    bk, st2, _ = cv2.calcOpticalFlowPyrLK(G[t], ref, nx, rp.copy(), winSize=(31, 31), maxLevel=4, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(bk - rp, axis=2)[:, 0] < 1.5)
    if ok.sum() < 20: return prevT, int(ok.sum())
    Mx, inl = cv2.estimateAffinePartial2D(rp[ok], nx[ok], method=cv2.RANSAC, ransacReprojThreshold=2.5)
    return (Mx if Mx is not None else prevT), int(ok.sum())
cnt = {}
for t in range(REF + 1, N): T[t], cnt[t] = step(t, T[t - 1])
for t in range(REF - 1, T0 - 1, -1): T[t], cnt[t] = step(t, T[t + 1])
for t in range(T0): T[t] = T[T0]
# параметры подобия и их лёгкое сглаживание
par = np.array([[np.log(np.hypot(m[0, 0], m[1, 0])), np.arctan2(m[1, 0], m[0, 0]), m[0, 2], m[1, 2]] for m in T])
par[T0:] = gaussian_filter1d(par[T0:], 1.2, axis=0)
def mat(q):
    s = np.exp(q[0]); c, sn = np.cos(q[1]) * s, np.sin(q[1]) * s
    return np.float32([[c, -sn, q[2]], [sn, c, q[3]]])
T = [mat(q) for q in par]
inv = [cv2.invertAffineTransform(m) for m in T]

# канонический торс в опорных координатах: среднее по кадрам, без шума
def tr(m, pts): return cv2.transform(pts[None].astype(np.float32), m)[0]
Pref = np.array([tr(inv[t], P[t]) for t in range(N)])
seated = slice(T0 + 10, N)
canon = np.median(Pref[seated], 0)
# плечи слегка «дышат» с движением рук: смещение плеча, сильно сглаженное, на 50%
off = gaussian_filter1d(Pref[:, [11, 12]] - canon[[11, 12]], 5, axis=0) * 0.5
V = []
for t in range(N):
    q = canon.copy(); q[[11, 12]] += off[t]
    q[[13, 14, 15, 16]] = Pref[t, [13, 14, 15, 16]]
    V.append(tr(T[t], q))
V = np.array(V)
json.dump(V.tolist(), open('pose_stab.json', 'w'))
json.dump(P.tolist(), open('pose_raw.json', 'w'))
np.save('T.npy', np.array(T))
bad=[t for t,c in cnt.items() if c<40]; print('bad frames',len(bad),bad[:10]); print('features', len(rp), 'min tracked', min(cnt.values()), 'scale range', np.exp(par[T0:, 0]).min().round(3), np.exp(par[T0:, 0]).max().round(3),
      'shift', np.ptp(par[T0:, 2]).round(1), np.ptp(par[T0:, 3]).round(1))

# Стиль «атлас»: анатомия поверх тела, рабочие мышцы красные с жёлтым контуром, остальные серые, жёлтые стрелки движения
import cv2, json, numpy as np, os, sys
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool

W, H = 1080, 1920
P = np.array(json.load(open('pose_stab.json')))
d = (P[:, 15, 1] + P[:, 16, 1]) / 2 - (P[:, 11, 1] + P[:, 12, 1]) / 2
C = gaussian_filter1d(np.clip((d + 320) / 320, 0, 1), 2)
START, END = 12, 270

def nrm(v): return v / (np.linalg.norm(v) + 1e-6)
def lerp(a, b, t): return a + (b - a) * t
def ease(t): t = float(np.clip(t, 0, 1)); return t * t * (3 - 2 * t)
def chaikin(pts, it=3):
    p = np.array(pts, np.float32)
    for _ in range(it):
        q = np.roll(p, -1, 0); p = np.stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q], 1).reshape(-1, 2)
    return p
def bez(a, c, b, n=18):
    t = np.linspace(0, 1, n)[:, None]; return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b
def ell(cen, ax, a, b, n=14):
    px = np.array([-ax[1], ax[0]])
    return [cen + ax * np.cos(t) * a + px * np.sin(t) * b for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]

def anatomy(p, c):
    S = {0: p[11], 1: p[12]}; E = {0: p[13], 1: p[14]}; Hh = {0: p[23], 1: p[24]}
    M = (S[0] + S[1]) / 2; Hm = (Hh[0] + Hh[1]) / 2
    v = nrm(Hm - M); hw = np.linalg.norm(S[0] - S[1]) / 2; sp = lambda t: lerp(M, Hm, t)
    bl = 0.05 * c
    G, R = [], []          # серые и рабочие
    lines = {}
    for s in (0, 1):
        u = nrm(S[s] - M); ax = nrm(E[s] - S[s]); px = np.array([-ax[1], ax[0]]); L = np.linalg.norm(E[s] - S[s])
        arm = S[s] + (E[s] - S[s]) * 0.14 - u * 0.08 * hw + v * 0.10 * hw
        # мышцы-разгибатели спины
        er = [sp(0.38) + u * 0.03 * hw, sp(0.38) + u * 0.20 * hw, sp(0.75) + u * 0.24 * hw, sp(1.02) + u * 0.16 * hw, sp(1.02) + u * 0.03 * hw]
        G.append(('erector', chaikin(er, 2), [bez(sp(0.38) + u * q * hw, sp(0.7) + u * (q + 0.02) * hw, sp(1.0) + u * q * 0.8 * hw, 16) for q in np.linspace(0.05, 0.2, 5)]))
        # косые живота — оранжевый градиент
        ob = [sp(0.58) + u * 0.85 * hw, sp(0.62) + u * 1.0 * hw, lerp(Hm, Hh[s], 1.15) - v * 0.02 * hw, lerp(Hm, Hh[s], 0.8) + v * 0.08 * hw]
        G.append(('oblique', chaikin(ob, 2), [bez(sp(0.6) + u * (0.85 + q * 0.15) * hw, sp(0.8) + u * (0.95 + q * 0.1) * hw, lerp(Hm, Hh[s], 0.8 + q * 0.3), 12) for q in np.linspace(0, 1, 6)]))
        # подостная на лопатке
        inf = [sp(0.10) + u * 0.30 * hw, S[s] - u * 0.14 * hw + v * 0.10 * hw, sp(0.30) + u * 0.80 * hw, sp(0.34) + u * 0.45 * hw]
        G.append(('infra', chaikin(inf, 2), [bez(lerp(sp(0.12) + u * 0.30 * hw, sp(0.34) + u * 0.45 * hw, q), lerp(sp(0.15) + u * 0.6 * hw, sp(0.3) + u * 0.7 * hw, q), S[s] - u * 0.1 * hw + v * 0.12 * hw, 12) for q in np.linspace(0, 1, 9)]))
        # большая круглая — рабочая
        sc = sp(0.26) + u * 0.62 * hw; ta = nrm(arm - sc); tl = np.linalg.norm(arm - sc)
        tm = ell((sc + arm) / 2, ta, tl * 0.55, (0.11 + bl) * hw, 12)
        R.append(('teres', chaikin(tm, 4), [bez(sc + np.array([-ta[1], ta[0]]) * q * 0.08 * hw, (sc + arm) / 2 + np.array([-ta[1], ta[0]]) * q * 0.1 * hw, arm, 14) for q in np.linspace(-0.9, 0.9, 8)]))
        # широчайшая — рабочая
        # мягкий органичный контур: округлый низ и выпуклый наружный край
        poly = [arm, sp(0.24) + u * 0.30 * hw, sp(0.34) + u * 0.12 * hw, sp(0.60) + u * 0.06 * hw, sp(0.86) + u * 0.08 * hw,
                sp(0.97) + u * 0.22 * hw, lerp(Hm, Hh[s], 0.78) - v * 0.02 * hw, sp(0.80) + u * (0.82 + bl) * hw,
                sp(0.58) + u * (0.98 + bl) * hw, sp(0.38) + u * (1.08 + bl) * hw, sp(0.20) + u * 1.00 * hw]
        poly = [poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], poly[8], poly[9], poly[10]]
        org = np.vstack([bez(poly[2], poly[3], poly[4], 18), bez(poly[4], poly[5], poly[6], 12)])
        fib = [bez(o, (o + arm) / 2 + u * (0.18 + bl) * hw, arm + ax * hw * 0.1 * (i / len(org) - 0.5), 22) for i, o in enumerate(org)]
        R.append(('lat', chaikin(poly, 5), fib))
        lines[s] = dict(ins=arm, org=lerp(Hm, Hh[s], 0.6) - v * 0.04 * hw, mid=sp(0.62) + u * 0.75 * hw)
        # задняя дельта и трицепс — серые
        cen = S[s] + ax * 0.04 * L + u * 0.22 * hw + v * 0.08 * hw
        G.append(('delt', chaikin(ell(cen, ax, 0.22 * L, 0.22 * hw), 2), [bez(cen - ax * 0.28 * L + px * q * 0.2 * hw, cen + px * q * 0.24 * hw, cen + ax * 0.3 * L, 14) for q in np.linspace(-0.9, 0.9, 10)]))
        ap = [S[s] + ax * 0.32 * L + px * 0.20 * hw, E[s] - ax * 0.02 * L + px * 0.13 * hw, E[s] - ax * 0.02 * L - px * 0.13 * hw, S[s] + ax * 0.32 * L - px * 0.20 * hw]
        if False: G.append(('arm', chaikin(ap, 2), [bez(S[s] + ax * 0.33 * L + px * q * 0.17 * hw, lerp(S[s], E[s], 0.65) + px * q * 0.18 * hw, E[s] - ax * 0.04 * L, 14) for q in np.linspace(-0.8, 0.8, 8)]))
    # трапеция — серая, поверх
    u0 = nrm(S[0] - M); u1 = nrm(S[1] - M); top = M - v * 0.55 * hw
    tp = [top + u0 * 0.14 * hw, S[0] - u0 * 0.12 * hw - v * 0.06 * hw, sp(0.24) + u0 * 0.42 * hw, sp(0.47),
          sp(0.24) + u1 * 0.42 * hw, S[1] - u1 * 0.12 * hw - v * 0.06 * hw, top + u1 * 0.14 * hw]
    fibt = []
    for t in np.linspace(0, 1, 16):
        o = lerp(top, sp(0.47), t)
        for s, uu in ((0, u0), (1, u1)):
            ins = lerp(S[s] - uu * 0.14 * hw - v * 0.04 * hw, sp(0.22) + uu * 0.40 * hw, t)
            fibt.append(bez(o, (o + ins) / 2 - v * 0.03 * hw, ins, 14))
    G.append(('trap', chaikin(tp, 5), fibt))
    order = ['erector', 'oblique', 'infra', 'teres', 'lat', 'trap', 'delt', 'arm']
    allm = [(n, po, fb, 'R') for n, po, fb in R] + [(n, po, fb, 'G') for n, po, fb in G]
    allm.sort(key=lambda m: order.index(m[0]))
    return allm, lines, dict(S=S, E=E, hw=hw)

GRAY = np.float32([186, 189, 196]); EDGE_O = np.float32([40, 120, 235])
RED = np.float32([35, 45, 245]); RED_HI = np.float32([50, 150, 255]); RED_LO = np.float32([20, 20, 150])
YEL = (60, 225, 255)

def draw_anatomy(p, c, rv_red, rv_gray):
    col = np.zeros((H, W, 3), np.float32); a = np.zeros((H, W), np.float32)
    glow = np.zeros((H, W), np.float32)
    muscles, lines, J = anatomy(p, c)
    for name, poly, fibs, kind in muscles:
        rv = rv_red if kind == 'R' else rv_gray
        if rv <= 0: continue
        x0, y0 = np.maximum(poly.min(0).astype(int) - 8, 0); x1, y1 = np.minimum(poly.max(0).astype(int) + 8, [W, H])
        if x1 <= x0 or y1 <= y0: continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [(poly - [x0, y0]).astype(np.int32)], 255, cv2.LINE_AA)
        dt = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5); dt = (dt / (dt.max() + 1e-6))[..., None]
        if kind == 'R':
            body = RED_LO + (RED - RED_LO) * np.clip(dt * 2.2, 0, 1)
            body = body + (RED_HI - RED) * np.clip((dt - 0.45) * 1.8, 0, 1) * (0.6 + 0.4 * c)
            fc, fw = (20, 20, 140), 1
        elif name == 'oblique':
            body = EDGE_O * 0.55 + EDGE_O * 0.45 * dt; fc, fw = (20, 80, 190), 1
        else:
            e = np.clip(1 - dt * 3.0, 0, 1) ** 1.5
            body = GRAY * (1 - e) + EDGE_O * e; fc, fw = (112, 115, 122), 1
        # волокна
        fl = np.zeros_like(m)
        for f in fibs:
            cv2.polylines(fl, [((f - [x0, y0]) * 4).astype(np.int32)], False, 255, fw, cv2.LINE_AA, 2)
        flf = (fl.astype(np.float32) / 255)[..., None] * 0.75
        body = body * (1 - flf) + np.float32(fc) * flf
        if kind == 'R':  # светлые блики вдоль волокон
            body = body + flf * 0.25 * 255 * np.clip(dt - 0.3, 0, 1)
        mf = (m.astype(np.float32) / 255)[..., None] * rv
        # тёмный шов по краю мышцы
        seam = (cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32) / 255)[..., None]
        body = body * (1 - seam * 0.5)
        col[y0:y1, x0:x1] = col[y0:y1, x0:x1] * (1 - mf) + body * mf
        a[y0:y1, x0:x1] = np.maximum(a[y0:y1, x0:x1], mf[..., 0])
        if kind == 'R':
            g = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(np.float32) / 255
            glow[y0:y1, x0:x1] = np.maximum(glow[y0:y1, x0:x1], g * rv)
    return col, a, glow, lines, J

def arrow(img, pts, color, th, alpha, head=16):
    if alpha <= 0.02 or len(pts) < 2: return
    pts = np.asarray(pts, np.float32)
    ov = img.copy()
    cv2.polylines(ov, [(pts * 4).astype(np.int32)], False, color, th, cv2.LINE_AA, 2)
    tip = pts[-1]; dv = nrm(pts[-1] - pts[-3 if len(pts) > 2 else 0]); pv = np.array([-dv[1], dv[0]])
    tri = np.array([tip + dv * 4, tip - dv * head + pv * head * 0.55, tip - dv * head - pv * head * 0.55])
    cv2.fillPoly(ov, [(tri * 4).astype(np.int32)], color, cv2.LINE_AA, 2)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_arrows(idx, lines, J, rv, t_phase):
    lay = np.zeros((H, W, 3), np.uint8)
    for s in (0, 1):
        k_e = 13 + s; k_s = 11 + s; k_w = 15 + s
        # 1) след локтя: куда идёт рука прямо сейчас
        tr = P[max(START, idx - 14):idx + 1, k_w]
        sp_ = np.linalg.norm(P[idx, k_w] - P[max(0, idx - 3), k_w]) / 3
        if len(tr) > 3:
            arrow(lay, tr + nrm(tr[-1] - P[idx, k_s]) * 0 + np.array([(1 if P[idx,k_w,0] > P[idx,k_s,0] else -1) * 55, 0]), YEL, 4, rv * ease((sp_ - 1.0) / 2.5), head=24)
        # 2) дуга вращения плеча
        Sj, Ej = P[idx, k_s], P[idx, k_e]
        ang_now = np.arctan2(*(Ej - Sj)[::-1]); ang_prev = np.arctan2(*(P[max(0, idx - 6), k_e] - P[max(0, idx - 6), k_s])[::-1])
        dang = (ang_now - ang_prev + np.pi) % (2 * np.pi) - np.pi
        r = 0.55 * np.linalg.norm(Ej - Sj)
        if abs(dang) > 0.03:
            span = np.linspace(ang_now - np.sign(dang) * 1.1, ang_now, 24)
            arc = np.c_[Sj[0] + r * np.cos(span), Sj[1] + r * np.sin(span) * 0.9]
            arrow(lay, arc, YEL, 4, rv * ease((abs(dang) - 0.03) / 0.10), head=22)
        # 3) линия тяги широчайшей: от плеча вниз к тазу, бегущие штрихи
        ln = lines[s]; path = bez(ln['ins'], ln['mid'], ln['org'], 30)
        arrow(lay, path, YEL, 2, 0.9, head=20)
        pos = path[int(((t_phase * 0.9) % 1) * 29)]
        cv2.circle(lay, tuple((pos * 4).astype(int)), 7 * 4, (200, 250, 255), -1, cv2.LINE_AA, 2)
    return lay

def render(fr, idx, tt):
    p = P[idx]; c = C[idx]
    rv_red = ease((tt - 0.5) / 0.5); rv_gray = ease((tt - 0.85) / 0.5); rv_arr = ease((tt - 1.3) / 0.4)
    img = fr.astype(np.float32)
    if rv_red <= 0: return fr
    col, a, glow, lines, J = draw_anatomy(p, c, rv_red, rv_gray)
    rng = [j for j in range(idx - 2, idx + 3) if 0 <= j < len(P)]
    body = np.mean([cv2.imread(f'{dd}/{j:04d}.png', 0) for j in rng for dd in ('masks', 'masks2')], 0).astype(np.float32) / 255
    body = np.clip((body - 0.3) / 0.3, 0, 1)
    body = cv2.GaussianBlur(cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))), (0, 0), 3)
    aa = cv2.GaussianBlur(a, (0, 0), 1.2) * body * 0.93
    img = img * (1 - aa[..., None]) + col * aa[..., None]
    gl = cv2.GaussianBlur(glow * body, (0, 0), 7)
    gl2 = cv2.GaussianBlur(glow * body, (0, 0), 1.5)
    yel = np.float32(YEL)
    img = 255 - (255 - img) * (1 - (gl * 0.9)[..., None] * yel / 255)
    img = 255 - (255 - img) * (1 - np.clip(gl2 * 1.3, 0, 1)[..., None] * yel / 255)
    if rv_arr > 0:
        arr = draw_arrows(idx, lines, J, 1.0, tt).astype(np.float32)
        am = arr.max(2, keepdims=True) / 255 * rv_arr
        ag = cv2.GaussianBlur(arr, (0, 0), 6) * rv_arr
        img = 255 - (255 - img) * (1 - ag * 0.8 / 255)
        img = img * (1 - am) + arr * am
    return np.clip(img, 0, 255).astype(np.uint8)

def work(rng):
    cap = cv2.VideoCapture('seg.mp4'); cur = -1
    for idx in rng:
        while cur < idx: ok, fr = cap.read(); cur += 1
        out = render(fr, idx, (idx - START) / 30)
        cv2.imwrite(f'out3/{idx - START:04d}.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 95])

if __name__ == '__main__':
    os.makedirs('out3', exist_ok=True); cv2.setNumThreads(1)
    if len(sys.argv) > 1:
        ids = list(map(int, sys.argv[1:])); work(ids)
        cv2.imwrite('f/a3.jpg', np.hstack([cv2.resize(cv2.imread(f'out3/{i - START:04d}.jpg'), (432, 768)) for i in ids]))
    else:
        allr = list(range(START, END)); h = len(allr) // 2
        with Pool(2) as pl: pl.map(work, [allr[:h], allr[h:]])
        print('done')

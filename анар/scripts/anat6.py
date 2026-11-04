# Полный корсет спины: гладкие сплайн-контуры, три тона (красный — работают, оранжевый — помогают, серый — держат)
import numpy as np, cv2

def nrm(v): return v / (np.linalg.norm(v) + 1e-6)
def lerp(a, b, t): return a + (b - a) * t

def cr_closed(pts, n=14):
    """замкнутый сплайн Катмулла-Рома через точки — контур без углов"""
    P = np.asarray(pts, np.float32); k = len(P); out = []
    for i in range(k):
        p0, p1, p2, p3 = P[(i - 1) % k], P[i], P[(i + 1) % k], P[(i + 2) % k]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out, np.float32)

def cr_open(pts, n=16):
    P = np.asarray(pts, np.float32); P = np.vstack([P[0] * 2 - P[1], P, P[-1] * 2 - P[-2]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2]); return np.array(out, np.float32)

def fan(origin_curve, target, bend, n=22):
    """волокна веером: от линии начала к точке крепления"""
    o = cr_open(origin_curve, 8); idx = np.linspace(0, len(o) - 1, n).astype(int); fib = []
    for i in idx:
        a = o[i]; c = (a + target) / 2 + bend
        t = np.linspace(0, 1, 20)[:, None]; fib.append((1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * target)
    return fib

def parallel(poly, direction, step):
    """параллельные волокна по направлению, режутся маской мышцы"""
    d = nrm(direction); pv = np.array([-d[1], d[0]]); c = poly.mean(0)
    r = np.max(np.linalg.norm(poly - c, axis=1)) * 1.1; fib = []
    for q in np.arange(-r, r, step):
        a = c + pv * q - d * r; b = c + pv * q + d * r; fib.append(np.array([a, b]))
    return fib

def spindle(a, b, w0, w1, n=10, step=6):
    ax = nrm(b - a); px = np.array([-ax[1], ax[0]]); L = np.linalg.norm(b - a)
    t = np.linspace(0, 1, n); w = (np.sin(np.pi * t) ** 0.55) * (w0 * (1 - t) + w1 * t) + 1
    c = a + ax * (t[:, None] * L)
    pts = np.vstack([c + px * w[:, None], (c - px * w[:, None])[::-1]])
    return cr_closed(pts, 6), parallel(pts, ax, step)

def corset(p, c):
    S = {0: p[11], 1: p[12]}; E = {0: p[13], 1: p[14]}; Wr = {0: p[15], 1: p[16]}; Hh = {0: p[23], 1: p[24]}
    M = (S[0] + S[1]) / 2; Hm = (Hh[0] + Hh[1]) / 2
    v = nrm(Hm - M); hw = np.linalg.norm(S[0] - S[1]) / 2; sp = lambda t: lerp(M, Hm, t)
    st = max(4.0, hw / 30)              # шаг волокон
    bl = 0.05 * c; out = []; lines = {}
    def add(name, tone, poly, fib, z): out.append((z, name, tone, poly, fib))
    for s in (0, 1):
        u = nrm(S[s] - M); X = lambda t, k: sp(t) + u * k * hw      # точка: t вдоль позвоночника, k вбок
        arm = S[s] + (E[s] - S[s]) * 0.12 - u * 0.06 * hw + v * 0.10 * hw
        # 1 разгибатели спины (серые, низ спины между широчайшими)
        er = cr_closed([X(0.30, 0.02), X(0.36, 0.20), X(0.70, 0.27), X(0.98, 0.22), X(1.06, 0.06), X(0.70, 0.02)])
        add('erector', 'gray', er, parallel(er, v, st), 1)
        # 2 пояснично-грудная фасция (оранжевый треугольник внизу)
        fa = cr_closed([X(0.80, 0.02), X(0.86, 0.36), X(1.02, 0.62), X(1.08, 0.30), X(1.08, 0.02)])
        add('fascia', 'orange', fa, parallel(fa, u + v * 0.3, st * 1.3), 2)
        # 3 косые живота (оранжевые, бока талии)
        ob = cr_closed([X(0.56, 0.86), X(0.60, 1.02), X(0.86, 1.00), X(1.04, 0.78), X(0.95, 0.66), X(0.72, 0.74)])
        add('oblique', 'orange', ob, parallel(ob, v - u * 0.5, st), 2)
        # 4 подостная + малая круглая на лопатке (серые)
        inf = cr_closed([X(0.08, 0.30), X(0.04, 0.62), S[s] - u * 0.10 * hw + v * 0.14 * hw, X(0.30, 0.86), X(0.40, 0.58), X(0.30, 0.30)])
        add('infra', 'gray', inf, fan([X(0.10, 0.32), X(0.22, 0.32), X(0.36, 0.52)], S[s] - u * 0.08 * hw + v * 0.13 * hw, u * 0.03 * hw, 16), 3)
        # 5 большая круглая (красная)
        sc = X(0.34, 0.60)
        tm, tf = spindle(sc, arm, 0.16 * hw, 0.12 * hw, step=st)
        add('teres', 'red', tm, tf, 4)
        # 6 широчайшая (красная) — большой мягкий веер до самого позвоночника
        lat = cr_closed([arm, X(0.26, 0.36), X(0.36, 0.06), X(0.62, 0.03), X(0.90, 0.06), X(0.99, 0.22),
                         lerp(Hm, Hh[s], 0.80) - v * 0.02 * hw, X(0.80, 0.84 + bl), X(0.58, 1.00 + bl), X(0.38, 1.08 + bl), X(0.20, 0.98)], 16)
        add('lat', 'red', lat, fan([X(0.36, 0.06), X(0.62, 0.03), X(0.88, 0.06), X(0.99, 0.24), lerp(Hm, Hh[s], 0.8)], arm, u * (0.2 + bl) * hw, 30), 5)
        lines[s] = dict(ins=arm, mid=X(0.62, 0.75), org=lerp(Hm, Hh[s], 0.6) - v * 0.04 * hw, scap=X(0.22, 0.48))
        # 7 нижняя/средняя трапеция (оранжевая — сводит лопатки вниз)
        lt = cr_closed([X(0.06, 0.04), X(0.10, 0.46), X(0.18, 0.58), X(0.34, 0.20), X(0.52, 0.02)])
        add('lowtrap', 'orange', lt, fan([X(0.06, 0.02), X(0.30, 0.02), X(0.50, 0.02)], X(0.15, 0.55), -v * 0.02 * hw, 16), 6)
        # 8 задняя дельта (серая с оранжевым краем)
        ax = nrm(E[s] - S[s]); L = np.linalg.norm(E[s] - S[s]); cen = S[s] + ax * 0.05 * L + u * 0.20 * hw + v * 0.06 * hw
        dl = cr_closed([cen + ax * np.cos(a) * 0.30 * L + np.array([-ax[1], ax[0]]) * np.sin(a) * 0.26 * hw for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)], 8)
        add('delt', 'gray', dl, parallel(dl, ax, st), 8)
        # 9 плечо и предплечье (оранжевые — помогают тянуть)
        ua, uf = spindle(S[s] + (E[s] - S[s]) * 0.14, E[s] + (E[s] - S[s]) * 0.05, 0.26 * hw, 0.14 * hw, step=st)
        add('arm', 'orange', ua, uf, 9)
        fo, ff = spindle(E[s] - (Wr[s] - E[s]) * 0.02, Wr[s] - (Wr[s] - E[s]) * 0.07, 0.19 * hw, 0.09 * hw, step=st)
        add('fore', 'orange', fo, ff, 9)
    # 10 верхняя трапеция (серая, от шеи к плечам) — поверх
    u0 = nrm(S[0] - M); u1 = nrm(S[1] - M); X0 = lambda t, k: sp(t) + u0 * k * hw; X1 = lambda t, k: sp(t) + u1 * k * hw
    top = sp(-0.42)
    tr = cr_closed([top + u0 * 0.12 * hw, X0(-0.22, 0.30), S[0] - u0 * 0.06 * hw + v * 0.02 * hw, X0(0.14, 0.52), X0(0.22, 0.12),
                    X1(0.22, 0.12), X1(0.14, 0.52), S[1] - u1 * 0.06 * hw + v * 0.02 * hw, X1(-0.22, 0.30), top + u1 * 0.12 * hw], 12)
    fib = fan([top, sp(-0.1), sp(0.2)], S[0] - u0 * 0.1 * hw, -v * 0.05 * hw, 14) + fan([top, sp(-0.1), sp(0.2)], S[1] - u1 * 0.1 * hw, -v * 0.05 * hw, 14)
    add('trap', 'gray', tr, fib, 7)
    out.sort(key=lambda m: m[0])
    return out, lines, dict(S=S, E=E, hw=hw, M=M, Hm=Hm, v=v)

# палитра из рефа (BGR)
TONES = {
    'red':    dict(lo=np.float32([26, 27, 170]), mid=np.float32([28, 49, 217]), hi=np.float32([66, 104, 225]), fib=np.float32([75, 105, 240]), fa=0.32),
    'orange': dict(lo=np.float32([21, 70, 190]), mid=np.float32([30, 105, 220]), hi=np.float32([67, 136, 232]), fib=np.float32([95, 160, 240]), fa=0.30),
    'gray':   dict(lo=np.float32([128, 137, 145]), mid=np.float32([152, 162, 169]), hi=np.float32([178, 188, 194]), fib=np.float32([212, 218, 222]), fa=0.50),
}
EDGE = np.float32([21, 85, 205]); EDGE_HI = np.float32([67, 136, 225])
OPA = {'red': 0.80, 'orange': 0.70, 'gray': 0.62}

def layer(p, c, rv_red, rv_gray, S, W, H):
    Wd, Hd = W * S, H * S
    col = np.zeros((Hd, Wd, 3), np.float32); a = np.zeros((Hd, Wd), np.float32)
    glow = np.zeros((Hd, Wd), np.float32); seam = np.zeros((Hd, Wd), np.float32)
    ms, lines, J = corset(p, c)
    light = nrm(np.array([-0.4, -1.0]))
    for z, name, tone, poly, fibs in ms:
        rv = rv_red if tone == 'red' else rv_gray
        if rv <= 0: continue
        x0, y0 = np.maximum(poly.min(0).astype(int) - 10, 0); x1, y1 = np.minimum(poly.max(0).astype(int) + 10, [Wd, Hd])
        if x1 - x0 < 4 or y1 - y0 < 4: continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [((poly - [x0, y0]) * 4).astype(np.int32)], 255, cv2.LINE_AA, 2)
        dt = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5); dm = dt.max() + 1e-6; dn = (dt / dm)[..., None]
        # объём: купол + свет сверху-слева
        gy, gx = np.gradient(dt); sh = np.clip(0.5 + (gx * light[0] + gy * light[1]) * 0.6, 0, 1)[..., None]
        T = TONES[tone]
        body = T['lo'] + (T['mid'] - T['lo']) * np.clip(dn * 2.4, 0, 1)
        body = body + (T['hi'] - T['mid']) * np.clip((dn - 0.35) * 1.6, 0, 1) * (0.5 + 0.5 * sh)
        if tone == 'gray':
            e = np.clip(1 - dn * 3.0, 0, 1) ** 1.4
            body = body * (1 - e * 0.85) + (EDGE * (1 - dn) + EDGE_HI * dn) * e * 0.85
        elif tone == 'red':
            body = body * (1 - 0.25 * (1 - sh)) + T['lo'] * 0.25 * (1 - sh)
        fl = np.zeros_like(m)
        for f in fibs:
            cv2.polylines(fl, [((f - [x0, y0]) * 4).astype(np.int32)], False, 255, max(1, S), cv2.LINE_AA, 2)
        flf = (fl.astype(np.float32) / 255)[..., None] * T['fa']
        body = body * (1 - flf) + T['fib'] * flf
        # мягкое перо по краю (3 px), без ступенек
        feather = np.clip(dt / (3.0 * S), 0, 1)
        mf = (feather * (m.astype(np.float32) / 255))[..., None] * min(1.0, rv * 1.0)
        col[y0:y1, x0:x1] = col[y0:y1, x0:x1] * (1 - mf) + body * mf          # верхняя мышца перекрывает нижнюю чисто
        a[y0:y1, x0:x1] = a[y0:y1, x0:x1] * (1 - mf[..., 0]) + OPA[tone] * rv * mf[..., 0]
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((2 * S + 1, 2 * S + 1), np.uint8)).astype(np.float32) / 255
        if tone == 'red': glow[y0:y1, x0:x1] = np.maximum(glow[y0:y1, x0:x1], edge * rv)
        else: seam[y0:y1, x0:x1] = np.maximum(seam[y0:y1, x0:x1], edge * rv)
    return col, a, glow, seam, lines, J

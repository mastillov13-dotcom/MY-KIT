# v4: корсет приклеен к спине, цвета/линии из рефа, статичные стрелки, стоп-кадры с наездом и подписями
import cv2, json, numpy as np, os, sys
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '..')
import atlas as A  # анатомия (геометрия мышц)

W, H = 1080, 1920
V = np.array(json.load(open('pose_stab.json')))     # приклеенная поза
PR = np.array(json.load(open('pose_raw.json')))     # сырая сглаженная (для кистей/локтей на стопах)
N = len(V)
d = (PR[:, 15, 1] + PR[:, 16, 1]) / 2 - (PR[:, 11, 1] + PR[:, 12, 1]) / 2
C = gaussian_filter1d(np.clip((d + 340) / 340, 0, 1), 2)
CFG = json.load(open('cfg.json'))
REV0 = CFG['rev0']

# цвета из рефа (BGR)
RED = np.float32([28, 49, 217]); RED_HI = np.float32([66, 104, 225]); RED_LO = np.float32([26, 27, 170])
ORG = np.float32([21, 85, 205]); ORG_HI = np.float32([67, 136, 225])
GRY = np.float32([150, 160, 167]); GRY_LO = np.float32([128, 137, 145]); FIB_G = np.float32([210, 216, 220])
YEL = (49, 183, 233); YEL_GLOW = np.float32([115, 220, 249])
OP_RED, OP_GRAY = 0.80, 0.78

def body_mask(idx, S=1):
    rng = [j for j in range(idx - 2, idx + 3) if 0 <= j < N]
    m = np.mean([cv2.imread(f'{dd}/{j:04d}.png', 0) for j in rng for dd in ('ma', 'mb')], 0)
    m = (m > 110).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1: m = ((lab == 1 + np.argmax(st[1:, 4])) * 255).astype(np.uint8)
    ff = m.copy(); cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 255)
    m = m | cv2.bitwise_not(ff)                       # заливаем дыры
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 4)
    if S != 1: m = cv2.resize(m, (W * S, H * S), interpolation=cv2.INTER_LINEAR)
    return m

def anat_layer(p, c, rv_red, rv_gray, S=1):
    Wd, Hd = W * S, H * S
    col = np.zeros((Hd, Wd, 3), np.float32); a = np.zeros((Hd, Wd), np.float32)
    glow = np.zeros((Hd, Wd), np.float32); thin = np.zeros((Hd, Wd), np.float32)
    muscles, lines, J = A.anatomy(p, c)
    for name, poly, fibs, kind in muscles:
        rv = rv_red if kind == 'R' else rv_gray
        if rv <= 0: continue
        x0, y0 = np.maximum(poly.min(0).astype(int) - 8, 0); x1, y1 = np.minimum(poly.max(0).astype(int) + 8, [Wd, Hd])
        if x1 <= x0 or y1 <= y0: continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [((poly - [x0, y0]) * 4).astype(np.int32)], 255, cv2.LINE_AA, 2)
        dt = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5); dt = (dt / (dt.max() + 1e-6))[..., None]
        band = np.clip(1 - dt * 3.2, 0, 1)            # краевая полоса
        if kind == 'R':
            body = RED_LO + (RED - RED_LO) * np.clip(dt * 3, 0, 1)
            body = body + (RED_HI - RED) * np.clip((dt - 0.5) * 2, 0, 1) * (0.7 + 0.3 * c)
            body = body * (1 - band * 0.35) + ORG * band * 0.35
            fcol, fa = np.float32([70, 100, 240]), 0.30
        elif name in ('oblique', 'arm', 'fore'):
            body = ORG + (ORG_HI - ORG) * dt; fcol, fa = np.float32([90, 150, 235]), 0.3
        else:
            body = GRY_LO + (GRY - GRY_LO) * dt
            e = band ** 1.3
            body = body * (1 - e * 0.9) + (ORG * (1 - dt) + ORG_HI * dt) * e * 0.9
            fcol, fa = FIB_G, 0.5
        fl = np.zeros_like(m)
        for f in fibs:
            cv2.polylines(fl, [((f - [x0, y0]) * 4).astype(np.int32)], False, 255, max(1, S), cv2.LINE_AA, 2)
        flf = (fl.astype(np.float32) / 255)[..., None] * fa
        body = body * (1 - flf) + fcol * flf
        mf = (m.astype(np.float32) / 255)[..., None] * rv * (OP_RED if kind == 'R' else OP_GRAY)
        col[y0:y1, x0:x1] = col[y0:y1, x0:x1] * (1 - mf) + body * mf
        a[y0:y1, x0:x1] = a[y0:y1, x0:x1] * (1 - mf[..., 0]) + mf[..., 0]
        g = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((2 * S + 1, 2 * S + 1), np.uint8)).astype(np.float32) / 255
        if kind == 'R': glow[y0:y1, x0:x1] = np.maximum(glow[y0:y1, x0:x1], g * rv)
        else: thin[y0:y1, x0:x1] = np.maximum(thin[y0:y1, x0:x1], g * rv * 0.5)
    return col, a, glow, thin, lines, J

def static_arrows(p, S=1):
    """Статичные линии как в рефе: тонкие прямые с маленькими наконечниками + орбиты вокруг плеч."""
    lay = np.zeros((H * S, W * S, 3), np.uint8)
    Sx = {0: p[11], 1: p[12]}; Hh = {0: p[23], 1: p[24]}
    M = (Sx[0] + Sx[1]) / 2; Hm = (Hh[0] + Hh[1]) / 2
    v = A.nrm(Hm - M); hw = np.linalg.norm(Sx[0] - Sx[1]) / 2; sp = lambda t: A.lerp(M, Hm, t)
    th = 2 * S; hd = 13 * S
    def line(a, b, head=True):
        cv2.line(lay, tuple((a * 4).astype(int)), tuple((b * 4).astype(int)), YEL, th, cv2.LINE_AA, 2)
        if head:
            dv = A.nrm(b - a); pv = np.array([-dv[1], dv[0]])
            tri = np.array([b + dv * 2 * S, b - dv * hd + pv * hd * 0.45, b - dv * hd - pv * hd * 0.45])
            cv2.fillPoly(lay, [(tri * 4).astype(np.int32)], YEL, cv2.LINE_AA, 2)
    for s in (0, 1):
        u = A.nrm(Sx[s] - M)
        # длинная вертикаль снаружи руки: направление тяги вниз
        top = Sx[s] + u * 1.55 * hw - v * 1.9 * hw; bot = Sx[s] + u * 1.55 * hw + v * 0.25 * hw
        line(top, bot)
        # линия тяги широчайшей: от плеча к тазу
        ins = Sx[s] + u * 0.02 * hw + v * 0.35 * hw; org = A.lerp(Hm, Hh[s], 0.55) - v * 0.02 * hw
        line(ins, org)
        # диагональ от вертикали к плечу (как линия от кисти к груди в рефе)
        # орбита вокруг плечевого сустава
        ang = np.degrees(np.arctan2(u[1], u[0]))
        cen = tuple((Sx[s] * 4).astype(int)); axes = (int(0.62 * hw * 4), int(0.2 * hw * 4))
        cv2.ellipse(lay, cen, axes, ang, 200, 470, YEL, th, cv2.LINE_AA, 2)
    # центральная линия вдоль позвоночника вниз
    line(sp(0.05), sp(0.85))
    return lay

def compose(fr, idx, rv_red, rv_gray, rv_arr, S=1):
    p = V[idx] * S; c = C[idx]
    img = fr.astype(np.float32)
    col, a, glow, thin, lines, J = anat_layer(p, c, rv_red, rv_gray, S)
    body = body_mask(idx, S)
    aa = a * body
    img = img * (1 - aa[..., None]) + col * aa[..., None]
    g1 = cv2.GaussianBlur(glow * body, (0, 0), 6 * S); g2 = cv2.GaussianBlur(glow * body, (0, 0), 1.0 * S)
    img = 255 - (255 - img) * (1 - (g1 * 0.8)[..., None] * YEL_GLOW / 255)
    img = 255 - (255 - img) * (1 - np.clip(g2 * 1.4, 0, 1)[..., None] * np.float32(YEL) / 255)
    img = 255 - (255 - img) * (1 - (cv2.GaussianBlur(thin * body, (0, 0), 0.8 * S) * 0.6)[..., None])
    if rv_arr > 0:
        arr = static_arrows(p, S).astype(np.float32)
        am = arr.max(2, keepdims=True) / 255 * rv_arr
        ag = cv2.GaussianBlur(arr, (0, 0), 4 * S) * rv_arr
        img = 255 - (255 - img) * (1 - ag * 0.7 / 255 * YEL_GLOW / 255 * 1.6)
        img = img * (1 - am) + arr * am
    return np.clip(img, 0, 255).astype(np.uint8)

def rv_at(idx):
    t = (idx - REV0) / 30
    return A.ease(t / 0.5), A.ease((t - 0.35) / 0.5), A.ease((t - 0.8) / 0.4)

FB = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Bold.otf', 58)
FS = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf', 36)
def callout(img, target, title, sub, a, side):
    if a <= 0: return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
    ov = Image.new('RGBA', pil.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    A8 = int(255 * a); yel = (233, 183, 49, A8)
    x, y = map(int, target)
    tw = max(dr.textlength(title, font=FB), dr.textlength(sub, font=FS) if sub else 0)
    if y < H * 0.4:
        tx = x + 120 if side > 0 else x - 120 - tw
        tx = int(np.clip(tx, 50, W - 50 - tw)); ty = int(np.clip(y - 260, 140, H - 300))
        ex = tx if side > 0 else tx + tw
    else:   # подпись в тёмной зоне снизу, выноска вверх к точке
        tx = int((W - tw) / 2); ty = int(H * 0.79)
        ex = int(np.clip(x, tx + 20, tx + tw - 20))
    dr.line([(x, y), (ex, ty + 80 if y < H * 0.4 else ty - 24)], fill=yel, width=4)
    if y >= H * 0.4: dr.line([(tx, ty - 24), (tx + tw, ty - 24)], fill=yel, width=4)
    if y < H * 0.4: dr.line([(tx, ty + 80), (tx + tw, ty + 80)], fill=yel, width=4)
    dr.ellipse([x - 12, y - 12, x + 12, y + 12], outline=yel, width=4)
    dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=yel)
    for o in ((3, 3), (0, 4)):
        dr.text((tx + o[0], ty + o[1]), title, font=FB, fill=(0, 0, 0, int(A8 * 0.55)))
    dr.text((tx, ty), title, font=FB, fill=(255, 255, 255, A8))
    if sub:
        dr.text((tx + 2, ty + 98), sub, font=FS, fill=(0, 0, 0, int(A8 * 0.5)))
        dr.text((tx, ty + 96), sub, font=FS, fill=(249, 220, 115, A8))
    return cv2.cvtColor(np.array(Image.alpha_composite(pil, ov).convert('RGB')), cv2.COLOR_RGB2BGR)

def zoom_view(big, S, cx, cy, z):
    """big — кадр в S-кратном разрешении; окно с центром (cx,cy) в координатах 1080, масштаб z."""
    ww, hh = W / z, H / z
    x0 = np.clip(cx - ww / 2, 0, W - ww); y0 = np.clip(cy - hh / 2, 0, H - hh)
    Mx = np.float32([[z / S, 0, -x0 * z], [0, z / S, -y0 * z]])
    return cv2.warpAffine(big, Mx, (W, H), flags=cv2.INTER_AREA if z < S else cv2.INTER_CUBIC), (x0, y0)

def work(job):
    kind = job[0]
    if kind == 'live':
        _, ids = job; cap = cv2.VideoCapture('seg.mp4'); cur = -1
        for idx in ids:
            while cur < idx: ok, fr = cap.read(); cur += 1
            r, g, ar = rv_at(idx)
            out = fr if r <= 0 else compose(fr, idx, r, g, ar)
            cv2.imwrite(f'o4/live_{idx:04d}.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        _, k, f = job
        big = cv2.imread(f'hi_{k}.png')
        out = compose(big, f['idx'], 1, 1, 1, S=2)
        cv2.imwrite(f'o4/freeze_{k}.png', out)

if __name__ == '__main__':
    os.makedirs('o4', exist_ok=True); cv2.setNumThreads(1)
    from multiprocessing import Pool
    ids = list(range(CFG['start'], CFG['end']))
    jobs = [('live', ids[j * len(ids) // 2:(j + 1) * len(ids) // 2]) for j in range(2)]
    jobs += [('freeze', k, f) for k, f in enumerate(CFG['freezes'])]
    if len(sys.argv) > 1 and sys.argv[1] == 'freeze':
        for j in jobs[2:]: work(j)
    else:
        with Pool(2) as pl: pl.map(work, jobs)
    print('done')

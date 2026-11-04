# v6: полный гладкий корсет, медленные импульсы, 5 стопов; стрелки/подписи после выравнивания горизонта
import cv2, json, numpy as np, os, sys
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
import render4 as R
import anat6 as K
R.V = np.array(json.load(open('pose_axis.json')))   # корсет строго по оси
_fix = np.median(R.V[150:400], 0)                    # с момента посадки корсет неподвижен
# каркас по оси шеи: середина шеи по силуэту (медиана по кадрам), сдвигаем весь корсет на неё
_neck_dx = 557.0 - (_fix[11, 0] + _fix[12, 0]) / 2
_fix[[11, 12, 23, 24], 0] += _neck_dx
R.V[120:] = _fix
W, H = R.W, R.H; ease = R.A.ease; lerp = K.lerp; nrm = K.nrm
ARM = np.array(json.load(open('arms.json')))
TILT = 1.6
START, END = 30, 482
FREEZES = [
    dict(idx=60,  what='grip',  title='ХВАТ ЧУТЬ ШИРЕ ПЛЕЧ', sub='прямой хват сверху', z=1.4),
    dict(idx=208, what='spine', title='СПИНА РОВНАЯ', sub='корпус не раскачивай', z=1.25),
    dict(idx=290, what='lat',   title='ШИРОЧАЙШИЕ ТЯНУТСЯ', sub='руки вверх — спина вытягивается', z=1.4),
    dict(idx=322, what='elbow', title='ЛОКТИ ВНИЗ К РЁБРАМ', sub='тяни локтями, не кистями', z=1.6),
    dict(idx=336, what='scap',  title='ЛОПАТКИ ВНИЗ И ВМЕСТЕ', sub='сведи в нижней точке', z=1.5, pre='ГРУДНЫЕ ВВЕРХ'),
]
FL = 42

def pose(idx):
    p = R.V[idx].copy(); p[[13, 14, 15, 16]] = ARM[idx]; return p

wy = (ARM[:, 2, 1] + ARM[:, 3, 1]) / 2
ACT = np.clip(gaussian_filter1d(np.clip(np.r_[0, np.diff(wy)], 0, None), 3) / 4.0, 0, 1) ** 0.7
PH = np.cumsum(0.005 + 0.016 * ACT)          # импульсы медленнее

def force_path(p, lines, s):
    ln = lines[s]
    pts = [p[15 + s], p[13 + s], p[11 + s] + (p[13 + s] - p[11 + s]) * 0.1, ln['ins'], ln['mid'], ln['org']]
    path = K.cr_open(pts, 14); dl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return path, dl

def pulses(img, p, lines, phase, act, S):
    if act <= 0.02: return img
    core = np.zeros(img.shape[:2], np.float32)
    for s in (0, 1):
        path, dl = force_path(p, lines, s); Lt = dl[-1]
        for k in range(2):
            u = (phase + k / 2) % 1
            for j in range(16):
                uu = u - j * 0.010
                if uu < 0: continue
                q = np.array([np.interp(uu * Lt, dl, path[:, 0]), np.interp(uu * Lt, dl, path[:, 1])])
                fall = (1 - j / 16) ** 1.5
                cv2.circle(core, tuple((q * 4).astype(int)), int(max(2, (9 - j * 0.4) * S) * 4), fall, -1, cv2.LINE_AA, 2)
    glow = cv2.GaussianBlur(core, (0, 0), 9 * S)
    out = img.astype(np.float32)
    out = 255 - (255 - out) * (1 - (glow * 1.1 * act)[..., None] * R.YEL_GLOW / 255)
    out = 255 - (255 - out) * (1 - (np.clip(core, 0, 1) * act)[..., None] * np.float32([200, 245, 255]) / 255)
    return np.clip(out, 0, 255).astype(np.uint8)

def compose(fr, idx, rv_red, rv_gray, S=1, phase=0.0, act=0.0):
    p = pose(idx) * S; c = R.C[idx]
    col, a, glow, seam, lines, J = K.layer(p, c, rv_red, rv_gray, S, W, H)
    body = R.body_mask(idx, S)
    aa = a * body
    base = fr.astype(np.float32)
    mult = base * col / 255 * 1.35                     # умножение: цвет сочный, складки майки видны
    mix = col * 0.45 + np.clip(mult, 0, 255) * 0.55
    img = base * (1 - aa[..., None]) + mix * aa[..., None]
    pk = 0.55 + 0.45 * c
    g1 = cv2.GaussianBlur(glow * body, (0, 0), 7 * S); g2 = cv2.GaussianBlur(glow * body, (0, 0), 1.1 * S)
    img = 255 - (255 - img) * (1 - (g1 * 0.7 * pk)[..., None] * R.YEL_GLOW / 255)
    img = 255 - (255 - img) * (1 - np.clip(g2 * 1.1 * pk, 0, 1)[..., None] * np.float32(R.YEL) / 255)
    sm = cv2.GaussianBlur(seam * body, (0, 0), 0.8 * S)[..., None]
    img = img * (1 - sm * 0.35) + np.float32([225, 230, 235]) * sm * 0.35   # тонкие светлые швы
    img = np.clip(img, 0, 255).astype(np.uint8)
    return pulses(img, p, lines, phase, act, S)

ZL = np.cos(np.radians(TILT)) + (H / W) * np.sin(np.radians(TILT)) * 0.98
MLEV = cv2.getRotationMatrix2D((W / 2, H / 2), TILT, ZL)
def level(img): return cv2.warpAffine(img, MLEV, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
def lev_pt(q): q = np.asarray(q, np.float32).reshape(-1, 2); return (q @ MLEV[:, :2].T + MLEV[:, 2])

# ---------- графика стопов (в финальных координатах) ----------
def arrow(img, pts, a, double=False, th=7):
    if a <= 0: return img
    pts = np.asarray(pts, np.float32); lay = np.zeros_like(img)
    cv2.polylines(lay, [(pts * 4).astype(np.int32)], False, R.YEL, th, cv2.LINE_AA, 2)
    def head(tip, prev):
        dv = nrm(tip - prev); pv = np.array([-dv[1], dv[0]]); hs = 32
        tri = np.array([tip + dv * 5, tip - dv * hs + pv * hs * 0.6, tip - dv * hs - pv * hs * 0.6])
        cv2.fillPoly(lay, [(tri * 4).astype(np.int32)], R.YEL, cv2.LINE_AA, 2)
    head(pts[-1], pts[-3])
    if double: head(pts[0], pts[2])
    lf = lay.astype(np.float32); g = cv2.GaussianBlur(lf, (0, 0), 9)
    out = img.astype(np.float32); out = 255 - (255 - out) * (1 - g * a * 1.3 / 255 * R.YEL_GLOW / 255)
    m = lf.max(2, keepdims=True) / 255 * a
    return np.clip(out * (1 - m) + lf * m, 0, 255).astype(np.uint8)

FB = R.FB; FS = R.FS
CAP_Y = int(H * 0.80)
def caption(img, title, sub, targets, a, pre=''):
    """подпись стоит на месте внизу по центру; выноска — к точкам"""
    if a <= 0: return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
    ov = Image.new('RGBA', pil.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    A8 = int(255 * a); yel = (233, 183, 49, A8)
    tw = max(dr.textlength(title, font=FB), dr.textlength(sub, font=FS), dr.textlength(pre, font=FB) if pre else 0)
    tx = int((W - tw) / 2); ty = CAP_Y - (74 if pre else 0)
    # подложка-градиент для читаемости
    for i in range(160 + (74 if pre else 0)):
        dr.line([(0, ty - 50 + i), (W, ty - 50 + i)], fill=(0, 0, 0, int(A8 * 0.42 * np.sin(np.pi * i / (160 + (74 if pre else 0))))))
    dr.line([(tx, ty - 22), (tx + tw, ty - 22)], fill=yel, width=4)
    for (x, y) in targets:
        ex = int(np.clip(x, tx + 10, tx + tw - 10))
        dr.line([(x, y + 16), (ex, ty - 22)], fill=yel, width=3)
        dr.ellipse([x - 15, y - 15, x + 15, y + 15], outline=yel, width=4); dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=yel)
    if pre:
        dr.text((tx + 3, ty + 3), pre, font=FB, fill=(0, 0, 0, int(A8 * 0.5))); dr.text((tx, ty), pre, font=FB, fill=(255, 255, 255, A8))
        ty += 74
    dr.text((tx + 3, ty + 3), title, font=FB, fill=(0, 0, 0, int(A8 * 0.5))); dr.text((tx, ty), title, font=FB, fill=(255, 255, 255, A8))
    dr.text((tx + 2, ty + 80), sub, font=FS, fill=(0, 0, 0, int(A8 * 0.5))); dr.text((tx, ty + 78), sub, font=FS, fill=(249, 220, 115, A8))
    return cv2.cvtColor(np.array(Image.alpha_composite(pil, ov).convert('RGB')), cv2.COLOR_RGB2BGR)

def freeze_geo(f):
    """цели и стрелки в координатах 1080 (до зума и выравнивания)"""
    idx = f['idx']; p = pose(idx); _, lines, J = K.corset(p, R.C[idx])
    M, Hm, hw, v = J['M'], J['Hm'], J['hw'], J['v']
    rs = 0 if p[11, 0] > p[12, 0] else 1
    w = f['what']
    if w == 'grip':
        B = np.array(json.load(open('pb.json')))[idx, :, :2]
        hands = [B[[15, 17, 19]].mean(0), B[[16, 18, 20]].mean(0)]
        mid = (hands[0] + hands[1]) / 2
        return dict(tg=hands, cen=np.array([mid[0], 700.0]), kind='hline', a=hands)
    if w == 'spine':
        return dict(tg=[lerp(M, Hm, 0.45)], cen=lerp(M, Hm, 0.4), kind='vline', a=[M - v * 0.35 * hw, Hm - v * 0.05 * hw])
    if w == 'lat':
        u = nrm(p[11 + rs] - M); tg = lerp(p[11 + rs], p[23 + rs], 0.45) + u * 0.5 * hw
        return dict(tg=[tg], cen=lerp(tg, (M + Hm) / 2, 0.4), kind='vup', a=[M + u * 1.35 * hw + v * 0.05 * hw, Hm + u * 1.35 * hw - v * 0.1 * hw])
    if w == 'elbow':
        e = ARM[idx, rs]; ox = 1 if e[0] > M[0] else -1
        return dict(tg=[e], cen=lerp(e, (M + Hm) / 2, 0.35), kind='vdown', a=[e + [ox * 100, -150], e + [ox * 100, 130]])
    if w == 'scap':
        sc = [lines[0]['scap'], lines[1]['scap']]
        return dict(tg=sc, cen=lerp(M, Hm, 0.25), kind='converge', a=sc, M=M, v=v, hw=hw)

def freeze_frames(f, big):
    g = freeze_geo(f); frames = []
    for i in range(FL):
        t = i / (FL - 1)
        zi = ease(t / 0.2) * ease((1 - t) / 0.16)
        z = 1 + (f['z'] - 1) * zi + 0.035 * ease(t) * zi
        c_ = lerp(np.array([W / 2, H / 2]), g['cen'], zi)
        view, (x0, y0) = R.zoom_view(big, 2, c_[0], c_[1], z)
        view = level(view)
        mp = lambda q: lev_pt((np.asarray(q, np.float32).reshape(-1, 2) - [x0, y0]) * z)
        aa = ease((t - 0.14) / 0.1) * ease((1 - t - 0.14) / 0.08)
        grow = ease((t - 0.14) / 0.2)
        k = g['kind']; A_ = mp(g['a'])
        if k == 'hline':
            y = A_[:, 1].mean() + 85; xa, xb = sorted(A_[:, 0])
            view = arrow(view, np.linspace([xa + 25, y], [xb - 25, y], 10), aa, double=True)
        elif k == 'vline':
            x = A_[:, 0].mean(); ya, yb = A_[0, 1], A_[1, 1]
            lay = view.copy(); cv2.line(lay, (int(x), int(ya)), (int(x), int(ya + (yb - ya) * grow)), R.YEL, 5, cv2.LINE_AA)
            for yy in (ya, ya + (yb - ya) * grow): cv2.line(lay, (int(x - 26), int(yy)), (int(x + 26), int(yy)), R.YEL, 5, cv2.LINE_AA)
            view = cv2.addWeighted(lay, aa, view, 1 - aa, 0)
        elif k == 'vdouble':
            x = A_[:, 0].mean(); view = arrow(view, np.linspace([x, A_[1, 1]], [x, A_[0, 1]], 10), aa, double=True)
        elif k == 'vup':
            x = A_[:, 0].mean(); ya, yb = A_[1, 1], A_[0, 1]
            view = arrow(view, np.linspace([x, ya], [x, ya + (yb - ya) * max(0.15, grow)], 10), aa)
        elif k == 'vdown':
            x = A_[0, 0]; ya, yb = A_[0, 1], A_[1, 1]
            view = arrow(view, np.linspace([x, ya], [x, ya + (yb - ya) * max(0.15, grow)], 10), aa)
        elif k == 'converge':
            Mv = mp(g['M'])[0]
            for q in A_:
                side = np.sign(q[0] - Mv[0]); a0 = q + [side * 70, -40]; a1 = [Mv[0] + side * 40, q[1] + 70]
                ctrl = [(a0[0] + a1[0]) / 2 + side * 10, a0[1] + 10]
                tt = np.linspace(0, max(0.2, grow), 14)[:, None]
                pts = (1 - tt) ** 2 * a0 + 2 * (1 - tt) * tt * np.array(ctrl) + tt ** 2 * np.array(a1)
                view = arrow(view, pts, aa)
        ca = ease((t - 0.18) / 0.1) * ease((1 - t - 0.14) / 0.08)
        view = caption(view, f['title'], f['sub'], [tuple(q) for q in mp(g['tg'])], ca, f.get('pre', ''))
        frames.append(view)
    return frames

def work(job):
    if job[0] == 'live':
        cap = cv2.VideoCapture('seg.mp4'); cur = -1
        for idx in job[1]:
            while cur < idx: ok, fr = cap.read(); cur += 1
            r, g, _ = R.rv_at(idx)
            out = fr if r <= 0 else compose(fr, idx, r, g, 1, PH[idx], ACT[idx] * r)
            cv2.imwrite(f'o7/live_{idx:04d}.jpg', level(out), [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        f = job[1]; big = cv2.imread(f"hi6_{f['idx']}.png")
        r, g, _ = R.rv_at(f['idx'])
        if r > 0: big = compose(big, f['idx'], 1, 1, S=2)
        for i, fr in enumerate(freeze_frames(f, big)):
            cv2.imwrite(f"o7/fz_{f['idx']:04d}_{i:03d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 95])

if __name__ == '__main__':
    os.makedirs('o7', exist_ok=True); cv2.setNumThreads(1)
    from multiprocessing import Pool
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'test':
        work(('live', list(map(int, sys.argv[2:])))); sys.exit()
    if mode == 'fz':
        for f in FREEZES:
            if str(f['idx']) in sys.argv[2:] or len(sys.argv) == 2: work(('fz', f))
        sys.exit()
    if mode == 'all':
        ids = list(range(START, END))
        jobs = [('live', ids[j * len(ids) // 2:(j + 1) * len(ids) // 2]) for j in range(2)] + [('fz', f) for f in FREEZES]
        with Pool(2) as pl: pl.map(work, jobs)
    # сборка
    os.makedirs('seq7', exist_ok=True); n = 0; fzi = {f['idx']: f for f in FREEZES}
    for idx in range(START, END):
        for src in [f'o7/live_{idx:04d}.jpg'] + ([f'o7/fz_{idx:04d}_{i:03d}.jpg' for i in range(FL)] if idx in fzi else []):
            os.replace(src, f'seq7/{n:04d}.jpg') if False else cv2.imwrite(f'seq7/{n:04d}.jpg', cv2.imread(src)); n += 1
    print('frames', n, round(n / 30, 2))

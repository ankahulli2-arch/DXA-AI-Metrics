import time, cv2
import numpy as np
import pydicom, torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

CFG = Path(__file__).resolve().parent / "config"
def _rp(n, d=""):
    p = CFG / n
    if p.exists():
        s = p.read_text(encoding="utf-8").strip()
        if s: return Path(s)
    return Path(d or "__x__")

A_M = CFG / "cls.pt"
S_M = CFG / "unet_spine_mt_addDice.pt"
H_M = CFG / "hip_b0_v4.pt"
SRC = _rp("input.txt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
I = 300

SPA_PCT, ILA_PCT, ARA_PCT = 0.5, 0.5, 0.05
TRO_PCT, NEA_PCT, ISA_PCT, HAA_PCT = 0.5, 0.3, 0.3, 0.1
ANG, SEG, WIN, EDG, SCB, SCT, MRN = 5.0, 15, 3, 3, 3.0, 10.0, 5
ACL, RCL = 0.5, 0.5
PXY, PXX = 1.05, 0.60
ITH, IBH, ISH = 3.0, 2.0, 2.0
ROV, RUN = 2.0, 6.0

C_BLACK=(0,0,0); C_WHITE=(255,255,255); C_GRAY=(150,150,150); C_PURP=(128,113,173)
C_NAVY=(76,72,96); C_RED=(255,0,0); C_GREEN=(0,150,0); C_LAV=(179,195,248)
C_SAND=(250,223,173); C_BLUE=(100,149,237); C_PINK=(255,203,219); C_DARK=(56,0,50)

ANG_SC = 0.5
ANG_TH = 2
FN = cv2.FONT_HERSHEY_SIMPLEX
FIX_H = 700
IW = 550
SW = 480
PD = 12

TS = A.Compose([A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
TH = A.Compose([A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
TC = A.Compose([A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])


class CN(nn.Module):
    def __init__(s, n=2):
        super().__init__()
        from torchvision import models
        s.backbone = models.resnet18(weights=None)
        d = s.backbone.fc.in_features
        s.backbone.fc = nn.Identity()
        s.drop = nn.Dropout(0.5)
        s.fc = nn.Linear(d, n)
    def forward(s, x):
        return s.fc(s.drop(s.backbone(x)))
    def ff(s, x):
        return s.backbone(x)


class SN(nn.Module):
    def __init__(s):
        super().__init__()
        s.unet = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=4)
        d = s.unet.encoder.out_channels[-1]
        s.ribs_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(d, 2))
        s.art_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(d, 2))
    def forward(s, x):
        f = s.unet.encoder(x)
        d = s.unet.decoder(f)
        return s.unet.segmentation_head(d), s.ribs_head(f[-1]), s.art_head(f[-1])


class HN(nn.Module):
    def __init__(s):
        super().__init__()
        s.unet = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=5)
        d = s.unet.encoder.out_channels[-1]
        s.art_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(d, 2))
    def forward(s, x):
        f = s.unet.encoder(x)
        d = s.unet.decoder(f)
        return s.unet.segmentation_head(d), s.art_head(f[-1])


def _ld(p, m):
    c = torch.load(p, map_location="cpu", weights_only=False)
    st = c.get("model_state_dict", c) if isinstance(c, dict) else c
    m.load_state_dict(st)
    m = m.to(DEV).eval()
    return m.half() if DEV == "cuda" else m


class CLS:
    def __init__(s, p):
        c = torch.load(p, map_location=DEV, weights_only=False)
        s.m = CN(c.get("num_classes", 2)).to(DEV).eval()
        s.m.load_state_dict(c["model_state_dict"])
        o = c["ood"]
        s.mu = o["mu"].cpu().numpy().astype(np.float32)
        s.ci = o["cov_inv"].cpu().numpy().astype(np.float32)
        s.th = float(o.get("threshold_maha", o["threshold"]))
        s.ts = float(o.get("threshold_soft", 0.0))
        s.tc = float(o.get("threshold_cos", -1.0))
        s.c0 = o["centroid0"].cpu().numpy().astype(np.float32)
        s.c1 = o["centroid1"].cpu().numpy().astype(np.float32)
        s.T = float(o.get("temperature", 2.0))
    @torch.no_grad()
    def __call__(s, img):
        x = TC(image=img)["image"].unsqueeze(0).to(DEV)
        f = s.m.ff(x)[0].cpu().numpy()
        lo = s.m(x).float()
        pr = F.softmax(lo, 1)[0].cpu().numpy()
        pt = F.softmax(lo / s.T, 1)[0].cpu().numpy()
        pm = float(np.max(pt))
        pm_raw = float(np.max(pr))
        d = (f - s.mu).astype(np.float32)
        sc = float(np.sum((d @ s.ci) * d))
        n = f / (np.linalg.norm(f) + 1e-8)
        cs = float(max(n @ s.c0, n @ s.c1))
        if sc > s.th or cs < s.tc or pm < s.ts:
            return -1, pm_raw
        return int(np.argmax(pr)), pm_raw


def _dcm(p):
    d = pydicom.dcmread(str(p))
    a = d.pixel_array.astype(np.float32)
    a = a - a.min()
    if a.max() > 0: a = a / a.max()
    a = (a * 255).astype(np.uint8)
    if a.ndim == 2: a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    full = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    small = cv2.resize(full, (I, I), interpolation=cv2.INTER_LINEAR)
    return full, small


def _up(m, w, h):
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _cc(m, cid, am):
    b = (m == cid).astype(np.uint8)
    if not b.sum(): return m
    n, l, st, _ = cv2.connectedComponentsWithStats(b, 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < am: m[l == i] = 0
    return m


def _no(m, cid):
    n, _, _, _ = cv2.connectedComponentsWithStats((m == cid).astype(np.uint8), 8)
    return max(0, n - 1)


def _area_mm2(mask, cid, px, py):
    return float((mask == cid).sum()) * px * py


def _mb(m):
    n, l, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return m if n <= 1 else (l == int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1).astype(np.uint8)


def _iliac_comps(m, total_px):
    b = (m == 2).astype(np.uint8)
    n, l, st, _ = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        blob = (l == i).astype(np.uint8)
        cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs: continue
        per = float(sum(cv2.arcLength(c, True) for c in cs))
        out.append({"area_px": a, "area_pct": 100.0 * a / total_px, "perim": per})
    out.sort(key=lambda c: -c["area_px"])
    return out


def _rc(b):
    ys = np.nonzero(b.any(1))[0]
    if len(ys) < 30: return None, None
    c = np.arange(b.shape[1], dtype=np.float32)
    s = b[ys].astype(np.float32); d = s.sum(1); v = d > 0
    ys, s, d = ys[v], s[v], d[v]
    if len(ys) < 30: return None, None
    return ys.astype(np.float32), ((s * c).sum(1) / d).astype(np.float32)


def _sg(x, y, L):
    n = len(x); k = max(3, n // L); ll = n // k
    ax = np.zeros(k, np.float32); ay = np.zeros(k, np.float32)
    for i in range(k):
        a = i * ll; b = (i + 1) * ll if i < k - 1 else n
        ay[i] = y[a:b].mean(); ax[i] = x[a:b].mean()
    return ax, ay, k


def _sl(x, y):
    if len(x) < 2: return 0.0
    dy = y - y.mean(); dd = (dy * dy).sum()
    if dd < 1e-6: return 0.0
    return float(np.degrees(np.arctan(float((dy * (x - x.mean())).sum() / dd))))


def _lr(a, s):
    b = c = 0
    for v in a:
        sg = 1 if v > 0 else (-1 if v < 0 else 0)
        if sg == s:
            c += 1
            if c > b: b = c
        else: c = 0
    return b


def _sp_geom(m, skip):
    e = {"ok": False, "ang": 0.0, "sc": False, "cb": 0.0, "tt": 0.0, "pA": (0,0), "pB": (0,0), "pC": (0,0)}
    b = (m == 1).astype(np.uint8)
    if b.sum() == 0: return e
    ys, xs = _rc(b)
    if ys is None or len(ys) < 30: return e
    n = len(ys); bd = max(3, n // 10)
    xt, yt = float(xs[:bd].mean()), float(ys[:bd].mean())
    xb, yb = float(xs[-bd:].mean()), float(ys[-bd:].mean())
    ang = float(np.degrees(np.arctan2(xb - xt, yb - yt)))
    pC = (int(xt), int(yt)); pB = (int(xb), int(yb)); pA = (int(pB[0]), int(pB[1] - (yb - yt)))
    r = {"ok": True, "ang": ang, "sc": False, "cb": 0.0, "tt": 0.0, "pA": pA, "pB": pB, "pC": pC}
    if skip: return r
    ax, ay, k = _sg(xs, ys, SEG)
    if k < EDG * 2 + 4: return r
    h = WIN // 2
    la = np.zeros(k, np.float32)
    for i in range(k):
        a = max(0, i - h); b2 = min(k, i + h + 1)
        la[i] = _sl(ax[a:b2], ay[a:b2])
    mid = k // 2
    if mid <= EDG or k - EDG <= mid: return r
    t = la[EDG:mid]; bo = la[mid:k - EDG]
    tm = float(t[int(np.argmax(np.abs(t)))]); bm = float(bo[int(np.argmax(np.abs(bo)))])
    cb = tm - bm; tt = 0.5 * (tm + bm)
    ro = (_lr(la, -1) >= MRN) or (_lr(la, 1) >= MRN)
    r["sc"] = bool(ro and (abs(cb) > SCB or abs(tt) > SCT))
    r["cb"] = float(cb); r["tt"] = float(tt)
    return r


def _hp_geom(m):
    g = {"ok": False, "pos": False, "ht": False, "hn": False, "hi": False, "rot": "unknown",
         "hum": 0.0, "hum_rel": 0.0, "ang": 0.0, "ang_ok": False, "pA": None, "pB": None, "pC": None,
         "bG": None, "bL": None, "cL": None}
    t = (m == 1).astype(np.uint8)
    n = (m == 2).astype(np.uint8)
    i = (m == 3).astype(np.uint8)
    if t.sum() == 0:
        return g
    t = _mb(t)
    n = _mb(n) if n.sum() > 0 else n
    i = _mb(i) if i.sum() > 0 else i
    g["ht"] = True
    tot = m.size
    g["hn"] = (100.0 * n.sum() / tot) >= NEA_PCT
    g["hi"] = (100.0 * i.sum() / tot) >= ISA_PCT
    g["pos"] = g["ht"] and g["hn"] and g["hi"]

    yt, xt = np.where(t > 0)
    yn, xn = np.where(n > 0)

    right = True
    if len(xn) >= 10 and len(xt) >= 10:
        yc, xc = np.where((t > 0) & (n > 0))
        right = (float(xc.mean()) > float(xn.mean())) if len(xc) >= 5 else (float(xt.mean()) > float(xn.mean()))

    if not g["hn"]:
        return g

    ytn, ybn = int(yn.min()), int(yn.max())
    hn = max(1, ybn - ytn)
    ma = yn <= ytn + int(0.25 * hn)
    if ma.sum() > 0:
        ia = int(np.argmin(xn[ma])) if right else int(np.argmax(xn[ma]))
        A = (int(xn[ma][ia]), int(yn[ma][ia]))
    else:
        ia = int(np.argmin(xn)) if right else int(np.argmax(xn))
        A = (int(xn[ia]), int(yn[ia]))

    td = cv2.dilate(t, np.ones((7, 7), np.uint8), 1)
    yc, xc = np.where((n > 0) & (td > 0))
    if len(xc) > 0:
        ymc = (int(yc.min()) + int(yc.max())) // 2
        ib = int(np.argmin(np.abs(yc - ymc)))
        B = (int(xc[ib]), int(yc[ib]))
    else:
        B = (int(xn.mean()), int(yn.mean()))

    ybt = int(yt.max())
    ycl = max(int(yt.min()), ybt - int(0.05 * max(1, ybt - int(yt.min()))))
    mc = yt >= ycl
    if mc.sum() > 0:
        C = (int(xt[mc].mean()), int(yt[mc].mean()))
    else:
        rb = np.where(t[ybt, :] > 0)[0]
        C = (int(rb.mean()), ybt) if len(rb) > 0 else (int(xt.mean()), ybt)

    g["pA"], g["pB"], g["pC"] = A, B, C
    v1 = np.array([A[0] - B[0], A[1] - B[1]], float)
    v2 = np.array([C[0] - B[0], C[1] - B[1]], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 > 1e-6 and n2 > 1e-6:
        ad = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))
        g["ang"] = 180 - ad if ad < 90 else ad
        g["ang_ok"] = True

    yt0, yb0 = int(yt.min()), int(yt.max())
    ht = max(1, yb0 - yt0)
    y_cut = yt0 + int(0.30 * ht)
    uma = yt <= y_cut
    if uma.sum() >= 10:
        xsu = xt[uma]
        ysu = yt[uma]
        if right:
            x_lat = int(xsu.max())
            x_med = int(xsu.min())
        else:
            x_lat = int(xsu.min())
            x_med = int(xsu.max())
        wu = abs(x_med - x_lat)
        x_med_use = x_lat + int(0.60 * wu) * (1 if x_med > x_lat else -1)
        bx0 = min(x_lat, x_med_use)
        bx1 = max(x_lat, x_med_use)
        by0 = int(ysu.min())
        by1 = int(ysu.max())
        g["bG"] = (bx0, by0, max(5, bx1 - bx0), max(5, by1 - by0))

    yr = int(yn.max())
    ylo = max(yt0, yr)
    yhi = min(yb0 - 3, yr + int(0.40 * ht))
    if yhi > ylo + 5:
        pr = {}
        for yy in range(ylo, yhi + 1):
            xs = np.where(t[yy, :] > 0)[0]
            if len(xs):
                pr[yy] = int(xs.min()) if right else int(xs.max())
        if len(pr) >= 7:
            ks = sorted(pr.keys())
            vv = np.array([pr[k] for k in ks], np.float32)
            sm = np.convolve(vv, np.ones(7, np.float32) / 7.0, "same")
            bi = (int(np.argmin(sm[3:-3])) if right else int(np.argmax(sm[3:-3]))) + 3
            py = ks[bi]
            px = pr[py]
            bl = float(np.median(sm))
            pv = abs(sm[bi] - bl)
            th = max(2.0, pv * 0.4)
            ya = py
            for j in range(bi, -1, -1):
                if abs(sm[j] - bl) < th:
                    ya = ks[j]
                    break
            yb = py
            for j in range(bi, len(ks)):
                if abs(sm[j] - bl) < th:
                    yb = ks[j]
                    break
            sy = max(8, yb - ya)
            sxx = max(10, int(round(pv * 1.6)))
            cy = (ya + yb) // 2
            cx = px + int(0.15 * sxx) if right else px - int(0.15 * sxx)
            g["bL"] = (max(0, cx - sxx // 2), max(0, cy - sy // 2), sxx, sy)

            yl2, yh2 = max(yt0, ya - 2), min(yb0, yb + 2)
            st = np.zeros_like(t)
            st[yl2:yh2 + 1, :] = t[yl2:yh2 + 1, :]
            cs, _ = cv2.findContours(st, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs:
                bg = max(cs, key=cv2.contourArea).reshape(-1, 2)
                od = np.argsort(bg[:, 0]) if right else np.argsort(-bg[:, 0])
                tk = bg[od][:max(4, len(bg) // 4)]
                if len(tk) >= 3:
                    g["cL"] = tk.astype(np.int32)
            cs, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs:
                cnt = max(cs, key=cv2.contourArea).reshape(-1, 2)
                sp = cnt[cnt[:, 0] < cnt[:, 0].mean()] if right else cnt[cnt[:, 0] > cnt[:, 0].mean()]
                if len(sp) >= 10:
                    sp = sp[np.argsort(sp[:, 1])]
                    yt1, yb1 = int(sp[:, 1].min()), int(sp[:, 1].max())
                    ym = (yt1 + yb1) // 2
                    hr = max(5, (yb1 - yt1) // 4)
                    bl2, bh2 = max(yt1, ym - hr), min(yb1, ym + hr)
                    bp = sp[(sp[:, 1] >= bl2) & (sp[:, 1] <= bh2)]
                    if len(bp) >= 5:
                        ysb = bp[:, 1].astype(np.float32)
                        xsb = bp[:, 0].astype(np.float32)
                        blx = float(np.median(xsb))
                        dv = (blx - xsb) if right else (xsb - blx)
                        g["hum"] = float(dv[int(np.argmax(dv))])
    hp = g["hum"]
    g["hum_rel"] = 100.0 * hp / max(1, ht)
    g["rot"] = "over" if g["hum_rel"] < ROV else ("normal" if g["hum_rel"] < RUN else "under")
    g["ok"] = True
    return g


def _hp_ia(m):
    t = (m == 1).astype(np.uint8)
    i = (m == 3).astype(np.uint8)
    if not t.sum():
        return {"ok": False}
    yt, xt = np.where(t > 0)
    ytop = int(yt.min())
    if i.sum():
        yi, _ = np.where(i > 0)
        ybot = int(yi.max())
    else:
        ybot = int(yt.max())
    h, w = m.shape

    n = (m == 2).astype(np.uint8)
    yn, xn = np.where(n > 0)
    if len(xn) > 0 and len(xt) > 0:
        right = float(xn.mean()) < float(xt.mean())
    else:
        right = float(xt.mean()) > w / 2.0

    yt0, yb0 = int(yt.min()), int(yt.max())
    ht = max(1, yb0 - yt0)
    y_cut = yt0 + int(0.30 * ht)
    top_mask = yt <= y_cut
    if top_mask.sum() > 0:
        xs_top = xt[top_mask]
        ys_top = yt[top_mask]
        if right:
            idx_edge = int(np.argmax(xs_top))
            xe = int(xs_top[idx_edge])
        else:
            idx_edge = int(np.argmin(xs_top))
            xe = int(xs_top[idx_edge])
        y_at_edge = int(ys_top[idx_edge])
    else:
        if right:
            xe = int(xt.max())
            idx_edge = int(np.argmax(xt))
        else:
            xe = int(xt.min())
            idx_edge = int(np.argmin(xt))
        y_at_edge = int(yt[idx_edge])

    x_at_top = int(np.mean(xt[yt == ytop])) if (yt == ytop).any() else int(xt.mean())
    if i.sum():
        yi2, xi2 = np.where(i > 0)
        x_at_bot = int(np.mean(xi2[yi2 == ybot])) if (yi2 == ybot).any() else int(xi2.mean())
    else:
        x_at_bot = int(np.mean(xt[yt == ybot])) if (yt == ybot).any() else int(xt.mean())

    mt = ytop * PXY / 10.0
    mb = (h - 1 - ybot) * PXY / 10.0
    ms = ((w - 1 - xe) if right else xe) * PXX / 10.0

    ok_top = mt >= ITH
    ok_bottom = mb >= IBH
    ok_side = ms >= ISH
    return {"ok": True, "area_ok": ok_top and ok_bottom and ok_side,
            "mt": mt, "mb": mb, "ms": ms, "right": right,
            "y_top": ytop, "x_at_top": x_at_top, "y_bot": ybot, "x_at_bot": x_at_bot,
            "x_edge": xe, "y_at_edge": y_at_edge,
            "ok_top": ok_top, "ok_bottom": ok_bottom, "ok_side": ok_side}


@torch.no_grad()
def _inf(m, img, spine):
    h, w = img.shape[:2]
    tf = TS if spine else TH
    x = tf(image=img)["image"].unsqueeze(0).to(DEV)
    if DEV == "cuda": x = x.half()
    with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
        out = m(x)
    if spine:
        sg, rl, al = out
        rp = float(F.softmax(rl.float(), 1)[0].cpu().numpy()[1])
        ap = float(F.softmax(al.float(), 1)[0].cpu().numpy()[1])
    else:
        sg, al = out
        rp = 0.0
        ap = float(F.softmax(al.float(), 1)[0].cpu().numpy()[1])
    sp = F.softmax(sg.float(), 1)[0].cpu().numpy()
    pr = _up(np.argmax(sp, 0).astype(np.uint8), w, h)
    return pr, rp, ap


def _mask_overlay(img, mask, cols):
    o = img.copy()
    for cid, c in cols.items():
        m = (mask == cid)
        if m.any():
            ca = np.array(c, np.uint8)
            o[m] = (o[m] * 0.5 + ca * 0.5).astype(np.uint8)
    for cid, c in cols.items():
        cs, _ = cv2.findContours((mask == cid).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cs, -1, c, 1)
    return o


def _lbl(c, txt, x, y, col, sc=0.5, th=1):
    (tw, thh), _ = cv2.getTextSize(txt, FN, sc, th)
    x = max(2, min(c.shape[1] - tw - 4, int(x))); y = max(thh + 4, int(y))
    cv2.putText(c, txt, (x, y), FN, sc, C_BLACK, th + 2, cv2.LINE_AA)
    cv2.putText(c, txt, (x, y), FN, sc, col, th, cv2.LINE_AA)


def _angle_box(c, txt, x, y):
    sc, th = ANG_SC, ANG_TH
    (tw, thh), _ = cv2.getTextSize(txt, FN, sc, th)
    pad = 6
    x0 = max(0, int(x - tw // 2 - pad)); y0 = max(0, int(y - thh - pad))
    x1 = min(c.shape[1] - 1, int(x + tw // 2 + pad)); y1 = min(c.shape[0] - 1, int(y + pad))
    sub = c[y0:y1, x0:x1].astype(np.float32)
    bg = np.array(C_LAV, np.float32)
    c[y0:y1, x0:x1] = (sub * 0.55 + bg * 0.45).astype(np.uint8)
    cv2.rectangle(c, (x0, y0), (x1, y1), C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(c, txt, (x0 + pad, y1 - pad), FN, sc, C_WHITE, th, cv2.LINE_AA)
    return c


def _draw_spine_axis(c, g, sx, sy):
    if not g["ok"]: return c
    pA = (int(g["pA"][0] * sx), int(g["pA"][1] * sy))
    pB = (int(g["pB"][0] * sx), int(g["pB"][1] * sy))
    pC = (int(g["pC"][0] * sx), int(g["pC"][1] * sy))
    cv2.line(c, pB, pA, C_LAV, 1, cv2.LINE_AA)
    cv2.line(c, pB, pC, C_LAV, 1, cv2.LINE_AA)
    _angle_box(c, f"{abs(g['ang']):.1f}°", pB[0] + 40, pB[1] - 20)
    return c


def _draw_shdu_angle(c, g, sx, sy):
    if not g.get("ang_ok"): return c
    A_pt, B_pt, C_pt = g["pA"], g["pB"], g["pC"]
    if A_pt is None or B_pt is None or C_pt is None: return c
    A = (int(A_pt[0] * sx), int(A_pt[1] * sy))
    B = (int(B_pt[0] * sx), int(B_pt[1] * sy))
    C = (int(C_pt[0] * sx), int(C_pt[1] * sy))
    cv2.line(c, B, A, C_LAV, 2, cv2.LINE_AA)
    cv2.line(c, B, C, C_LAV, 2, cv2.LINE_AA)
    a_ba = float(np.degrees(np.arctan2(A[1] - B[1], A[0] - B[0])))
    a_bc = float(np.degrees(np.arctan2(C[1] - B[1], C[0] - B[0])))
    a1, a2 = min(a_ba, a_bc), max(a_ba, a_bc)
    if a2 - a1 > 180.0: a1, a2 = a2, a1 + 360.0
    r = 50
    cv2.ellipse(c, B, (r, r), 0, a1, a2, C_LAV, 2, cv2.LINE_AA)
    mid = np.radians((a1 + a2) / 2.0)
    tx = int(B[0] + np.cos(mid) * (r + 50)); ty = int(B[1] + np.sin(mid) * (r + 50))
    _angle_box(c, f"{g['ang']:.1f}°", tx, ty)
    for p in (A, B, C):
        cv2.circle(c, p, 4, C_BLACK, -1, cv2.LINE_AA); cv2.circle(c, p, 3, C_LAV, -1, cv2.LINE_AA)
    return c


def _draw_hip_interest(c, ia, sx, sy):
    if not ia.get("ok"): return c
    H, W = c.shape[:2]
    y_top = int(ia["y_top"] * sy); x_top = int(ia["x_at_top"] * sx)
    y_bot = int(ia["y_bot"] * sy); x_bot = int(ia["x_at_bot"] * sx)
    x_edge = int(ia["x_edge"] * sx); y_edge = int(ia["y_at_edge"] * sy)
    c1 = C_GREEN if ia["ok_top"] else C_RED
    c2 = C_GREEN if ia["ok_bottom"] else C_RED
    c3 = C_GREEN if ia["ok_side"] else C_RED
    cv2.arrowedLine(c, (x_top, y_top - 10), (x_top, 5), c1, 2, cv2.LINE_AA, tipLength=0.03)
    cv2.arrowedLine(c, (x_bot, y_bot + 10), (x_bot, H - 5), c2, 2, cv2.LINE_AA, tipLength=0.03)
    if ia["right"]:
        cv2.arrowedLine(c, (x_edge + 10, y_edge), (W - 5, y_edge), c3, 2, cv2.LINE_AA, tipLength=0.03)
        lx_side = W - 90
    else:
        cv2.arrowedLine(c, (x_edge - 10, y_edge), (5, y_edge), c3, 2, cv2.LINE_AA, tipLength=0.03)
        lx_side = 10
    _lbl(c, f"{ia['mt']:.2f} cm", x_top + 8, 20, c1, 0.55, 2)
    _lbl(c, f"{ia['mb']:.2f} cm", x_bot + 8, H - 12, c2, 0.55, 2)
    _lbl(c, f"{ia['ms']:.2f} cm", lx_side, y_edge - 10, c3, 0.55, 2)
    return c


def _draw_hip_geom(c, g, sx, sy):
    if g.get("bG"):
        gx, gy, gw, gh = g["bG"]
        x0 = int(gx * sx); y0 = int(gy * sy)
        x1 = int((gx + gw) * sx); y1 = int((gy + gh) * sy)
        cv2.rectangle(c, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), C_BLACK, 3, cv2.LINE_AA)
        cv2.rectangle(c, (x0, y0), (x1, y1), C_LAV, 2, cv2.LINE_AA)
        _lbl(c, "greater troch.", x0, y0 - 6, C_LAV, 0.5, 1)
    if g.get("cL") is not None and len(g["cL"]) >= 3:
        pts = np.array([[int(p[0] * sx), int(p[1] * sy)] for p in g["cL"]], np.int32)
        cv2.polylines(c, [pts], False, C_BLACK, 5, cv2.LINE_AA)
        cv2.polylines(c, [pts], False, C_LAV, 2, cv2.LINE_AA)
        _lbl(c, "lesser troch.", int(pts[:, 0].min()), int(pts[:, 1].min()) - 6, C_LAV, 0.5, 1)
    elif g.get("bL"):
        gx, gy, gw, gh = g["bL"]
        x0 = int(gx * sx); y0 = int(gy * sy)
        x1 = int((gx + gw) * sx); y1 = int((gy + gh) * sy)
        cv2.rectangle(c, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), C_BLACK, 3, cv2.LINE_AA)
        cv2.rectangle(c, (x0, y0), (x1, y1), C_LAV, 2, cv2.LINE_AA)
        _lbl(c, "lesser troch.", x0, y0 - 6, C_LAV, 0.5, 1)
    return c


def _sidebar(c, i, n, fn, ts, ri, kind, data):
    x, y = PD, 26
    W = c.shape[1]
    def line(yy, col=C_NAVY, th=2): cv2.line(c, (x, yy), (W - PD, yy), col, th, cv2.LINE_AA)
    def txt(s, xx, yy, sc=0.44, col=C_BLACK, th=1): cv2.putText(c, s, (int(xx), int(yy)), FN, sc, col, th, cv2.LINE_AA)
    cv2.putText(c, f"{i+1}/{n}", (x, y), FN, 0.5, C_BLACK, 2, cv2.LINE_AA); y += 16
    sn = fn[:50] + "..." if len(fn) > 53 else fn
    txt(sn, x, y, 0.34, C_GRAY, 1); y += 16
    line(y); y += 16
    cv2.putText(c, "TIME", (x, y), FN, 0.52, C_NAVY, 2, cv2.LINE_AA); y += 20
    t_open, t_cls, t_cnn, t_post = ts
    txt(f"open = {t_open*1000:6.2f} ms, {t_open:7.4f} s  | post = {t_post*1000:6.2f} ms, {t_post:7.4f} s", x, y, 0.4, C_PURP, 1); y += 17
    txt(f"cls  = {t_cls*1000:6.2f} ms, {t_cls:7.4f} s  | mltH = {t_cnn*1000:6.2f} ms, {t_cnn:7.4f} s", x, y, 0.4, C_PURP, 1); y += 17
    txt(f"TOTAL = {sum(ts)*1000:6.2f} ms, {sum(ts):7.4f} s", x, y, 0.44, C_PURP, 2); y += 20
    line(y); y += 16
    cv2.putText(c, "CLASSIFIER", (x, y), FN, 0.52, C_NAVY, 2, cv2.LINE_AA); y += 20
    txt(f"region = {ri[0]}  {ri[1]*100:.1f}%", x, y, 0.46, C_BLACK, 1); y += 20
    if kind == "spine":
        rp, ap, iliac_comps, mask_lines = data
        line(y); y += 16
        txt(f"RIBS CLASSIFIER (Th12)  acc = {rp*100:.1f}%", x, y, 0.48, C_NAVY, 2); y += 22
        txt(f"ARTIFACTS CLASSIFIER  acc = {ap*100:.1f}%", x, y, 0.48, C_NAVY, 2); y += 22
        line(y); y += 16
        cv2.putText(c, "ILIAC BONES", (x, y), FN, 0.52, C_NAVY, 2, cv2.LINE_AA); y += 20
        if not iliac_comps:
            txt("  not found", x, y, 0.44, C_BLACK, 1); y += 16
        else:
            for k, cm in enumerate(iliac_comps, 1):
                txt(f"{k}. area: {cm['area_px']} px ({cm['area_pct']:.2f}%) / perim: {cm['perim']:.1f}", x, y, 0.42, C_BLACK, 1); y += 16
        line(y); y += 16
        cv2.putText(c, "MASK STATS", (x, y), FN, 0.52, C_NAVY, 2, cv2.LINE_AA); y += 20
        for l in mask_lines:
            txt(l, x, y, 0.44, C_BLACK, 1); y += 16
        line(y); y += 16
        cv2.putText(c, "ПРЕДИКТ:", (x, y), FN, 0.55, C_NAVY, 2, cv2.LINE_AA); y += 22
    elif kind == "hip":
        ap, mask_lines = data
        line(y); y += 16
        txt(f"ARTIFACTS CLASSIFIER  acc = {ap*100:.1f}%", x, y, 0.48, C_NAVY, 2); y += 22
        line(y); y += 16
        cv2.putText(c, "MASK STATS", (x, y), FN, 0.52, C_NAVY, 2, cv2.LINE_AA); y += 20
        for l in mask_lines:
            txt(l, x, y, 0.44, C_BLACK, 1); y += 16
        line(y); y += 16
        cv2.putText(c, "ПРЕДИКТ:", (x, y), FN, 0.55, C_NAVY, 2, cv2.LINE_AA); y += 22
    elif kind == "unknown":
        line(y); y += 16
        cv2.putText(c, "ПРЕДИКТ:", (x, y), FN, 0.55, C_NAVY, 2, cv2.LINE_AA); y += 22
    return c, y


def _verdict_spine(c, y, g, rp, ar, lay):
    x = PD
    def txt(s, xx, yy, sc=0.5, col=C_BLACK, th=2): cv2.putText(c, s, (int(xx), int(yy)), FN, sc, col, th, cv2.LINE_AA)
    txt("Верная ли укладка: ", x, y, 0.5); (tw, _), _ = cv2.getTextSize("Верная ли укладка: ", FN, 0.5, 2)
    txt("Да" if lay else "Нет", x + tw, y, 0.5, C_GREEN if lay else C_RED); y += 22
    if g["ok"]:
        ang = abs(g["ang"]); ok_a = ang <= ANG
        txt("Угол отклонения: ", x, y, 0.5); (tw, _), _ = cv2.getTextSize("Угол отклонения: ", FN, 0.5, 2)
        txt(f"{ang:.2f}° ({'корректен' if ok_a else 'некорректен'})", x + tw, y, 0.5, C_GREEN if ok_a else C_RED); y += 22
    else:
        txt("Угол отклонения: n/a", x, y, 0.5); y += 22
    skip_sc = not lay
    txt("Сколиоз: ", x, y, 0.5); (tw, _), _ = cv2.getTextSize("Сколиоз: ", FN, 0.5, 2)
    if skip_sc: txt("n/a", x + tw, y, 0.5, C_GRAY)
    elif g["ok"] and g["sc"]: txt("да", x + tw, y, 0.5, C_RED)
    else: txt("нет", x + tw, y, 0.5, C_GREEN)
    y += 22
    txt("Артефакты: ", x, y, 0.5); (tw, _), _ = cv2.getTextSize("Артефакты: ", FN, 0.5, 2)
    if ar > 0: txt(f"есть ({ar} obj)", x + tw, y, 0.5, C_RED)
    else: txt("нет", x + tw, y, 0.5, C_GREEN)
    y += 26
    cv2.putText(c, "Подробно:", (x, y), FN, 0.5, C_NAVY, 2, cv2.LINE_AA); y += 20
    if lay: t1 = "Укладка верная, так как присутствуют подвздошные кости и Th12."
    else:
        rs = []
        if rp < RCL: rs.append("отсутствует Th12")
        rs.append("недостаточно подвздошных костей")
        t1 = "Укладка неверная, потому что " + "; ".join(rs) + "."
    txt(t1, x, y, 0.42, C_BLACK, 1); y += 18
    if g["ok"]:
        t2 = (f"Угол наклона {abs(g['ang']):.2f}°, корректен (<= {ANG:.1f}°)."
              if abs(g["ang"]) <= ANG else f"Угол наклона {abs(g['ang']):.2f}°, некорректен (> {ANG:.1f}°).")
    else: t2 = "Угол наклона не определён."
    txt(t2, x, y, 0.42, C_BLACK, 1); y += 18
    if skip_sc: t3 = "Сколиоз не оценивался: укладка некорректна."
    elif g["ok"] and g["sc"]: t3 = f"Сколиоз обнаружен: Cobb {g['cb']:.2f}°, total {g['tt']:.2f}°."
    else: t3 = "Сколиоз не обнаружен."
    txt(t3, x, y, 0.42, C_BLACK, 1); y += 18
    t4 = f"Артефакты найдены ({ar} obj)." if ar > 0 else "Артефакты отсутствуют."
    txt(t4, x, y, 0.42, C_BLACK, 1)
    return c


def _verdict_hip(c, y, ia, g, ar):
    x = PD
    def txt(s, xx, yy, sc=0.5, col=C_BLACK, th=2): cv2.putText(c, s, (int(xx), int(yy)), FN, sc, col, th, cv2.LINE_AA)
    pos_ok = g["pos"]
    txt("1. Позиционирование корректно: ", x, y, 0.5)
    (tw, _), _ = cv2.getTextSize("1. Позиционирование корректно: ", FN, 0.5, 2)
    txt("Да" if pos_ok else "Нет", x + tw, y, 0.5, C_GREEN if pos_ok else C_RED); y += 20
    txt(f"     troch: {'есть' if g['ht'] else 'нет'}, neck: {'есть' if g['hn'] else 'нет'}, ischium: {'есть' if g['hi'] else 'нет'}", x, y, 0.42, C_GRAY, 1); y += 22
    area_ok = ia.get("area_ok", False) if ia.get("ok") else False
    txt("2. Область интереса корректна: ", x, y, 0.5)
    (tw, _), _ = cv2.getTextSize("2. Область интереса корректна: ", FN, 0.5, 2)
    txt("Да" if area_ok else "Нет", x + tw, y, 0.5, C_GREEN if area_ok else C_RED); y += 20
    if ia.get("ok"):
        c1 = C_GREEN if ia["ok_top"] else C_RED
        c2 = C_GREEN if ia["ok_bottom"] else C_RED
        c3 = C_GREEN if ia["ok_side"] else C_RED
        side = "right" if ia["right"] else "left"
        txt(f"     top = {ia['mt']:.2f} см (>= {ITH})", x, y, 0.42, c1, 1); y += 16
        txt(f"     bot = {ia['mb']:.2f} см (>= {IBH})", x, y, 0.42, c2, 1); y += 16
        txt(f"     {side} = {ia['ms']:.2f} см (>= {ISH})", x, y, 0.42, c3, 1); y += 20
    else:
        txt("     Область интереса не определена", x, y, 0.42, C_RED, 1); y += 20
    txt("3. Ротация: ", x, y, 0.5)
    (tw, _), _ = cv2.getTextSize("3. Ротация: ", FN, 0.5, 2)
    if g["hn"]:
        cls = g["rot"]
        if cls == "over": s, col = "переротировано", C_RED
        elif cls == "normal": s, col = "норма", C_GREEN
        elif cls == "under": s, col = "недоротировано", C_RED
        else: s, col = "n/a", C_GRAY
        txt(s, x + tw, y, 0.5, col); y += 20
        hum_mm = g["hum"] * PXX
        txt(f"     выступ малого вертела = {hum_mm:.1f} мм", x, y, 0.42, C_GRAY, 1); y += 22
    else:
        txt("n/a", x + tw, y, 0.5, C_GRAY); y += 22
    txt("4. Артефакты: ", x, y, 0.5)
    (tw, _), _ = cv2.getTextSize("4. Артефакты: ", FN, 0.5, 2)
    if ar > 0: txt(f"есть ({ar} obj)", x + tw, y, 0.5, C_RED)
    else: txt("нет", x + tw, y, 0.5, C_GREEN)
    y += 24
    txt("Дополнительно (справочно):", x, y, 0.5, C_NAVY, 2); y += 20
    txt(f"     ШДУ = {g['ang']:.1f}°" if g["ang_ok"] else "     ШДУ = n/a", x, y, 0.42, C_GRAY, 1)
    return c


def _verdict_unknown(c, y, reg):
    x = PD
    cv2.putText(c, "Область: ", (x, y), FN, 0.5, C_BLACK, 2, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("Область: ", FN, 0.5, 2)
    cv2.putText(c, reg, (x + tw, y), FN, 0.5, C_NAVY, 2, cv2.LINE_AA)
    return c


def _find(r):
    return [p for p in sorted(Path(r).rglob("*")) if p.is_file() and (p.suffix.lower() in (".dcm", ".dicom") or p.suffix == "")]


def main():
    t0 = time.perf_counter()
    cls = CLS(A_M)
    print(f"[WARM] cls load:   {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    spn = _ld(S_M, SN())
    print(f"[WARM] spn load:   {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    hip = _ld(H_M, HN())
    print(f"[WARM] hip load:   {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    _ = cls(np.zeros((I, I, 3), np.uint8))
    print(f"[WARM] cls run:    {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    _ = _inf(spn, np.zeros((I, I, 3), np.uint8), True)
    print(f"[WARM] spn run:    {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    _ = _inf(hip, np.zeros((I, I, 3), np.uint8), False)
    print(f"[WARM] hip run:    {time.perf_counter()-t0:.2f}s")

    fl = _find(SRC)
    if not fl: return
    WN = "visual"
    cv2.namedWindow(WN, cv2.WINDOW_AUTOSIZE)
    idx = 0
    while True:
        p = fl[idx]
        t0 = time.perf_counter()
        full, small = _dcm(p)
        t_open = time.perf_counter() - t0
        t0 = time.perf_counter(); cid, cconf = cls(small); t_cls = time.perf_counter() - t0

        h_f, w_f = full.shape[:2]

        kind = "unknown"
        data = None
        mask_region = None
        mask_clean = None
        verdict_payload = None
        if cid == -1:
            reg = "Не определена"
            t_cnn = t_post = 0.0
        elif cid == 0:
            t0 = time.perf_counter()
            pred_300, rp, ap = _inf(spn, small, True)
            pred = _up(pred_300, w_f, h_f)
            t_cnn = time.perf_counter() - t0
            t0 = time.perf_counter()
            mc = pred.copy()
            if ap < ACL: mc[mc == 3] = 0
            tot = mc.size
            mc = _cc(mc, 1, int(SPA_PCT * tot / 100))
            mc = _cc(mc, 2, int(ILA_PCT * tot / 100))
            mc = _cc(mc, 3, int(ARA_PCT * tot / 100))
            iliac_comps = _iliac_comps(mc, mc.size)
            il = len(iliac_comps)
            ar = _no(mc, 3)
            lay = (rp >= RCL) and (il >= 2)
            g = _sp_geom(mc, skip=not lay)
            t_post = time.perf_counter() - t0
            reg = "Поясничный отдел позвоночника"; mask_region = "spine"; mask_clean = mc
            mask_lines = []
            for cid_, nm in [(1, "spine"), (2, "iliac"), (3, "artifact")]:
                mm2 = _area_mm2(mc, cid_, PXX, PXY)
                pct = 100.0 * (mc == cid_).sum() / mc.size
                mask_lines.append(f"{nm}: {_no(mc, cid_)} obj | {pct:.2f}% | {mm2:.1f} mm²")
            kind = "spine"
            data = (rp, ap, iliac_comps, mask_lines)
            verdict_payload = (g, rp, ar, lay)
        else:
            t0 = time.perf_counter()
            pred_300, rp, ap = _inf(hip, small, False)
            pred = _up(pred_300, w_f, h_f)
            t_cnn = time.perf_counter() - t0
            t0 = time.perf_counter()
            mc = pred.copy()
            if ap < ACL: mc[mc == 4] = 0
            tot = mc.size
            mc = _cc(mc, 1, int(TRO_PCT * tot / 100))
            mc = _cc(mc, 2, int(NEA_PCT * tot / 100))
            mc = _cc(mc, 3, int(ISA_PCT * tot / 100))
            mc = _cc(mc, 4, int(HAA_PCT * tot / 100))
            g = _hp_geom(mc); ia = _hp_ia(mc); ar = _no(mc, 4)
            t_post = time.perf_counter() - t0
            reg = "Проксимальный отдел бедра"; mask_region = "hip"; mask_clean = mc
            mask_lines = []
            for cid_, nm in [(1, "trochanter"), (2, "neck"), (3, "ischium"), (4, "artifact")]:
                mm2 = _area_mm2(mc, cid_, PXX, PXY)
                pct = 100.0 * (mc == cid_).sum() / mc.size
                mask_lines.append(f"{nm}: {_no(mc, cid_)} obj | {pct:.2f}% | {mm2:.1f} mm²")
            kind = "hip"
            data = (ap, mask_lines)
            verdict_payload = (ia, g, ar)

        nh = FIX_H
        sx = IW / w_f; sy = nh / h_f

        if mask_clean is not None and mask_region == "spine":
            rgb = _mask_overlay(full, mask_clean, {1: C_SAND, 2: C_BLUE, 3: C_DARK})
        elif mask_clean is not None and mask_region == "hip":
            rgb = _mask_overlay(full, mask_clean, {1: C_SAND, 2: C_BLUE, 3: C_PINK, 4: C_DARK})
        else:
            rgb = full.copy()

        od = cv2.resize(full, (IW, nh), interpolation=cv2.INTER_LINEAR)
        ov = cv2.resize(rgb, (IW, nh), interpolation=cv2.INTER_LINEAR)

        if kind == "spine":
            ov = _draw_spine_axis(ov, verdict_payload[0], sx, sy)
        elif kind == "hip":
            ia_p, g_p, _ = verdict_payload
            ov = _draw_hip_interest(ov, ia_p, sx, sy)
            ov = _draw_hip_geom(ov, g_p, sx, sy)
            ov = _draw_shdu_angle(ov, g_p, sx, sy)

        side = np.full((nh, SW, 3), C_WHITE, np.uint8)
        side, yy = _sidebar(side, idx, len(fl), p.name, [t_open, t_cls, t_cnn, t_post], (reg, cconf), kind, data)
        if kind == "spine":
            g, rp, ar, lay = verdict_payload
            _verdict_spine(side, yy, g, rp, ar, lay)
        elif kind == "hip":
            ia, g, ar = verdict_payload
            _verdict_hip(side, yy, ia, g, ar)
        else:
            _verdict_unknown(side, yy, reg)

        sep = np.full((nh, 2, 3), C_NAVY, np.uint8)
        comb = np.hstack([od, sep, ov, sep, side])
        cv2.setWindowTitle(WN, "visual | A - prev | D - next | Q - quit")
        cv2.imshow(WN, cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(0) & 0xFF
        if k == ord("a"): idx = (idx - 1) % len(fl)
        elif k == ord("d"): idx = (idx + 1) % len(fl)
        elif k in (ord("q"), 27): break
    cv2.destroyAllWindows()

def save_all(out_dir):
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cls = CLS(A_M)
    spn = _ld(S_M, SN())
    hip = _ld(H_M, HN())

    _ = cls(np.zeros((I, I, 3), np.uint8))
    _ = _inf(spn, np.zeros((I, I, 3), np.uint8), True)
    _ = _inf(hip, np.zeros((I, I, 3), np.uint8), False)

    files = _find(SRC)
    if not files:
        print(f"[ERROR] нет DICOM в {SRC}")
        return

    for idx, p in enumerate(files):
        full, small = _dcm(p)
        cid, cconf = cls(small)
        h_f, w_f = full.shape[:2]
        nh = FIX_H
        sx = IW / w_f
        sy = nh / h_f

        kind = "unknown"
        mask_clean = None
        mask_region = None
        verdict_payload = None
        data = None

        if cid == -1:
            reg = "Не определена"
        elif cid == 0:
            pred_300, rp, ap = _inf(spn, small, True)
            pred = _up(pred_300, w_f, h_f)
            mc = pred.copy()
            if ap < ACL: mc[mc == 3] = 0
            tot = mc.size
            mc = _cc(mc, 1, int(SPA_PCT * tot / 100))
            mc = _cc(mc, 2, int(ILA_PCT * tot / 100))
            mc = _cc(mc, 3, int(ARA_PCT * tot / 100))
            iliac_comps = _iliac_comps(mc, mc.size)
            ar = _no(mc, 3)
            lay = (rp >= RCL) and (len(iliac_comps) >= 2)
            g = _sp_geom(mc, skip=not lay)
            reg = "Поясничный отдел позвоночника"
            mask_clean = mc; mask_region = "spine"; kind = "spine"
            mask_lines = []
            for cid_, nm in [(1, "spine"), (2, "iliac"), (3, "artifact")]:
                mm2 = _area_mm2(mc, cid_, PXX, PXY)
                pct = 100.0 * (mc == cid_).sum() / mc.size
                mask_lines.append(f"{nm}: {_no(mc, cid_)} obj | {pct:.2f}% | {mm2:.1f} mm2")
            data = (rp, ap, iliac_comps, mask_lines)
            verdict_payload = (g, rp, ar, lay)
        else:
            pred_300, rp, ap = _inf(hip, small, False)
            pred = _up(pred_300, w_f, h_f)
            mc = pred.copy()
            if ap < ACL: mc[mc == 4] = 0
            tot = mc.size
            mc = _cc(mc, 1, int(TRO_PCT * tot / 100))
            mc = _cc(mc, 2, int(NEA_PCT * tot / 100))
            mc = _cc(mc, 3, int(ISA_PCT * tot / 100))
            mc = _cc(mc, 4, int(HAA_PCT * tot / 100))
            g = _hp_geom(mc); ia = _hp_ia(mc); ar = _no(mc, 4)
            reg = "Проксимальный отдел бедра"
            mask_clean = mc; mask_region = "hip"; kind = "hip"
            mask_lines = []
            for cid_, nm in [(1, "trochanter"), (2, "neck"), (3, "ischium"), (4, "artifact")]:
                mm2 = _area_mm2(mc, cid_, PXX, PXY)
                pct = 100.0 * (mc == cid_).sum() / mc.size
                mask_lines.append(f"{nm}: {_no(mc, cid_)} obj | {pct:.2f}% | {mm2:.1f} mm2")
            data = (ap, mask_lines)
            verdict_payload = (ia, g, ar)

        if mask_clean is not None and mask_region == "spine":
            rgb = _mask_overlay(full, mask_clean, {1: C_SAND, 2: C_BLUE, 3: C_DARK})
        elif mask_clean is not None and mask_region == "hip":
            rgb = _mask_overlay(full, mask_clean, {1: C_SAND, 2: C_BLUE, 3: C_PINK, 4: C_DARK})
        else:
            rgb = full.copy()

        ov = cv2.resize(rgb, (IW, nh), interpolation=cv2.INTER_LINEAR)
        if kind == "spine":
            ov = _draw_spine_axis(ov, verdict_payload[0], sx, sy)
        elif kind == "hip":
            ia_p, g_p, _ = verdict_payload
            ov = _draw_hip_interest(ov, ia_p, sx, sy)
            ov = _draw_hip_geom(ov, g_p, sx, sy)
            ov = _draw_shdu_angle(ov, g_p, sx, sy)

        side = np.full((nh, SW, 3), C_WHITE, np.uint8)
        side, yy = _sidebar(side, idx, len(files), p.name,
                            [0.0, 0.0, 0.0, 0.0], (reg, cconf), kind, data)
        if kind == "spine":
            g, rp, ar, lay = verdict_payload
            _verdict_spine(side, yy, g, rp, ar, lay)
        elif kind == "hip":
            ia, g, ar = verdict_payload
            _verdict_hip(side, yy, ia, g, ar)
        else:
            _verdict_unknown(side, yy, reg)

        sep = np.full((nh, 2, 3), C_NAVY, np.uint8)
        comb = np.hstack([ov, sep, side])
        out_path = out / (p.stem + ".png")
        cv2.imwrite(str(out_path), cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))
        print(f"[{idx+1}/{len(files)}] {out_path}")


def _cli():
    import argparse
    p = argparse.ArgumentParser(prog="visual")
    p.add_argument("--save", default=None,
                   help="Папка для сохранения PNG. Без флага — интерактивный просмотр.")
    args = p.parse_args()
    if args.save:
        save_all(args.save)
    else:
        main()


if __name__ == "__main__":
    _cli()
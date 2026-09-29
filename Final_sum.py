import os, sys, time, zipfile, warnings
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_CFG = _HERE / "config"

def _read_path(name, default=""):
    p = _CFG / name
    if p.exists():
        s = p.read_text(encoding="utf-8").strip()
        if s: return Path(s)
    return Path(default) if default else Path("__no_input__")

CLS_MODEL   = _CFG / "cls.pt"
SPINE_MODEL = _CFG / "unet_spine_mt_addDice.pt"
HIP_MODEL   = _CFG / "hip_b0_v4.pt"
INPUT_DIR   = _read_path("input.txt")
OUT_XLSX    = _HERE / "report.xlsx"

def _die(msg, hint=""):
    print(f"[ERROR] {msg}")
    if hint: print(f"        {hint}")
    sys.exit(1)

if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
    _die(f"папка не найдена: {INPUT_DIR}",
         "проверьте путь в ./config/input.txt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 300
BATCH = 8
SPA_PCT, ILA_PCT, ARA_PCT = 0.5, 0.5, 0.05
TRO_PCT, NEA_PCT, ISA_PCT, HAA_PCT = 0.5, 0.3, 0.3, 0.1
ANG, SEG, WIN, EDG, SCB, SCT, MRN = 5.0, 15, 3, 3, 3.0, 10.0, 5
ACL, RCL = 0.5, 0.5
PXY, PXX = 1.05, 0.60
ITH, IBH, ISH = 3.0, 2.0, 2.0
ROV, RUN = 2.0, 6.0


class _A(nn.Module):
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


class _S(nn.Module):
    def __init__(s, nc=4, nr=2, na=2, enc="efficientnet-b0"):
        super().__init__()
        s.unet = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=3, classes=nc)
        d = s.unet.encoder.out_channels[-1]
        s.ribs_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(d, nr))
        s.art_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(d, na))
    def forward(s, x):
        f = s.unet.encoder(x)
        d = s.unet.decoder(f)
        return s.unet.segmentation_head(d), s.ribs_head(f[-1]), s.art_head(f[-1])


class _H(nn.Module):
    def __init__(s, nc=5, na=2, enc="efficientnet-b0"):
        super().__init__()
        s.unet = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=3, classes=nc)
        d = s.unet.encoder.out_channels[-1]
        s.art_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(d, na))
    def forward(s, x):
        f = s.unet.encoder(x)
        d = s.unet.decoder(f)
        return s.unet.segmentation_head(d), s.art_head(f[-1])


def _ld(p, m):
    c = torch.load(p, map_location="cpu", weights_only=False)
    st = c.get("model_state_dict", c) if isinstance(c, dict) else c
    m.load_state_dict(st)
    m = m.to(DEVICE).eval()
    return m.half() if DEVICE == "cuda" else m


class _C:
    def __init__(s, p):
        c = torch.load(p, map_location=DEVICE, weights_only=False)
        s.m = _A(c.get("num_classes", 2)).to(DEVICE).eval()
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
        s.tf = A.Compose([A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
    @torch.no_grad()
    def __call_batch__(s, imgs_rgb):
        xs = torch.stack([s.tf(image=im)["image"] for im in imgs_rgb], 0).to(DEVICE)
        feats = s.m.ff(xs).cpu().numpy()
        lo = s.m(xs).float()
        pr = F.softmax(lo, 1).cpu().numpy()
        pt = F.softmax(lo / s.T, 1).cpu().numpy()
        out = []
        for i in range(xs.size(0)):
            f = feats[i]
            pm = float(np.max(pt[i]))
            d = (f - s.mu).astype(np.float32)
            sc = float(np.sum((d @ s.ci) * d))
            n = f / (np.linalg.norm(f) + 1e-8)
            cs = float(max(n @ s.c0, n @ s.c1))
            if sc > s.th or cs < s.tc or pm < s.ts:
                out.append((-1, pm))
            else:
                out.append((int(np.argmax(pr[i])), pm))
        return out
    @torch.no_grad()
    def __call__(s, img_rgb):
        return s.__call_batch__([img_rgb])[0]


_S_TF = A.Compose([A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
_H_TF = A.Compose([A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])


def _up(mask, w, h):
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _dcm(p):
    d = pydicom.dcmread(str(p))
    a = d.pixel_array.astype(np.float32)
    a = a - a.min()
    if a.max() > 0: a = a / a.max()
    a = (a * 255).astype(np.uint8)
    if a.ndim == 2: a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    full = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    small = cv2.resize(full, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    return full, small, str(getattr(d, "StudyInstanceUID", "")), str(getattr(d, "SOPInstanceUID", ""))


def _cc(mask, cid, amin):
    b = (mask == cid).astype(np.uint8)
    if b.sum() == 0: return mask
    n, l, st, _ = cv2.connectedComponentsWithStats(b, 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < amin: mask[l == i] = 0
    return mask


def _nobj(mask, cid):
    n, _, _, _ = cv2.connectedComponentsWithStats((mask == cid).astype(np.uint8), 8)
    return max(0, n - 1)


def _mainbin(m):
    n, l, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return m
    i = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
    return (l == i).astype(np.uint8)


def _rows(b):
    ys = np.nonzero(b.any(1))[0]
    if len(ys) < 30: return None, None
    cols = np.arange(b.shape[1], dtype=np.float32)
    sub = b[ys].astype(np.float32); d = sub.sum(1); v = d > 0
    ys, sub, d = ys[v], sub[v], d[v]
    if len(ys) < 30: return None, None
    return ys.astype(np.float32), ((sub * cols).sum(1) / d).astype(np.float32)


def _seg(sx, sy, L):
    n = len(sx); k = max(3, n // L); ll = n // k
    ax = np.zeros(k, np.float32); ay = np.zeros(k, np.float32)
    for i in range(k):
        a = i * ll; b = (i + 1) * ll if i < k - 1 else n
        ay[i] = sy[a:b].mean(); ax[i] = sx[a:b].mean()
    return ax, ay, k


def _sl(sx, sy):
    if len(sx) < 2: return 0.0
    ym = sy.mean(); xm = sx.mean()
    dy = sy - ym; dd = (dy * dy).sum()
    if dd < 1e-6: return 0.0
    return float(np.degrees(np.arctan(float((dy * (sx - xm)).sum() / dd))))


def _run(a, s):
    b = c = 0
    for v in a:
        sg = 1 if v > 0 else (-1 if v < 0 else 0)
        if sg == s:
            c += 1
            if c > b: b = c
        else: c = 0
    return b


def _sp_geom(mask, skip):
    e = {"ok": False, "ang": 0.0, "sco": False, "cobb": 0.0, "tot": 0.0}
    b = (mask == 1).astype(np.uint8)
    if b.sum() == 0: return e
    ys, xs = _rows(b)
    if ys is None: return e
    n = len(ys); bd = max(3, n // 10)
    xt, yt = float(xs[:bd].mean()), float(ys[:bd].mean())
    xb, yb = float(xs[-bd:].mean()), float(ys[-bd:].mean())
    ang = float(np.degrees(np.arctan2(xb - xt, yb - yt)))
    r = {"ok": True, "ang": ang, "sco": False, "cobb": 0.0, "tot": 0.0}
    if skip: return r
    ax, ay, k = _seg(xs, ys, SEG)
    if k < EDG * 2 + 4: return r
    h = WIN // 2
    la = np.zeros(k, np.float32)
    for i in range(k):
        a = max(0, i - h); b2 = min(k, i + h + 1)
        la[i] = _sl(ax[a:b2], ay[a:b2])
    mid = k // 2
    if mid <= EDG or k - EDG <= mid: return r
    ts = la[EDG:mid]; bs = la[mid:k - EDG]
    tm = float(ts[int(np.argmax(np.abs(ts)))])
    bm = float(bs[int(np.argmax(np.abs(bs)))])
    cb = tm - bm; tt = 0.5 * (tm + bm)
    ro = (_run(la, -1) >= MRN) or (_run(la, 1) >= MRN)
    r["sco"] = bool(ro and (abs(cb) > SCB or abs(tt) > SCT))
    r["cobb"] = float(cb); r["tot"] = float(tt)
    return r


def _hp_geom(mask):
    g = {"ok": False, "pos": False, "ht": False, "hn": False, "hi": False, "rot": "unknown", "hum": 0.0, "hum_rel": 0.0}
    t = (mask == 1).astype(np.uint8); n = (mask == 2).astype(np.uint8); i = (mask == 3).astype(np.uint8)
    if t.sum() == 0: return g
    t = _mainbin(t)
    n = _mainbin(n) if n.sum() > 0 else n
    i = _mainbin(i) if i.sum() > 0 else i
    g["ht"] = True
    tot = mask.size
    g["hn"] = (100.0 * n.sum() / tot) >= NEA_PCT
    g["hi"] = (100.0 * i.sum() / tot) >= ISA_PCT
    g["pos"] = g["ht"] and g["hn"] and g["hi"]
    if not g["hn"]: return g
    yt, xt = np.where(t > 0); yn, xn = np.where(n > 0)
    right = True
    if len(xn) >= 10 and len(xt) >= 10:
        yc, xc = np.where((t > 0) & (n > 0))
        right = (float(xc.mean()) > float(xn.mean())) if len(xc) >= 5 else (float(xt.mean()) > float(xn.mean()))
    yt0, yb0 = int(yt.min()), int(yt.max()); ht = max(1, yb0 - yt0)
    yr = int(yn.max()); ylo = max(yt0, yr); yhi = min(yb0 - 3, yr + int(0.40 * ht))
    if yhi > ylo + 5:
        pr = {}
        for yy in range(ylo, yhi + 1):
            xs = np.where(t[yy, :] > 0)[0]
            if len(xs): pr[yy] = int(xs.min()) if right else int(xs.max())
        if len(pr) >= 7:
            ks = sorted(pr.keys()); vv = np.array([pr[k] for k in ks], np.float32)
            sm = np.convolve(vv, np.ones(7, np.float32) / 7.0, "same")
            bi = (int(np.argmin(sm[3:-3])) if right else int(np.argmax(sm[3:-3]))) + 3
            py = ks[bi]; px = pr[py]; bl = float(np.median(sm)); pv = abs(sm[bi] - bl)
            th = max(2.0, pv * 0.4)
            ya = py
            for j in range(bi, -1, -1):
                if abs(sm[j] - bl) < th: ya = ks[j]; break
            yb = py
            for j in range(bi, len(ks)):
                if abs(sm[j] - bl) < th: yb = ks[j]; break
            yl2, yh2 = max(yt0, ya - 2), min(yb0, yb + 2)
            st = np.zeros_like(t); st[yl2:yh2 + 1, :] = t[yl2:yh2 + 1, :]
            cs, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs:
                cnt = max(cs, key=cv2.contourArea).reshape(-1, 2)
                sp = cnt[cnt[:, 0] < cnt[:, 0].mean()] if right else cnt[cnt[:, 0] > cnt[:, 0].mean()]
                if len(sp) >= 10:
                    sp = sp[np.argsort(sp[:, 1])]
                    yt1, yb1 = int(sp[:, 1].min()), int(sp[:, 1].max())
                    ym = (yt1 + yb1) // 2; hr = max(5, (yb1 - yt1) // 4)
                    bl2, bh2 = max(yt1, ym - hr), min(yb1, ym + hr)
                    bp = sp[(sp[:, 1] >= bl2) & (sp[:, 1] <= bh2)]
                    if len(bp) >= 5:
                        ysb = bp[:, 1].astype(np.float32); xsb = bp[:, 0].astype(np.float32)
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
        if right:
            xe = int(xs_top.max())
        else:
            xe = int(xs_top.min())
    else:
        if right:
            xe = int(xt.max())
        else:
            xe = int(xt.min())

    mt = ytop * PXY / 10.0
    mb = (h - 1 - ybot) * PXY / 10.0
    ms = ((w - 1 - xe) if right else xe) * PXX / 10.0

    return {"ok": True, "area_ok": mt >= ITH and mb >= IBH and ms >= ISH}


@torch.no_grad()
def _inf_batch(m, imgs, spine):
    tf = _S_TF if spine else _H_TF
    xs = torch.stack([tf(image=im)["image"] for im in imgs], 0).to(DEVICE)
    if DEVICE == "cuda": xs = xs.half()
    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        out = m(xs)
    res = []
    if spine:
        sg, rl, al = out
        sgs = sg.float().cpu().numpy()
        rps = F.softmax(rl.float(), 1).cpu().numpy()
        aps = F.softmax(al.float(), 1).cpu().numpy()
        for i in range(len(imgs)):
            h, w = imgs[i].shape[:2]
            pr = _up(np.argmax(sgs[i], 0).astype(np.uint8), w, h)
            res.append((pr, float(rps[i][1]), float(aps[i][1])))
    else:
        sg, al = out
        sgs = sg.float().cpu().numpy()
        aps = F.softmax(al.float(), 1).cpu().numpy()
        for i in range(len(imgs)):
            h, w = imgs[i].shape[:2]
            pr = _up(np.argmax(sgs[i], 0).astype(np.uint8), w, h)
            res.append((pr, 0.0, float(aps[i][1])))
    return res


def _xml_esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def _write_xlsx(rows, path):
    headers = ["path_to_study", "study_uid", "image_uid", "anatomical_region",
               "quality_class", "violation_type", "processing_status", "time_of_processing"]
    ncol = len(headers); nrow = len(rows) + 1
    shared = []; idx_map = {}
    def sid(t):
        t = str(t)
        if t not in idx_map:
            idx_map[t] = len(shared); shared.append(t)
        return idx_map[t]
    for h in headers: sid(h)
    for r in rows:
        for c in r: sid(c)
    def colref(i):
        s = ""
        while i >= 0:
            s = chr(i % 26 + 65) + s; i = i // 26 - 1
        return s
    rows_xml = []
    for ri in range(nrow):
        cells = []
        for ci in range(ncol):
            if ri == 0:
                cells.append(f'<c r="{colref(ci)}{ri+1}" t="s"><v>{sid(headers[ci])}</v></c>')
            else:
                v = rows[ri - 1][ci]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cells.append(f'<c r="{colref(ci)}{ri+1}"><v>{v}</v></c>')
                else:
                    cells.append(f'<c r="{colref(ci)}{ri+1}" t="s"><v>{sid(v)}</v></c>')
        rows_xml.append(f'<row r="{ri+1}">' + "".join(cells) + '</row>')
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             '<sheetData>' + "".join(rows_xml) + '</sheetData></worksheet>')
    sst_items = "".join(f'<si><t xml:space="preserve">{_xml_esc(t)}</t></si>' for t in shared)
    sst = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(shared)}" uniqueCount="{len(shared)}">{sst_items}</sst>')
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<sheets><sheet name="report" sheetId="1" r:id="rId1"/></sheets></workbook>')
    wb_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
               '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
               '</Relationships>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                 '</Relationships>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
          '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
          '</Types>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)


def _region_spine(mask, ribs, art_p):
    mc = mask.copy()
    if art_p < ACL: mc[mc == 3] = 0
    tot = mc.size
    mc = _cc(mc, 1, int(SPA_PCT * tot / 100))
    mc = _cc(mc, 2, int(ILA_PCT * tot / 100))
    mc = _cc(mc, 3, int(ARA_PCT * tot / 100))
    iliac = _nobj(mc, 2); art = _nobj(mc, 3)
    layout = (ribs >= RCL) and (iliac >= 2)
    geo = _sp_geom(mc, skip=not layout)
    viol = []
    if not layout:
        if ribs < RCL: viol.append("Некорректная укладка")
        if iliac < 2: viol.append("Некорректная укладка")
    if geo["ok"] and abs(geo["ang"]) > ANG: viol.append("Не выравнена ось позвоночника")
    if layout and geo["ok"] and geo["sco"]: viol.append("Не выравнена ось позвоночника")
    if art > 0: viol.append("Присутствуют посторонние предметы")
    viol = list(dict.fromkeys(viol))
    return "Поясничный отдел позвоночника", (0 if not viol else 1), ";".join(viol)


def _region_hip(mask, art_p):
    mc = mask.copy()
    if art_p < ACL: mc[mc == 4] = 0
    tot = mc.size
    mc = _cc(mc, 1, int(TRO_PCT * tot / 100))
    mc = _cc(mc, 2, int(NEA_PCT * tot / 100))
    mc = _cc(mc, 3, int(ISA_PCT * tot / 100))
    mc = _cc(mc, 4, int(HAA_PCT * tot / 100))
    geo = _hp_geom(mc); ia = _hp_ia(mc); art = _nobj(mc, 4)
    viol = []
    if not geo["pos"]: viol.append("Некорректная укладка")
    if not ia.get("ok", False) or not ia.get("area_ok", False): viol.append("Некорректная область интереса")
    if geo["hn"] and geo["rot"] == "over": viol.append("Переротировано")
    if geo["hn"] and geo["rot"] == "under": viol.append("Недоротировано")
    if art > 0: viol.append("Присутствуют посторонние предметы")
    viol = list(dict.fromkeys(viol))
    return "Проксимальный отдел бедра", (0 if not viol else 1), ";".join(viol)


def _find_dicoms(root):
    exts = {".dcm", ".dicom"}
    return [p for p in sorted(Path(root).rglob("*"))
            if p.is_file() and (p.suffix.lower() in exts or p.suffix == "")]


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    t_start = time.perf_counter()
    files = _find_dicoms(INPUT_DIR)
    if not files:
        _die(f"в папке нет DICOM-файлов: {INPUT_DIR}",
             "проверьте путь в ./config/input.txt")

    t0 = time.perf_counter()
    cls = _C(CLS_MODEL)
    print(f"[WARM] cls load:   {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    spn = _ld(SPINE_MODEL, _S())
    print(f"[WARM] spn load:   {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    hip = _ld(HIP_MODEL, _H())
    print(f"[WARM] hip load:   {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    _ = cls(np.zeros((IMG, IMG, 3), np.uint8))
    print(f"[WARM] cls run:    {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    _ = _inf_batch(spn, [np.zeros((IMG, IMG, 3), np.uint8)], True)
    print(f"[WARM] spn run:    {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    _ = _inf_batch(hip, [np.zeros((IMG, IMG, 3), np.uint8)], False)
    print(f"[WARM] hip run:    {time.perf_counter()-t0:.2f}s")

    loaded = []
    rows = []
    for f in files:
        t0 = time.perf_counter()
        try:
            full, small, st_uid, im_uid = _dcm(f)
        except Exception:
            rows.append([str(f), "", "", "", 1, "Некорректная укладка", "Failure", 0.0])
            continue
        t_read = time.perf_counter() - t0
        loaded.append({"f": f, "full": full, "small": small, "st": st_uid, "im": im_uid,
                       "t_read": t_read, "t_cls": 0.0, "t_unet": 0.0, "t_post": 0.0})

    idxs = list(range(len(loaded)))
    cls_res = [None] * len(loaded)

    for ch in _chunks(idxs, BATCH):
        imgs = [loaded[i]["small"] for i in ch]
        t0 = time.perf_counter()
        out = cls.__call_batch__(imgs)
        t_batch = time.perf_counter() - t0
        t_share = t_batch / max(1, len(ch))
        for k, i in enumerate(ch):
            cls_res[i] = out[k]
            loaded[i]["t_cls"] = t_share

    spines = [i for i in idxs if cls_res[i][0] == 0]
    hips = [i for i in idxs if cls_res[i][0] == 1]

    post = {}

    for ch in _chunks(spines, BATCH):
        imgs = [loaded[i]["small"] for i in ch]
        t0 = time.perf_counter()
        outs = _inf_batch(spn, imgs, True)
        t_batch = time.perf_counter() - t0
        t_share = t_batch / max(1, len(ch))
        for k, i in enumerate(ch):
            loaded[i]["t_unet"] = t_share
            post[i] = ("spine", outs[k])

    for ch in _chunks(hips, BATCH):
        imgs = [loaded[i]["small"] for i in ch]
        t0 = time.perf_counter()
        outs = _inf_batch(hip, imgs, False)
        t_batch = time.perf_counter() - t0
        t_share = t_batch / max(1, len(ch))
        for k, i in enumerate(ch):
            loaded[i]["t_unet"] = t_share
            post[i] = ("hip", outs[k])

    for i in idxs:
        it = loaded[i]
        t0 = time.perf_counter()
        cid, _ = cls_res[i]
        if cid == -1:
            reg, q, viol = "Не определена", 0, ""
        elif cid == 0:
            pred_300, ribs_p, art_p = post[i][1]
            pred = _up(pred_300, it["full"].shape[1], it["full"].shape[0])
            reg, q, viol = _region_spine(pred, ribs_p, art_p)
        elif cid == 1:
            pred_300, _, art_p = post[i][1]
            pred = _up(pred_300, it["full"].shape[1], it["full"].shape[0])
            reg, q, viol = _region_hip(pred, art_p)
        else:
            reg, q, viol = "Не определена", 0, ""
        it["t_post"] = time.perf_counter() - t0
        dt = it["t_read"] + it["t_cls"] + it["t_unet"] + it["t_post"]
        rows.append([str(it["f"]), it["st"], it["im"], reg, q, viol, "Success", round(dt, 4)])

    _write_xlsx(rows, OUT_XLSX)
    total = time.perf_counter() - t_start
    print(f"files={len(files)} total={total:.4f}s avg={total/max(1,len(files)):.4f}s")
    print(f"out={OUT_XLSX}")


if __name__ == "__main__":
    main()
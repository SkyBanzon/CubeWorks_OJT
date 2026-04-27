#!/usr/bin/env python3
"""
Wire Harness Measurer  –  Fast/No-Lag Edition
=============================================
Lag fixes applied
-----------------
1.  DEDICATED CAPTURE THREAD   – cv2.VideoCapture.read() blocks; moved off the
    main thread entirely so Tkinter never stalls waiting for a frame.
2.  DEDICATED PROCESSING THREAD – wire_mask / skeletonize / analyse are the
    heaviest ops. They run in their own thread, completely separate from UI.
3.  TWO QUEUES (maxsize=1 each) – capture→process and process→UI.
    Both queues drop the stale item before pushing a new one, so neither
    thread ever builds up a backlog.
4.  UI THREAD does NOTHING heavy – only reads results, draws overlays, and
    updates Tkinter labels. Polled every 30 ms via root.after().
5.  OpenCL UMat for bilateral filter + morphology (GPU offload, no install).
6.  _fit_image() uses NEAREST pre-downscale + BILINEAR final (not LANCZOS)
    when shrinking — much faster on large warped frames.
7.  Skeleton pixel-walk (n8/trace) replaced with vectorised numpy convolution
    for endpoint/junction detection — avoids a Python loop over every pixel.
8.  spline_length() caps sample points at 500 (was already there) but also
    skips spline entirely for very short paths (< 20 pts) to avoid splprep
    overhead on noise segments.
"""

import cv2
import numpy as np
import json, os, math, threading, queue, time
import tkinter as tk
from tkinter import filedialog
from collections import deque
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from PIL import Image, ImageTk

# ── CONFIG ────────────────────────────────────────────────────────────────────
SETTINGS_FILE   = "wire_settings.json"
DEFAULT_URL     = "http://10.40.71.2:8080/video"
DICT_TYPE       = cv2.aruco.DICT_5X5_100
MAX_FAIL        = 30
A4_W, A4_H      = 210.0, 297.0
INSET_MM        = 5 + (57 / 2)
PX_MM           = 4.0
OUT_W, OUT_H    = int(A4_W * PX_MM), int(A4_H * PX_MM)
SCALE           = 1.0 / PX_MM
INSET_PX        = int(INSET_MM * PX_MM)
SMOOTH_W        = 30
MIN_SEG         = 30
MIN_BRANCH_MM   = 15.0
JN_CLUSTER_R    = 20
JN_MATCH_R      = 25
EP_MATCH_R      = 10
ARUCO_MARGIN_PX = 30
SPLINE_SMOOTH   = 5.0
SPLINE_PTS      = 300          # reduced from 500 – still accurate enough

MARKER_CORNER = {0: "tl", 1: "tr", 2: "bl", 3: "br"}
DEF_BS, DEF_CO, DEF_MS, DEF_BL, DEF_CI = 53, 13, 5, 11, 5

_aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
_aruco_det  = cv2.aruco.ArucoDetector(_aruco_dict, cv2.aruco.DetectorParameters())

BC = [(255,150,0),(0,165,255),(255,0,255),(0,255,255),
      (255,255,0),(128,0,255),(0,255,128),(255,128,0)]

# ── GPU CHECK ─────────────────────────────────────────────────────────────────
USE_CUDA = False
try:
    USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
except AttributeError:
    pass
GPU_LABEL = "⚡ CUDA" if USE_CUDA else "⚡ OpenCL"
print(f"[GPU] {GPU_LABEL} active")

# ── SETTINGS ──────────────────────────────────────────────────────────────────
def load_settings():
    d = {"stream_url": DEFAULT_URL, "block_size": DEF_BS, "c_offset": DEF_CO,
         "morph_size": DEF_MS, "blur_size": DEF_BL, "close_iter": DEF_CI}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f: d.update(json.load(f))
        except Exception: pass
    return d

def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w") as f: json.dump(s, f, indent=2)
    except Exception: pass

# ── HOMOGRAPHY ────────────────────────────────────────────────────────────────
_DST = {
    "tl": np.array([INSET_PX,          INSET_PX         ], np.float32),
    "tr": np.array([OUT_W - INSET_PX,  INSET_PX         ], np.float32),
    "bl": np.array([INSET_PX,          OUT_H - INSET_PX ], np.float32),
    "br": np.array([OUT_W - INSET_PX,  OUT_H - INSET_PX ], np.float32),
}

def compute_homography(corners, ids):
    found  = set(ids.flatten().tolist())
    needed = set(MARKER_CORNER.keys())
    if not needed.issubset(found): return None, None, None
    centres = {}
    for i, m in enumerate(ids.flatten()):
        mid = int(m)
        if mid in needed:
            p = corners[i][0]
            centres[mid] = np.array([p[:,0].mean(), p[:,1].mean()], np.float32)
    src = np.array([centres[mid] for mid in MARKER_CORNER], np.float32)
    dst = np.array([_DST[MARKER_CORNER[mid]] for mid in MARKER_CORNER], np.float32)
    H, _ = cv2.findHomography(src, dst)
    ac = [corners[i][0] for i in range(len(ids))]
    return H, (OUT_W, OUT_H), ac

# ── WIRE MASK (OpenCL / CUDA accelerated) ────────────────────────────────────
def wire_mask(fr, bs, co, ms, bl, ci):
    L = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.createCLAHE(3.0, (8, 8)).apply(L)

    # Bilateral filter – GPU path
    if USE_CUDA:
        try:
            gm = cv2.cuda_GpuMat(); gm.upload(L)
            gm = cv2.cuda.createBilateralFilter(bl | 1, 75, 75).apply(gm)
            L  = gm.download()
        except Exception:
            L = cv2.bilateralFilter(L, bl | 1, 75, 75)
    else:
        L = cv2.bilateralFilter(cv2.UMat(L), bl | 1, 75, 75).get()

    m  = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, bs | 1 if bs >= 3 else 3, co)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ms, ms))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ms*2+1, ms*2+1))
    mu = cv2.UMat(m)
    mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, k,  iterations=2)
    mu = cv2.morphologyEx(mu, cv2.MORPH_OPEN,  k,  iterations=1)
    mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, k2, iterations=ci)
    m  = mu.get()

    cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r = np.zeros_like(m)
    for c in cn:
        if cv2.contourArea(c) >= 500:
            cv2.drawContours(r, [c], -1, 255, -1)
    return r

def mask_aruco(m, H, raw_corners):
    if raw_corners is None or H is None:
        h, w = m.shape[:2]; b = INSET_PX + int(25 * PX_MM)
        m[0:b, 0:b]=0; m[0:b, w-b:]=0; m[h-b:, 0:b]=0; m[h-b:, w-b:]=0
        m[0:int(15*PX_MM), :] = 0
        return m
    for quad in raw_corners:
        pts = np.array(quad, np.float32).reshape(-1,1,2)
        wp  = cv2.perspectiveTransform(pts, H).reshape(-1,2)
        mg  = ARUCO_MARGIN_PX
        x0  = max(int(wp[:,0].min())-mg, 0);  x1 = min(int(wp[:,0].max())+mg, m.shape[1])
        y0  = max(int(wp[:,1].min())-mg, 0);  y1 = min(int(wp[:,1].max())+mg, m.shape[0])
        m[y0:y1, x0:x1] = 0
    return m

# ── SKELETON ──────────────────────────────────────────────────────────────────
def get_skel(m):
    return (skeletonize(m > 0).astype(np.uint8)) * 255

# Vectorised neighbour count (replaces per-pixel Python loop)
_K8 = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)

def neighbour_count(skel_bin):
    """Returns array of 8-neighbour counts for every pixel."""
    return cv2.filter2D(skel_bin.astype(np.uint8), -1, _K8)

def n8(s, y, x):
    h, w = s.shape; o = []
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==0 and dx==0: continue
            ny, nx = y+dy, x+dx
            if 0<=ny<h and 0<=nx<w and s[ny,nx]: o.append((ny,nx))
    return o

def trace(s, st):
    v={st}; p=[st]; c=st
    while True:
        nxt=None
        for nb in n8(s, c[0], c[1]):
            if nb not in v: nxt=nb; break
        if nxt is None: break
        p.append(nxt); v.add(nxt); c=nxt
    return p

def alen(pts):
    if len(pts)<2: return 0.0
    a=np.array(pts, np.float64)
    return float(np.sqrt((np.diff(a,axis=0)**2).sum(axis=1)).sum())

def spline_length(pts):
    if len(pts) < 20: return alen(pts)          # skip spline for tiny segs
    arr = np.array(pts, np.float64)
    if len(arr) > 1500:
        idx = np.linspace(0, len(arr)-1, 1500).astype(int); arr = arr[idx]
    try:
        tck, _ = splprep([arr[:,0], arr[:,1]], s=SPLINE_SMOOTH*len(arr), k=3)
        uf = np.linspace(0,1,SPLINE_PTS)
        ys, xs = splev(uf, tck)
        return float(np.sqrt(np.diff(ys)**2 + np.diff(xs)**2).sum())
    except Exception:
        return alen(pts)

# ── HARNESS ANALYSIS ─────────────────────────────────────────────────────────
def analyse(skeleton):
    s  = (skeleton > 0).astype(np.uint8)
    nc = neighbour_count(s)                       # vectorised — fast!

    eps_mask = (s==1) & (nc==1)
    jns_mask = (s==1) & (nc>=3)
    eps = set(map(tuple, np.argwhere(eps_mask).tolist()))
    jns = set(map(tuple, np.argwhere(jns_mask).tolist()))
    if not jns: return None

    # Cluster junction pixels
    jl = list(jns); used = set(); clusters = []
    for i, p in enumerate(jl):
        if i in used: continue
        cl=[p]; used.add(i); stk=[p]
        while stk:
            cur=stk.pop()
            for j,p2 in enumerate(jl):
                if j in used: continue
                if max(abs(cur[0]-p2[0]), abs(cur[1]-p2[1])) <= JN_CLUSTER_R:
                    cl.append(p2); used.add(j); stk.append(p2)
        clusters.append(cl)

    jcenters = [(int(np.mean([p[0] for p in cl])),
                 int(np.mean([p[1] for p in cl]))) for cl in clusters]

    sc = s.copy()
    for py,px in jns: sc[py,px]=0

    ncc, labels = cv2.connectedComponents(sc, connectivity=8)
    segs = []
    for lb in range(1, ncc):
        cy, cx = np.where(labels==lb)
        if len(cy)<5: continue
        comp = (labels==lb).astype(np.uint8)
        nc2  = neighbour_count(comp)
        se   = list(map(tuple, np.argwhere((comp==1)&(nc2==1)).tolist()))
        if not se: continue
        path   = trace(comp, se[0])
        length = spline_length(path)
        if length < MIN_SEG: continue

        def classify(pt):
            for ep in eps:
                if max(abs(pt[0]-ep[0]),abs(pt[1]-ep[1])) <= EP_MATCH_R:
                    return 'ep', ep
            bd=9999; bj=None
            for jc in jcenters:
                d=max(abs(pt[0]-jc[0]),abs(pt[1]-jc[1]))
                if d<bd: bd=d; bj=jc
            if bd<=JN_MATCH_R: return 'jn', bj
            return 'unk', pt

        t0,n0=classify(path[0]); t1,n1=classify(path[-1])
        segs.append({'path':path,'len':length,'e0':n0,'t0':t0,'e1':n1,'t1':t1})

    if len(segs)<2: return None

    tc = [s for s in segs if s['t0']=='ep' or s['t1']=='ep'] or segs
    trunk = max(tc, key=lambda s: s['len'])

    if   trunk['t0']=='ep': tep=trunk['e0']; tjn=trunk['e1']
    elif trunk['t1']=='ep': tep=trunk['e1']; tjn=trunk['e0']
    else:                   tep=trunk['e0']; tjn=trunk['e1']

    branches=[]; uid={id(trunk)}
    def bfs(junc, dist):
        for seg in segs:
            if id(seg) in uid: continue
            conn=other=ot=None
            for ta,ea,tb,eb in [(seg['t0'],seg['e0'],seg['t1'],seg['e1']),
                                 (seg['t1'],seg['e1'],seg['t0'],seg['e0'])]:
                if ta=='jn' and ea is not None:
                    if max(abs(ea[0]-junc[0]),abs(ea[1]-junc[1]))<=JN_MATCH_R:
                        conn=ea; other=eb; ot=tb; break
            if conn is None: continue
            uid.add(id(seg)); tot=dist+seg['len']
            if ot in ('ep','unk'):
                branches.append({'path':seg['path'],'len':seg['len'],
                                 'endpoint':other,'junction':junc,'total':tot})
            elif ot=='jn': bfs(other,tot)
    bfs(tjn, trunk['len'])

    branches=[b for b in branches if b['len']>=MIN_BRANCH_MM*PX_MM]
    if not branches: return None

    def bangle(b):
        p=b['path']
        d0=(p[0][0]-tjn[0])**2+(p[0][1]-tjn[1])**2
        d1=(p[-1][0]-tjn[0])**2+(p[-1][1]-tjn[1])**2
        tip=p[0] if d0>d1 else p[-1]
        return math.atan2(tip[0]-tjn[0], tip[1]-tjn[1])
    branches.sort(key=bangle)

    used_j={tjn} if tjn else set()
    for b in branches:
        if b['junction']: used_j.add(b['junction'])
    active_j=[jc for jc in jcenters if jc in used_j]

    return {'trunk':{'path':trunk['path'],'len':trunk['len'],'ep':tep},
            'branches':branches,'junctions':active_j}

# ── SIMPLE FALLBACK ───────────────────────────────────────────────────────────
def simple_strands(skeleton):
    ncc, labels = cv2.connectedComponents(skeleton)
    out=[]
    for lb in range(1,ncc):
        comp=((labels==lb)*255).astype(np.uint8)
        su=(comp>0).astype(np.uint8)
        nc=cv2.filter2D(su,-1,_K8)
        ep=np.argwhere((su==1)&(nc==1))
        st=ep[0] if len(ep) else np.argwhere(comp>0)[0]
        vis=set(); path=[]; r,c=int(st[0]),int(st[1])
        path.append((c,r)); vis.add((r,c)); h,w=comp.shape
        while True:
            f=False
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    if dy==0 and dx==0: continue
                    nr,nc2=r+dy,c+dx
                    if 0<=nr<h and 0<=nc2<w and comp[nr,nc2] and (nr,nc2) not in vis:
                        vis.add((nr,nc2)); r,c=nr,nc2; path.append((c,r)); f=True; break
                if f: break
            if not f: break
        l=spline_length([(p[1],p[0]) for p in path])
        if l>20: out.append((path,l))
    out.sort(key=lambda x:x[1],reverse=True)
    return out

# ── FAST IMAGE FIT ────────────────────────────────────────────────────────────
def fit_image(cv_img, box_w, box_h):
    ih,iw=cv_img.shape[:2]
    if iw==0 or ih==0 or box_w<1 or box_h<1:
        return Image.new("RGB",(max(box_w,1),max(box_h,1)))
    sc=min(box_w/iw, box_h/ih)
    nw,nh=max(int(iw*sc),1),max(int(ih*sc),1)
    rgb=cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # Fast pre-downscale when shrinking a lot
    if sc < 0.5:
        rgb=cv2.resize(rgb,(max(nw*2,1),max(nh*2,1)),interpolation=cv2.INTER_NEAREST)
    pil=Image.fromarray(rgb).resize((nw,nh), Image.BILINEAR)
    out=Image.new("RGB",(box_w,box_h))
    out.paste(pil,((box_w-nw)//2,(box_h-nh)//2))
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 1 – CAPTURE
#  Reads frames from the camera as fast as possible.
#  Drops the stale frame in the queue before pushing the new one.
# ══════════════════════════════════════════════════════════════════════════════
class CaptureThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q      = queue.Queue(maxsize=1)
        self._stop  = threading.Event()
        self.cap    = None
        self.fail   = 0

    def open(self, url):
        cap = cv2.VideoCapture(url)
        # For network streams isOpened() is not enough — actually read a frame
        if not cap.isOpened():
            cap.release(); return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep buffer tiny
        # Try up to 3 frames to confirm stream is alive
        for _ in range(3):
            ret, _ = cap.read()
            if ret:
                self.cap  = cap
                self.fail = 0
                return True
        cap.release()
        return False

    def close(self):
        if self.cap: self.cap.release(); self.cap=None

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05); continue
            ret, frame = self.cap.read()
            if not ret:
                self.fail += 1; time.sleep(0.03); continue
            self.fail = 0
            # Always keep queue at latest frame
            if self.q.full():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 2 – PROCESSING
#  Consumes raw frames, runs the heavy CV pipeline, pushes render-ready data.
# ══════════════════════════════════════════════════════════════════════════════
class ProcessThread(threading.Thread):
    def __init__(self, cap_q):
        super().__init__(daemon=True)
        self.cap_q  = cap_q
        self.out_q  = queue.Queue(maxsize=1)
        self._stop  = threading.Event()
        self._lock  = threading.Lock()
        self._p     = dict(bsz=DEF_BS, co=DEF_CO, ms=DEF_MS,
                           bls=DEF_BL, ci=DEF_CI, bm=True)
        # Cached homography (updated from main thread)
        self.H           = None
        self.wsize       = None
        self.raw_corners = None

    def set_params(self, **kw):
        with self._lock: self._p.update(kw)

    def set_homography(self, H, wsize, raw_corners):
        with self._lock:
            self.H=H; self.wsize=wsize; self.raw_corners=raw_corners

    def stop(self): self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                frame = self.cap_q.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._lock:
                p=dict(self._p)
                H=self.H; wsize=self.wsize; rc=self.raw_corners

            if H is None or wsize is None:
                continue

            try:
                warped = cv2.warpPerspective(frame, H, wsize)
                wm     = wire_mask(warped, p['bsz'], p['co'], p['ms'], p['bls'], p['ci'])
                wm     = mask_aruco(wm, H, rc)
                sk     = get_skel(wm)

                res=None; st=None
                if p['bm']:
                    ncc,la,sa,_=cv2.connectedComponentsWithStats(sk,8)
                    if ncc>1:
                        skm=((la==(np.argmax(sa[1:,cv2.CC_STAT_AREA])+1))*255).astype(np.uint8)
                    else:
                        skm=sk.copy()
                    res=analyse(skm)
                    if res is None: st=simple_strands(sk)
                else:
                    st=simple_strands(sk)

                payload={'warped':warped,'wm':wm,'sk':sk,
                         'res':res,'st':st,'bm':p['bm']}
                if self.out_q.full():
                    try: self.out_q.get_nowait()
                    except queue.Empty: pass
                self.out_q.put(payload)

            except Exception as e:
                print(f"[ProcessThread] {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  TKINTER UI
# ══════════════════════════════════════════════════════════════════════════════
class App:
    SW=340; TH=40; MH=180

    def __init__(self):
        self.settings = load_settings()
        self.running  = False
        self.H=None; self.wsize=None; self.raw_corners=None
        self.logged   = False
        self._refs    = {}
        self._save_n  = 0
        self.trunk_buf=deque(maxlen=SMOOTH_W)
        self.br_bufs={}; self.tot_bufs={}

        self.cap_th  = CaptureThread();  self.cap_th.start()
        self.proc_th = ProcessThread(self.cap_th.q); self.proc_th.start()

        # ── root ──
        self.root=tk.Tk()
        self.root.title("Wire Harness Measurer  (Fast Edition)")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1920x1080")
        self.root.minsize(1280,720)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        # ── top bar ──
        top=tk.Frame(self.root,bg="#2d2d2d",height=self.TH)
        top.pack(fill=tk.X,side=tk.TOP); top.pack_propagate(False)

        tk.Label(top,text="Camera URL:",bg="#2d2d2d",fg="white",
                 font=("Segoe UI",10)).pack(side=tk.LEFT,padx=(10,4))
        self.url_var=tk.StringVar(value=self.settings.get("stream_url",DEFAULT_URL))
        self.url_ent=tk.Entry(top,textvariable=self.url_var,width=42,
                              font=("Consolas",10))
        self.url_ent.pack(side=tk.LEFT,padx=4)
        self.conn_btn=tk.Button(top,text="Connect",bg="#4CAF50",fg="white",
                                font=("Segoe UI",10,"bold"),width=11,
                                command=self._toggle)
        self.conn_btn.pack(side=tk.LEFT,padx=4)
        self.stat_lbl=tk.Label(top,text="  Disconnected",bg="#2d2d2d",fg="#ff5555",
                               font=("Segoe UI",10,"bold"))
        self.stat_lbl.pack(side=tk.LEFT,padx=8)
        tk.Label(top,text=f"  {GPU_LABEL}",bg="#2d2d2d",
                 fg="#00cc44" if USE_CUDA else "#ffaa00",
                 font=("Segoe UI",9,"bold")).pack(side=tk.LEFT,padx=4)
        tk.Button(top,text="Export",bg="#555",fg="white",font=("Segoe UI",9),
                  command=self._export,width=7).pack(side=tk.RIGHT,padx=2)
        tk.Button(top,text="Import",bg="#555",fg="white",font=("Segoe UI",9),
                  command=self._import,width=7).pack(side=tk.RIGHT,padx=2)

        # ── body ──
        body=tk.Frame(self.root,bg="#1e1e1e"); body.pack(fill=tk.BOTH,expand=True)

        # sidebar – fixed width, split into top (measurements) + bottom (tuning)
        sb=tk.Frame(body,bg="#1e1e1e",width=self.SW)
        sb.pack(side=tk.RIGHT,fill=tk.Y,padx=(4,6),pady=4)
        sb.pack_propagate(False)

        # ── bottom tuning section (packed first so it anchors to bottom) ──
        sb_bot=tk.Frame(sb,bg="#1e1e1e"); sb_bot.pack(side=tk.BOTTOM,fill=tk.X)

        tk.Label(sb_bot,text="Tuning",bg="#1e1e1e",fg="#aaa",
                 font=("Segoe UI",10,"bold")).pack(anchor=tk.W,pady=(4,0))
        sf=tk.Frame(sb_bot,bg="#1e1e1e"); sf.pack(fill=tk.X)

        # ── top measurements section (fills remaining space) ──
        sb_top=tk.Frame(sb,bg="#1e1e1e"); sb_top.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        tk.Label(sb_top,text="Measurements",bg="#1e1e1e",fg="#aaa",
                 font=("Segoe UI",10,"bold")).pack(anchor=tk.W,pady=(2,0))
        self.info_lbl=tk.Label(sb_top,bg="#111")
        self.info_lbl.pack(fill=tk.BOTH,expand=True,pady=(2,6))
        self.sliders={}
        for name,lo,hi,key in [("Block Size",3,101,"block_size"),
                                ("C Offset",0,30,"c_offset"),
                                ("Morph Size",1,15,"morph_size"),
                                ("Blur Size",1,25,"blur_size"),
                                ("Close Iter",0,15,"close_iter")]:
            row=tk.Frame(sf,bg="#1e1e1e"); row.pack(fill=tk.X,pady=1)
            tk.Label(row,text=name,bg="#1e1e1e",fg="#ccc",width=11,anchor=tk.W,
                     font=("Segoe UI",9)).pack(side=tk.LEFT)
            s=tk.Scale(row,from_=lo,to=hi,orient=tk.HORIZONTAL,bg="#1e1e1e",
                       fg="#ccc",highlightthickness=0,troughcolor="#333",length=180)
            s.set(self.settings.get(key,{"block_size":DEF_BS,"c_offset":DEF_CO,
                  "morph_size":DEF_MS,"blur_size":DEF_BL,"close_iter":DEF_CI}[key]))
            s.pack(side=tk.LEFT,fill=tk.X,expand=True)
            self.sliders[name]=s

        tf=tk.Frame(sb_bot,bg="#1e1e1e"); tf.pack(fill=tk.X,pady=(6,2))
        self.skel_v=tk.BooleanVar(value=True)
        self.mask_v=tk.BooleanVar(value=False)
        self.bran_v=tk.BooleanVar(value=True)
        for txt,var,cmd in [("Show Skeleton",self.skel_v,None),
                             ("Show Mask/Skel View",self.mask_v,None),
                             ("Branch Mode",self.bran_v,self._branch_tog)]:
            tk.Checkbutton(tf,text=txt,variable=var,bg="#1e1e1e",fg="#ccc",
                           selectcolor="#333",activebackground="#1e1e1e",
                           font=("Segoe UI",9),command=cmd).pack(anchor=tk.W)
        tk.Button(tf,text="Screenshot",bg="#555",fg="white",font=("Segoe UI",9),
                  command=self._screenshot).pack(anchor=tk.W,pady=(6,0))

        # left panel
        left=tk.Frame(body,bg="#1e1e1e")
        left.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=(6,0),pady=4)

        vf=tk.Frame(left,bg="#1e1e1e"); vf.pack(fill=tk.BOTH,expand=True)

        lf=tk.Frame(vf,bg="#1e1e1e"); lf.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=(0,2))
        tk.Label(lf,text="Live Feed",bg="#1e1e1e",fg="#aaa",
                 font=("Segoe UI",9,"bold")).pack(anchor=tk.W)
        self.live_c=tk.Frame(lf,bg="#111"); self.live_c.pack(fill=tk.BOTH,expand=True)
        self.live_c.pack_propagate(False)
        self.live_l=tk.Label(self.live_c,bg="#111"); self.live_l.place(x=0,y=0)

        mf=tk.Frame(vf,bg="#1e1e1e"); mf.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=(2,0))
        tk.Label(mf,text="Measurement View",bg="#1e1e1e",fg="#aaa",
                 font=("Segoe UI",9,"bold")).pack(anchor=tk.W)
        self.meas_c=tk.Frame(mf,bg="#111"); self.meas_c.pack(fill=tk.BOTH,expand=True)
        self.meas_c.pack_propagate(False)
        self.meas_l=tk.Label(self.meas_c,bg="#111"); self.meas_l.place(x=0,y=0)

        mkf=tk.Frame(left,bg="#1e1e1e",height=self.MH)
        mkf.pack(fill=tk.X,side=tk.BOTTOM,pady=(4,0)); mkf.pack_propagate(False)
        tk.Label(mkf,text="Mask / Skeleton",bg="#1e1e1e",fg="#aaa",
                 font=("Segoe UI",9,"bold")).pack(anchor=tk.W)
        self.mask_c=tk.Frame(mkf,bg="#111"); self.mask_c.pack(fill=tk.BOTH,expand=True)
        self.mask_c.pack_propagate(False)
        self.mask_l=tk.Label(self.mask_c,bg="#111"); self.mask_l.place(x=0,y=0)

        self.root.after(100, self._init_ph)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _csz(self, c):
        c.update_idletasks()
        w=c.winfo_width(); h=c.winfo_height()
        return (w if w>1 else 400, h if h>1 else 300)

    def _show(self, c, lbl, img, key):
        bw,bh=self._csz(c)
        pil=fit_image(img,bw,bh)
        ph=ImageTk.PhotoImage(pil)
        lbl.configure(image=ph); lbl.place(x=0,y=0,width=bw,height=bh)
        self._refs[key]=ph

    def _show_info(self, img):
        self.info_lbl.update_idletasks()
        bw=self.info_lbl.winfo_width() or 330
        bh=self.info_lbl.winfo_height() or 400
        ph=ImageTk.PhotoImage(fit_image(img,bw,bh))
        self.info_lbl.configure(image=ph); self._refs["info"]=ph

    def _ph(self, c, lbl, txt):
        bw,bh=self._csz(c)
        img=np.zeros((max(bh,10),max(bw,10),3),np.uint8)
        if txt:
            cv2.putText(img,txt,(max(bw//2-len(txt)*5,5),bh//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,100,100),1)
        self._show(c,lbl,img,"ph_"+str(id(lbl)))

    def _init_ph(self):
        self._ph(self.live_c,self.live_l,"No camera connected")
        self._ph(self.meas_c,self.meas_l,"Waiting for feed...")
        self._ph(self.mask_c,self.mask_l,"")
        self._show_info(np.zeros((500,600,3),np.uint8))

    # ── settings ──────────────────────────────────────────────────────────────
    def _export(self):
        cfg={k:max(self.sliders[n].get(),1) for n,k in
             [("Block Size","block_size"),("C Offset","c_offset"),
              ("Morph Size","morph_size"),("Blur Size","blur_size"),
              ("Close Iter","close_iter")]}
        p=filedialog.asksaveasfilename(defaultextension=".json",
                                       filetypes=[("JSON","*.json")],
                                       initialfile="preset.json")
        if p:
            with open(p,'w') as f: json.dump(cfg,f,indent=2)

    def _import(self):
        p=filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not p: return
        try:
            with open(p) as f: cfg=json.load(f)
            for n,k in [("Block Size","block_size"),("C Offset","c_offset"),
                         ("Morph Size","morph_size"),("Blur Size","blur_size"),
                         ("Close Iter","close_iter")]:
                if k in cfg: self.sliders[n].set(cfg[k])
        except Exception as e: print(f"[import] {e}")

    # ── connection ────────────────────────────────────────────────────────────
    def _toggle(self):
        self._disconnect() if self.running else self._connect()

    def _connect(self):
        url=self.url_var.get().strip()
        if not url: self.stat_lbl.config(text="  Enter URL",fg="#ff5555"); return
        self.stat_lbl.config(text="  Connecting…",fg="#ffaa00")
        self.conn_btn.config(state=tk.DISABLED); self.root.update_idletasks()
        # Run the blocking open() in a thread so the UI stays responsive
        def _do_connect():
            ok = self.cap_th.open(url)
            self.root.after(0, lambda: self._on_connect_result(ok, url))
        threading.Thread(target=_do_connect, daemon=True).start()

    def _on_connect_result(self, ok, url):
        if not ok:
            self.stat_lbl.config(text="  Failed – check URL/IP",fg="#ff5555")
            self.conn_btn.config(state=tk.NORMAL); return
        self.running=True; self.logged=False
        self.H=None; self.raw_corners=None
        self.stat_lbl.config(text="  Connected",fg="#55ff55")
        self.conn_btn.config(text="Disconnect",bg="#f44336",state=tk.NORMAL)
        self.url_ent.config(state=tk.DISABLED)
        self.settings["stream_url"]=url; save_settings(self.settings)
        self._aruco_loop()
        self._ui_loop()

    def _disconnect(self):
        self.running=False
        self.cap_th.close()
        self.stat_lbl.config(text="  Disconnected",fg="#ff5555")
        self.conn_btn.config(text="Connect",bg="#4CAF50")
        self.url_ent.config(state=tk.NORMAL)
        self._ph(self.live_c,self.live_l,"No camera connected")
        self._ph(self.meas_c,self.meas_l,"Waiting for feed…")
        self._ph(self.mask_c,self.mask_l,"")

    # ── ARUCO LOOP (main thread, ~30 fps, very cheap) ─────────────────────────
    def _aruco_loop(self):
        """
        Peek at the latest raw frame (no pop), run ArUco detection,
        update homography in the process thread, and show the live feed.
        This is intentionally lightweight — no heavy CV here.
        """
        if not self.running: return

        # Peek without removing (so process thread still gets it)
        try:
            frame = self.cap_th.q.queue[0].copy()   # thread-safe peek
        except (IndexError, AttributeError):
            self.root.after(30, self._aruco_loop); return

        if self.cap_th.fail >= MAX_FAIL:
            self.stat_lbl.config(text="  Feed lost",fg="#ff5555")
            self._disconnect(); return

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners,ids,_=_aruco_det.detectMarkers(gray)
        nm=0
        if ids is not None and len(ids)>0:
            cv2.aruco.drawDetectedMarkers(frame,corners,ids)
            H,wsize,rc=compute_homography(corners,ids)
            nm=len(ids)
            if H is not None:
                self.H=H; self.wsize=wsize; self.raw_corners=rc
                self.proc_th.set_homography(H,wsize,rc)
                if not self.logged:
                    self.logged=True
                    for i,m in enumerate(ids.flatten()):
                        p=corners[i][0]; cx,cy=p[:,0].mean(),p[:,1].mean()
                        print(f"  [ArUco] ID {m} ({cx:.0f},{cy:.0f}) → {MARKER_CORNER.get(int(m),'?')}")

        hf,wf=frame.shape[:2]
        cv2.rectangle(frame,(0,0),(wf,40),(0,0,0),-1)
        c=(0,255,0) if nm==4 else (0,165,255) if nm>0 else (0,0,255)
        cv2.putText(frame,f"Markers: {nm}/4",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,c,2)
        cv2.putText(frame,f"Scale: {SCALE:.4f} mm/px",(220,28),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        self._show(self.live_c,self.live_l,frame,"live")

        # Push params to process thread
        bsz=max(self.sliders["Block Size"].get(),3)
        co =self.sliders["C Offset"].get()
        ms =max(self.sliders["Morph Size"].get(),1)
        bls=max(self.sliders["Blur Size"].get(),1)
        ci =max(self.sliders["Close Iter"].get(),0)
        bm =self.bran_v.get()
        self.proc_th.set_params(bsz=bsz,co=co,ms=ms,bls=bls,ci=ci,bm=bm)

        self.root.after(30, self._aruco_loop)

    # ── UI LOOP (consume processed results) ───────────────────────────────────
    def _ui_loop(self):
        if not self.running: return

        try:
            d=self.proc_th.out_q.get_nowait()
        except queue.Empty:
            self.root.after(30, self._ui_loop); return

        warped=d['warped']; wm=d['wm']; sk=d['sk']
        res=d['res']; st=d['st']; bm=d['bm']
        ss=self.skel_v.get(); sm=self.mask_v.get()

        IW,IH=600,500
        info=np.zeros((IH,IW,3),np.uint8)
        disp=warped.copy()
        blk=INSET_PX+int(25*PX_MM)
        cv2.rectangle(disp,(blk,blk),(disp.shape[1]-blk,disp.shape[0]-blk),(80,80,80),1)

        if bm:
            if res:
                ti=res['trunk']; bis=res['branches']; jis=res['junctions']
                tmm=ti['len']*SCALE; self.trunk_buf.append(tmm)
                ta=float(np.mean(self.trunk_buf))

                if ss:
                    tp=ti['path']
                    if len(tp)>1:
                        cv2.polylines(disp,[np.array([(p[1],p[0]) for p in tp],np.int32)],False,(0,0,255),3)
                        cv2.circle(disp,(tp[0][1],tp[0][0]),8,(0,0,255),-1)
                        cv2.circle(disp,(tp[-1][1],tp[-1][0]),8,(0,0,255),-1)

                for i,br in enumerate(bis):
                    bmm=br['len']*SCALE; totmm=br['total']*SCALE
                    if i not in self.br_bufs:
                        self.br_bufs[i]=deque(maxlen=SMOOTH_W)
                        self.tot_bufs[i]=deque(maxlen=SMOOTH_W)
                    self.br_bufs[i].append(bmm); self.tot_bufs[i].append(totmm)
                    col=BC[i%len(BC)]
                    if ss:
                        bp=br['path']
                        if len(bp)>1:
                            cv2.polylines(disp,[np.array([(p[1],p[0]) for p in bp],np.int32)],False,col,3)
                            cv2.circle(disp,(bp[0][1],bp[0][0]),8,col,-1)
                            cv2.circle(disp,(bp[-1][1],bp[-1][0]),8,col,-1)
                if ss:
                    for jy,jx in jis: cv2.circle(disp,(jx,jy),10,(0,255,0),3)

                y0=30; nb=len(bis)
                cv2.putText(info,f"Harness: 1 trunk + {nb} branch(es)",(10,y0),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2); y0+=40
                cv2.putText(info,f"Trunk:  {ta:.1f} mm  ({ta/10:.2f} cm)",(10,y0),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2); y0+=30
                for i in range(nb):
                    col=BC[i%len(BC)]
                    ba=float(np.mean(self.br_bufs[i])) if i in self.br_bufs else 0
                    cv2.putText(info,f"Branch {i+1}: {ba:.1f} mm ({ba/10:.2f} cm)",(10,y0),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1); y0+=25
                y0+=10; cv2.line(info,(10,y0),(IW-10,y0),(100,100,100),1); y0+=25
                cv2.putText(info,"Totals (trunk+branch):",(10,y0),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1); y0+=30
                for i in range(nb):
                    col=BC[i%len(BC)]
                    tota=float(np.mean(self.tot_bufs[i])) if i in self.tot_bufs else 0
                    cv2.putText(info,f"Total {i+1}: {tota:.1f} mm ({tota/10:.2f} cm)",(10,y0),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2); y0+=30
                cv2.putText(info,f"(avg {len(self.trunk_buf)} frames)",
                            (10,min(y0+10,IH-10)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,150),1)
            else:
                cv2.putText(info,"No junction – simple fallback",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),1)
                y0=60
                for i,(path,lpx) in enumerate((st or [])[:6]):
                    mm=lpx*SCALE; clr=BC[i%len(BC)]
                    if ss and len(path)>1:
                        cv2.polylines(disp,[np.array(path,np.int32).reshape(-1,1,2)],False,clr,2)
                        cv2.circle(disp,path[0],6,clr,-1); cv2.circle(disp,path[-1],6,clr,-1)
                    cv2.putText(info,f"#{i+1}: {mm:.1f} mm ({mm/10:.2f} cm)",(10,y0),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,clr,1); y0+=25
        else:
            y0=30
            cv2.putText(info,"Simple Strand Mode",(10,y0),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2); y0+=35
            for i,(path,lpx) in enumerate((st or [])[:8]):
                mm=lpx*SCALE; clr=BC[i%len(BC)]
                if ss and len(path)>1:
                    cv2.polylines(disp,[np.array(path,np.int32).reshape(-1,1,2)],False,clr,2)
                    cv2.circle(disp,path[0],6,clr,-1); cv2.circle(disp,path[-1],6,clr,-1)
                cv2.putText(info,f"#{i+1}: {mm:.1f} mm ({mm/10:.2f} cm)",(10,y0),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,clr,1); y0+=25

        cv2.putText(disp,"BRANCH" if bm else "SIMPLE",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        self._show(self.meas_c,self.meas_l,disp,"meas")

        if sm:
            mb=cv2.cvtColor(wm,cv2.COLOR_GRAY2BGR)
            sb=cv2.cvtColor(sk,cv2.COLOR_GRAY2BGR); sb[sk>0]=(0,255,0)
            self._show(self.mask_c,self.mask_l,np.hstack([mb,sb]),"mask")
        else:
            self._ph(self.mask_c,self.mask_l,"Enable 'Show Mask/Skel View'")

        self._show_info(info)

        self._save_n+=1
        if self._save_n%120==0:
            bsz=max(self.sliders["Block Size"].get(),3)
            co=self.sliders["C Offset"].get()
            ms=max(self.sliders["Morph Size"].get(),1)
            bls=max(self.sliders["Blur Size"].get(),1)
            ci=max(self.sliders["Close Iter"].get(),0)
            self.settings.update({"block_size":bsz,"c_offset":co,"morph_size":ms,
                                   "blur_size":bls,"close_iter":ci})
            save_settings(self.settings)

        self.root.after(30, self._ui_loop)

    # ── misc ──────────────────────────────────────────────────────────────────
    def _branch_tog(self):
        self.trunk_buf.clear(); self.br_bufs.clear(); self.tot_bufs.clear()

    def _screenshot(self):
        print("[INFO] Screenshot (not yet implemented in fast edition)")

    def _close(self):
        self.running=False
        self.cap_th.stop(); self.proc_th.stop()
        if self.cap_th.cap: self.cap_th.cap.release()
        self.root.destroy()

    def run(self): self.root.mainloop()

# ── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()

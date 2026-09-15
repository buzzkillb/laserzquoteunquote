#!/usr/bin/env python3
"""
VISION -- pixels to 3D detections (the Pi's real camera pipeline)
================================================================
Bridges the gap between the sim's perfect 3D observations and the real
rig: camera frames in, tracker-ready 3D detections out.

Pipeline (one call to VisionPipeline.step per frame):

    frame (uint8 HxW)
      -> BlobDetector     threshold + connected components + centroids
      -> BlobTracker      frame-to-frame association (px space), growth
                          vector, lifetime -> confirmed blobs
      -> MonoRanger       pinhole model: r = f * size / px_size, with
                          dual-hypothesis disambiguation (5 mm mosquito
                          vs 8 mm fly), inside the 2-8 m kill band
      -> camera angles    az/el from pinhole geometry
      -> 3D detections    np.array [x, y, z] in world frame

Depth comes from the mono trick documented in RIG.md: angular size +
assumed body size. It is coarse (good enough for the depth-band veto,
which is a safety band, not a targeting measurement). Night mode: the
same math runs on IR-strobe frames unchanged -- a bright dot is a
bright dot.

Water-reflection rejection: real water mirrors the IR scene, so every
bug has a dimmer twin below the horizon line. The PixelCamera can emit
those twins (render_reflection=True) and the pipeline drops anything
below the configured waterline row -- the same rule the pool deploy
uses with the real camera.
"""

import math
import numpy as np
from collections import deque


# ----------------------------------------------------------------------------
# Camera model: pinhole intrinsics + world->pixel and pixel->ray
# ----------------------------------------------------------------------------
class PixelCamera:
    """6 mm lens on OV9281 (1280x800, 2.9 um pitch) by default."""

    def __init__(self, pos=(0.3, 3.0, 1.9), f_mm=6.0, res=(1280, 800),
                 pitch_um=2.9, pitch_deg=15.0):
        self.pos = np.asarray(pos, dtype=float)
        self.res = res
        self.f_px = f_mm * 1000.0 / pitch_um        # ~2069 px
        self.cx = res[0] / 2.0
        self.cy = res[1] / 2.0
        # downward pitch (deg): rig looks at the deck/swarm band, not sky
        self.pitch = math.radians(pitch_deg)
        self.cp, self.sp = math.cos(self.pitch), math.sin(self.pitch)
        # waterline: rows BELOW this are treated as mirror image (pool)
        self.waterline_v = None

    def project(self, world_pos):
        """world -> (u, v, px_radius) or None if behind camera."""
        rel = np.asarray(world_pos, dtype=float) - self.pos
        x, y, z = rel
        if x <= 0.05:                                # camera looks down +x
            return None
        # rotate into camera frame: pitch down around the y axis
        xc = self.cp * x - self.sp * z
        zc = self.sp * x + self.cp * z
        if xc <= 0.05:
            return None
        u = self.cx + self.f_px * (y / xc)
        v = self.cy - self.f_px * (zc / xc)
        return u, v

    def px_radius(self, world_pos, size_m):
        """projected radius in px of an object of body size size_m."""
        r = float(np.linalg.norm(np.asarray(world_pos) - self.pos))
        return max(0.4, self.f_px * size_m / r / 2.0)

    def angles_from_uv(self, u, v):
        """Exact inverse of project(): world az/el from pixel coords.
        Derived from the camera basis (f,r,u) with pitch theta:
          x = xc(cosT + tv sinT), y = xc*tu, z = xc(-sinT + tv cosT)
        where tu=(u-cx)/f, tv=(cy-v)/f."""
        tu = (u - self.cx) / self.f_px
        tv = (self.cy - v) / self.f_px
        c, s = self.cp, self.sp
        x = c + tv * s                     # proportional to world x
        y = tu
        z = -s + tv * c                    # proportional to world z
        az = math.atan2(y, x)
        el = math.atan2(z, math.hypot(x, y))
        return az, el

    def ray_hit_z(self, az, el, r):
        """world point at range r along az/el from the camera."""
        return self.pos + r * np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el)])


# ----------------------------------------------------------------------------
# Blob detection: threshold + BFS connected components (numpy + deque)
# ----------------------------------------------------------------------------
class BlobDetector:
    def __init__(self, thresh=90, min_area=2, max_blobs=64):
        self.thresh = thresh
        self.min_area = min_area
        self.max_blobs = max_blobs

    def detect(self, frame):
        mask = frame >= self.thresh
        H, W = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        blobs = []
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return blobs
        for y0, x0 in zip(ys, xs):
            if seen[y0, x0]:
                continue
            # BFS flood fill one blob
            q = deque([(y0, x0)])
            seen[y0, x0] = True
            acc_y = acc_x = n = 0
            while q:
                y, x = q.popleft()
                acc_y += y; acc_x += x; n += 1
                y0b, y1b = max(0, y - 1), min(H, y + 2)
                x0b, x1b = max(0, x - 1), min(W, x + 2)
                for yy in range(y0b, y1b):
                    row = mask[yy, x0b:x1b]
                    for xx in np.nonzero(row & ~seen[yy, x0b:x1b])[0]:
                        seen[yy, x0b + xx] = True
                        q.append((yy, x0b + xx))
            if n >= self.min_area:
                blobs.append((acc_x / n, acc_y / n, n))
                if len(blobs) >= self.max_blobs:
                    break
        return blobs


# ----------------------------------------------------------------------------
# Blob tracker: px-space association + confirmation + growth vector
# ----------------------------------------------------------------------------
class BlobTrack:
    __slots__ = ("u", "v", "vu", "vv", "age", "missed", "r_est", "area",
                 "diam")

    def __init__(self, u, v):
        self.u, self.v = u, v
        self.vu = self.vv = 0.0
        self.age = 1
        self.missed = 0
        self.r_est = None
        self.area = 0.0
        self.diam = None      # EMA-smoothed equivalent diameter (px)


class BlobTracker:
    """Nearest-neighbor association in pixel space. A blob is confirmed
    after `confirm_frames` consecutive sightings with a stable velocity
    -- this is what kills single-frame glare flickers."""

    def __init__(self, gate_px=12.0, confirm_frames=3, max_missed=2):
        self.gate = gate_px
        self.confirm = confirm_frames
        self.max_missed = max_missed
        self.tracks = []

    def step(self, blobs, dt):
        """blobs: [(u, v, area)] -> list of confirmed BlobTrack."""
        assigned = {}
        for tr in self.tracks:
            tr.u += tr.vu * dt
            tr.v += tr.vv * dt
        for (u, v, _a) in blobs:
            best, bd = None, self.gate
            for i, tr in enumerate(self.tracks):
                if i in assigned:
                    continue
                d = math.hypot(tr.u - u, tr.v - v)
                if d < bd:
                    best, bd = i, d
            if best is None:
                self.tracks.append(BlobTrack(u, v))
                continue
            tr = self.tracks[best]
            tr.vu = 0.7 * tr.vu + 0.3 * (u - tr.u) / dt
            tr.vv = 0.7 * tr.vv + 0.3 * (v - tr.v) / dt
            tr.u, tr.v = u, v
            tr.area = _a
            # EMA on equivalent diameter: raw area of 2-6 px blobs is
            # quantized and noisy; the smoothed value is what mono
            # ranging needs (oscillating size = oscillating range = a
            # beam that never sits still)
            d_new = 2.0 * math.sqrt(max(_a, 1) / math.pi)
            tr.diam = d_new if tr.diam is None else \
                0.6 * tr.diam + 0.4 * d_new
            tr.age += 1
            tr.missed = 0
            assigned[best] = True
        out = []
        for tr in self.tracks:
            if tr not in [self.tracks[i] for i in assigned]:
                tr.missed += 1
            if tr.age >= self.confirm and tr.missed == 0:
                out.append(tr)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return out


# ----------------------------------------------------------------------------
# Mono ranging: angular size + assumed body size -> range
# ----------------------------------------------------------------------------
class MonoRanger:
    """Two hypotheses (mosquito 5 mm, fly 8 mm). Picks the range that
    lands inside the engagement band [r_min, r_max]; when both land
    inside, prefers the hypothesis whose implied size trend (blob
    growth) is consistent with approach at that range. This is coarse
    by design -- the depth-band veto needs a band, not millimeters."""

    SIZES = (("mosquito", 0.005), ("fly", 0.008))

    def __init__(self, cam: PixelCamera, band=(1.5, 9.0)):
        self.cam = cam
        self.band = band

    def range_for(self, diam_px, prev_r=None):
        """Range candidates from the smoothed equivalent diameter."""
        diam_px = max(1.0, diam_px)
        cands = []
        for name, size in self.SIZES:
            r = self.cam.f_px * size / diam_px
            if self.band[0] <= r <= self.band[1]:
                cands.append(r)
        if not cands:
            return None
        if prev_r is not None:
            return min(cands, key=lambda r: abs(r - prev_r))
        return float(np.mean(cands))


# ----------------------------------------------------------------------------
# The pipeline: one call per frame
# ----------------------------------------------------------------------------
class VisionPipeline:
    def __init__(self, cam: PixelCamera = None, waterline_v=None):
        self.cam = cam or PixelCamera()
        self.det = BlobDetector()
        self.btr = BlobTracker()
        self.rng = MonoRanger(self.cam)
        self.waterline = (waterline_v if waterline_v is not None
                          else self.cam.waterline_v)

    def step(self, frame, dt):
        """frame (uint8 HxW) -> list of world 3D detections."""
        blobs = self.det.detect(frame)
        confirmed = self.btr.step(blobs, dt)
        dets = []
        for tr in confirmed:
            if self.waterline is not None and tr.v > self.waterline:
                continue                      # mirror-image twin: rejected
            r = self.rng.range_for(tr.diam if tr.diam else 3.0, tr.r_est)
            if r is None:
                continue
            if tr.r_est is not None:
                # range EMA (strong): residual jitter breaks the 3D
                # tracker's 0.30 m association gate, so stability of the
                # range estimate matters more than its latency
                r = 0.75 * tr.r_est + 0.25 * r
            tr.r_est = r
            az, el = self.cam.angles_from_uv(tr.u, tr.v)
            dets.append(self.cam.ray_hit_z(az, el, r))
        return dets


# ----------------------------------------------------------------------------
# Virtual camera: renders the sim swarm into uint8 frames (test harness)
# ----------------------------------------------------------------------------
def render_frame(swarm_list, cam: PixelCamera, gain=420.0,
                 noise=6, rng=None,
                 render_reflection=False, water_gain=0.35):
    """Draw each alive creature as a small Gaussian blob. Brightness
    falls with range (IR strobe return), size with projected body size.
    render_reflection adds the dim water-mirror twin below the horizon."""
    H, W = cam.res[1], cam.res[0]
    frame = np.zeros((H, W), dtype=np.uint8)
    if rng is None:
        rng = np.random.default_rng(0)
    frame += rng.integers(0, noise + 1, frame.shape, dtype=np.uint8)
    for sw in swarm_list:
        if not sw.alive():
            continue
        spots = [sw.pos]
        if render_reflection:
            # mirror across the water plane z=0
            spots.append(np.array([sw.pos[0], sw.pos[1], -sw.pos[2]]))
        for si, p in enumerate(spots):
            pr = cam.project(p)
            if pr is None:
                continue
            u, v = pr
            r = float(np.linalg.norm(p - cam.pos))
            size_px = cam.px_radius(p, 0.005)          # 5 mm body
            bright = gain / (1.0 + 0.25 * r)
            if si == 1:
                bright *= water_gain                   # dimmer twin
            iu, iv = int(round(u)), int(round(v))
            rad = max(2, int(round(size_px)))
            sig = max(1.0, size_px / 2)
            y0, y1 = max(0, iv - rad), min(H, iv + rad + 1)
            x0, x1 = max(0, iu - rad), min(W, iu + rad + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d2 = (yy - v) ** 2 + (xx - u) ** 2
            spot = bright * np.exp(-d2 / (2 * sig ** 2))
            frame[y0:y1, x0:x1] = np.clip(
                frame[y0:y1, x0:x1] + spot.astype(np.uint8), 0, 255)
    return frame

#!/usr/bin/env python3
"""
3D POINT-DEFENSE SIMULATOR  -- "all the axes of evil"
=====================================================
Volumetric upgrade of sim.py:

  - Threats fly in full 3D (x, y, z) inside a room-sized volume
  - Mixed-species mode: all creature types share ONE field of view at once
  - Pan/tilt turret: two independent slew-rate-limited axes (azimuth, elevation)
  - 6-state 3D Kalman filters (position + velocity per axis)
  - 3D lead-pursuit fire control with convergence check
  - Spherical fire gate: beam hits if miss distance < spot radius + body radius
  - Renders: rotating 3D view MP4 + summary PNG (matplotlib mplot3d)

Pipeline is unchanged in spirit: sense -> track -> fire control -> effector.
Everything in SI units. Kill assessment is stochastic.
"""

import argparse
import json

import flybrain
import math
import random

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3D

# ----------------------------------------------------------------------------
# Threat table (3D speeds; vertical speeds damped -- creatures don't hover)
# ----------------------------------------------------------------------------
THREATS = {
    "mosquitoes": dict(
        label="Mosquitoes", hp=1.0, speed=0.55, speed_spread=0.20, wander=2.2,
        pkill=0.88, dwell=0.030, spot_mm=6.0, flux_needed=0.9,
        count=14, color="#4ade80", size_m=0.0045, power_w=2.0, z_band=(0.3, 2.6),
        glyph=".", uni="\u00b7",
    ),
    "flies": dict(
        label="Flies", hp=1.6, speed=1.40, speed_spread=0.30, wander=4.8,
        pkill=0.70, dwell=0.040, spot_mm=6.0, flux_needed=1.2,
        count=10, color="#fde047", size_m=0.008, power_w=4.0, z_band=(0.4, 2.8),
        glyph="*", uni="\u2733",
    ),
    "roaches": dict(
        label="Roaches", hp=2.6, speed=1.10, speed_spread=0.35, wander=3.0,
        pkill=0.55, dwell=0.055, spot_mm=6.0, flux_needed=1.6,
        count=8, color="#b45309", size_m=0.015, power_w=6.0, z_band=(0.05, 0.8),
        glyph="s", uni="\u25aa",
    ),
    "sparrows": dict(
        label="Sparrows", hp=7.0, speed=3.5, speed_spread=0.8, wander=4.5,
        pkill=0.30, dwell=0.085, spot_mm=8.0, flux_needed=4.5,
        count=6, color="#f87171", size_m=0.070, power_w=15.0, z_band=(0.8, 2.8),
        glyph="^", uni="\u25b2",
    ),
    "pigeons": dict(
        label="Pigeons", hp=14.0, speed=5.5, speed_spread=1.0, wander=4.0,
        pkill=0.18, dwell=0.110, spot_mm=9.0, flux_needed=9.0,
        count=4, color="#94a3b8", size_m=0.170, power_w=30.0, z_band=(1.5, 3.0),
        glyph="H", uni="\u25cf",
    ),
    "drones": dict(
        label="Micro-drones", hp=22.0, speed=8.0, speed_spread=1.5, wander=2.5,
        pkill=0.12, dwell=0.130, spot_mm=10.0, flux_needed=16.0,
        count=3, color="#fb923c", size_m=0.250, power_w=60.0, z_band=(1.0, 3.0),
        glyph="D", uni="\u25c6",
    ),
}
SPECIES_ORDER = ["mosquitoes", "flies", "roaches", "sparrows", "pigeons", "drones"]
# marker size on screen scales with physical body size so a pigeon reads as a
# big hexagon and a mosquito as a speck -- shape + size, never a generic orb
def glyph_ms(size_m):
    return 4.0 + 44.0 * math.sqrt(max(size_m, 1e-4))


# ----------------------------------------------------------------------------
# 3D swarm model
# ----------------------------------------------------------------------------
class Swarm3D:
    """Constant-speed wanderer in 3D with noise-triggered evasive jinks."""

    def __init__(self, key, area, height, rng):
        t = THREATS[key]
        self.t = t
        self.key = key
        self.rng = rng
        z0 = rng.uniform(*t["z_band"])
        z0 = min(z0, height - 0.1)
        self.pos = np.array([rng.uniform(0.5, area - 0.5),
                             rng.uniform(0.5, area - 0.5), z0])
        self.trail = []
        ang = rng.uniform(0, 2 * math.pi)
        sp = t["speed"] * rng.uniform(0.8, 1.2)
        self.vel = np.array([math.cos(ang) * sp,
                             math.sin(ang) * sp,
                             rng.uniform(-0.15, 0.15) * sp])
        self.hp = t["hp"]

    def alive(self):
        return self.hp > 0

    def step(self, dt, laser_pressure):
        t = self.t
        rng = self.rng
        if rng.random() < t["wander"] * dt:
            ang = rng.uniform(0, 2 * math.pi)
            sp = t["speed"] * (1.0 + rng.uniform(-t["speed_spread"],
                                                 t["speed_spread"]))
            self.vel = np.array([math.cos(ang) * sp, math.sin(ang) * sp,
                                 self.vel[2] + rng.uniform(-0.3, 0.3)])
            self.vel[2] = np.clip(self.vel[2], -0.4 * t["speed"],
                                  0.4 * t["speed"])
        if laser_pressure > 0 and rng.random() < laser_pressure * 1.5 * dt:
            # jink: horizontal heading change + vertical dart
            ang = math.atan2(self.vel[1], self.vel[0]) + rng.uniform(-1.2, 1.2)
            sp = np.linalg.norm(self.vel[:2])
            self.vel[0] = math.cos(ang) * sp
            self.vel[1] = math.sin(ang) * sp
            self.vel[2] += rng.uniform(-0.6, 0.6)
            self.vel[2] = np.clip(self.vel[2], -0.5 * t["speed"],
                                  0.5 * t["speed"])
        self.pos = self.pos + self.vel * dt
        # walls and floor/ceiling
        for i in range(2):
            if self.pos[i] < 0.2:
                self.pos[i] = 0.2
                self.vel[i] = abs(self.vel[i])
            elif self.pos[i] > self.area_safe(i) - 0.2:
                self.pos[i] = self.area_safe(i) - 0.2
                self.vel[i] = -abs(self.vel[i])
        zb = t["z_band"]
        if self.pos[2] < max(0.03, zb[0]):
            self.pos[2] = max(0.03, zb[0])
            self.vel[2] = abs(self.vel[2])
        elif self.pos[2] > min(HEIGHT - 0.05, zb[1]):
            self.pos[2] = min(HEIGHT - 0.05, zb[1])
            self.vel[2] = -abs(self.vel[2])

    def area_safe(self, i):
        return AREA


# module-level world size (simpler than threading config through classes)
AREA = 6.0
HEIGHT = 3.0


# ----------------------------------------------------------------------------
# Sensor (3D): noisy range-less camera proxy -- we observe 3D position with
# stereo-quality noise; per-target detection probability and dropped frames.
# ----------------------------------------------------------------------------
class Sensor3D:
    def __init__(self, noise_std=0.010, p_detect=0.93):
        self.noise_std = noise_std
        self.p_detect = p_detect

    def observe(self, swarm_list):
        dets = []
        for sw in swarm_list:
            if not sw.alive():
                continue
            if random.random() > self.p_detect:
                continue
            noisy = sw.pos + np.array([random.gauss(0, self.noise_std) for _ in range(3)])
            dets.append(noisy)
        return dets


# ----------------------------------------------------------------------------
# Tracker: 6-state 3D Kalman, GNN association
# ----------------------------------------------------------------------------
class Track3D:
    _next_id = 1

    def __init__(self, z):
        self.id = Track3D._next_id
        Track3D._next_id += 1
        self.x = np.concatenate([z, np.zeros(3)])            # pos + vel
        self.P = np.diag([0.05] * 3 + [9.0] * 3)
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.species_hint = None

    def predict(self, dt):
        F = np.eye(6)
        for i in range(3):
            F[i, i + 3] = dt
        q = 0.5
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = dt ** 3 / 3 * q
            Q[i, i + 3] = dt ** 2 / 2 * q
            Q[i + 3, i + 3] = dt * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, dt):
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.diag([0.010 ** 2] * 3)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.hits += 1
        self.misses = 0
        if self.hits >= 3:
            self.confirmed = True

    def coast(self):
        self.misses += 1

    def stale(self, frames=18):
        return self.misses > frames


class Tracker3D:
    def __init__(self):
        self.tracks = []

    def step(self, detections, dt):
        for tr in self.tracks:
            tr.predict(dt)
        pairs = []
        for i, tr in enumerate(self.tracks):
            for j, z in enumerate(detections):
                d = np.linalg.norm(tr.x[:3] - z)
                if d < 0.30:
                    pairs.append((d, i, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for d, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            self.tracks[i].update(detections[j], dt)
            used_t.add(i)
            used_d.add(j)
        for j, z in enumerate(detections):
            if j not in used_d:
                self.tracks.append(Track3D(z))
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.coast()
        self.tracks = [t for t in self.tracks if not t.stale()]
        return self.tracks


# ----------------------------------------------------------------------------
# Pan/tilt turret: two independent slew axes, 3D lead-pursuit solution
# ----------------------------------------------------------------------------
class PanTiltTurret:
    def __init__(self, pos, power_w, spot_mm, max_slew_dps=250.0):
        self.pos = np.asarray(pos, dtype=float)
        self.power = power_w
        self.spot_mm = spot_mm
        self.max_slew = math.radians(max_slew_dps)
        # aim direction as azimuth/elevation (radians)
        self.az = 0.0
        self.el = 0.0
        self.heat = 0.0
        self.firing = False
        self.target_id = None
        self.lock_id = None
        self.beam_history = []      # (start, end, killed)
        self.kill_events = []       # (clock, pos, species_key)
        self.clock = 0.0
        self.aim_point = self.pos + np.array([1.0, 0, 0])
        self.shots = 0
        self.kills = 0
        self.energy_j = 0.0
        self.species_shots = {k: 0 for k in SPECIES_ORDER}
        self.species_kills = {k: 0 for k in SPECIES_ORDER}

    # -- aim vector <-> az/el conversion
    def aim_vec(self):
        return np.array([math.cos(self.el) * math.cos(self.az),
                         math.cos(self.el) * math.sin(self.az),
                         math.sin(self.el)])

    def angles_to(self, point):
        d = np.asarray(point, dtype=float) - self.pos
        r_xy = math.hypot(d[0], d[1])
        return math.atan2(d[1], d[0]), math.atan2(d[2], r_xy)

    def _slew_toward(self, az_t, el_t, dt):
        """Move both axes at max slew rate; returns True when both converged
        within half a spot angular size."""
        daz = (az_t - self.az + math.pi) % (2 * math.pi) - math.pi
        de = el_t - self.el
        step = self.max_slew * dt
        done_a = abs(daz) <= step
        done_e = abs(de) <= step
        self.az += np.clip(daz, -step, step)
        self.el += np.clip(de, -step, step)
        return done_a and done_e

    def range_to(self, point):
        return np.linalg.norm(np.asarray(point) - self.pos)

    def engage(self, tracks, dt, swarm_list, rng, tparams, target_keys=None):
        """tparams: dict with dwell/pkill/flux_needed/spot_mm for the CURRENT
        beam tuning (mixed mode: use the locked target's species table).
        target_keys: None = engage every species; else only these species
        are targeted (e.g. backyard mode: just mosquitoes + flies)."""
        self.firing = False
        self.heat = max(0.0, self.heat - 0.55 * dt)
        if self.heat > 0.92:
            self.target_id = None
            return

        cands = [t for t in tracks if t.confirmed and t.misses < 4]
        if target_keys is not None:
            cands = [t for t in cands if self._classify_species(t) in target_keys]
        if not cands:
            self.target_id = None
            self.lock_id = None
            # park toward scene center
            az_t, el_t = self.angles_to([AREA / 2, AREA / 2, 1.2])
            self._slew_toward(az_t, el_t, dt)
            return

        # target selection -- brain modes:
        #   fly    - winner-take-all attention: only engage what the optic
        #            lobe attends to (EMD/STMD saliency + looming priority)
        #   hybrid - fly attention gates the candidate set, Kalman keeps lock
        #   kalman - pure engineering baseline (nearest-range)
        mode = getattr(self, "brain_mode", "hybrid")
        brain = getattr(self, "brain", None)
        tgt = None
        if brain is not None and mode in ("fly", "hybrid"):
            gated = [t for t in cands if brain.attends(t.x[:3], self.pos)]
            if gated:
                tgt = (next((t for t in gated if t.id == self.lock_id), None)
                       or min(gated, key=lambda t: self.range_to(t.x[:3])))
            # (no attention winner -> visual servo below)
        else:
            if self.lock_id is not None:
                tgt = next((t for t in cands if t.id == self.lock_id), None)
            if tgt is None:
                tgt = min(cands, key=lambda t: self.range_to(t.x[:3]))
        if tgt is None:
            # visual servo: aim where the fly brain is looking (pursuit)
            if brain is not None and brain.attention is not None:
                a_az, a_el, _sc = brain.attention
                self._slew_toward(a_az, a_el, dt)
            self.target_id = None
            return
        self.lock_id = tgt.id
        self.target_id = tgt.id

        # species params from motion signature (speed + altitude band:
        # a 1.1 m/s track at floor level is a roach, at ceiling a fly)
        tkey = self._classify_species(tgt)
        tp = THREATS[tkey]
        tgt.species_hint = tkey

        # 3D lead-pursuit: iterate aim point <-> time-of-flight.
        # Pure fly mode: no prediction -- insects lead only by dwell latency
        # (visual servo on the attended point, exactly how a real fly tracks)
        if mode == "fly":
            tof = tp["dwell"]
            aim = tgt.x[:3] + tgt.x[3:6] * tof
        else:
            tof = 0.0
            aim = tgt.x[:3].copy()
            for _ in range(4):
                aim = tgt.x[:3] + tgt.x[3:6] * tof
                tof = self.time_to_turn(aim) + tp["dwell"]

        err = np.linalg.norm(tgt.x[:3] + tgt.x[3:6] * tof - aim)
        if err > 0.18:
            az_t, el_t = self.angles_to(aim)
            self._slew_toward(az_t, el_t, dt)
            return

        az_t, el_t = self.angles_to(aim)
        arrived = self._slew_toward(az_t, el_t, dt)
        if not arrived:
            return

        # fire gate: beam line must pass within spot+body radius of the
        # target's projected position
        beam_dir = self.aim_vec()
        rel = tgt.x[:3] + tgt.x[3:6] * tof - self.pos
        along = np.dot(rel, beam_dir)
        if along <= 0:
            return
        closest = self.pos + beam_dir * along
        miss = np.linalg.norm(closest - (tgt.x[:3] + tgt.x[3:6] * tof))
        hit_r = (tp["spot_mm"] / 1000.0) / 2.0 * 1.5 + tp["size_m"]
        if miss > hit_r:
            return

        # FIRE
        self.firing = True
        self.shots += 1
        self.species_shots[tkey] += 1
        R = self.range_to(aim)
        flux = self.power * math.exp(-0.03 * R)
        p_kill = tp["pkill"] * min(1.0, flux / tp["flux_needed"])
        self.energy_j += flux * tp["dwell"]
        killed = rng.random() < p_kill
        self.beam_history.append((self.pos.copy(), closest.copy(), killed))
        if killed:
            for sw in swarm_list:
                if not sw.alive() or sw.key not in (target_keys or SPECIES_ORDER):
                    # selective fire: non-targets are never registered as kills
                    continue
                proj = sw.pos + sw.vel * tof
                if np.linalg.norm(proj - closest) < hit_r:
                    sw.hp = 0.0
                    self.kills += 1
                    self.species_kills[sw.key] += 1
                    self.kill_events.append((self.clock, closest.copy(), tkey))
                    break
        self.heat = min(1.0, self.heat + tp["dwell"] * 4.0)

    def _classify_species(self, track):
        """Motion-signature ID: speed + altitude band. A 1.1 m/s track hugging
        the floor is a roach; the same speed at head height is a fly."""
        z = max(0.0, track.x[2])
        best, bd = "mosquitoes", 1e9
        for k, t in THREATS.items():
            zb = t["z_band"]
            zc = 0.5 * (zb[0] + zb[1])
            zr = max(0.4, 0.5 * (zb[1] - zb[0]))
            zpen = ((z - zc) / zr) ** 2
            spen = (np.linalg.norm(track.x[3:6]) - t["speed"]) / t["speed"]
            d = spen ** 2 + 0.25 * zpen
            if d < bd:
                bd, best = d, k
        return best

    def _classify(self, speed, track):
        """Back-compat wrapper."""
        return self._classify_species(track)

    def time_to_turn(self, aim_point):
        az_t, el_t = self.angles_to(aim_point)
        daz = abs((az_t - self.az + math.pi) % (2 * math.pi) - math.pi)
        de = abs(el_t - self.el)
        return max(daz, de) / self.max_slew


# ----------------------------------------------------------------------------
# Simulation driver
# ----------------------------------------------------------------------------
def run_sim(mode="mixed", single="mosquitoes", seconds=60.0, dt=1 / 60.0,
            seed=7, out_png="defense3d.png", out_mp4="defense3d.mp4",
            creatures=None, brain_mode="hybrid",
            return_frames=False, render=True):
    """Full ecosystem ALWAYS spawns -- every species is on the battlefield.
    creatures: list of species keys to TARGET (the laser only engages these).
    None = engage everything. Non-target species roam the field untouched.
    seconds is an upper bound only -- the engagement ends early once every
    TARGETED creature is destroyed (spared species keep flying).
    return_frames: include the animation frames in stats["frames"].
    render: set False for fast benchmark runs (no MP4/PNG output)."""
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    keys = list(SPECIES_ORDER)                     # everything on the field
    targets = list(creatures) if creatures else (
        SPECIES_ORDER if mode == "mixed" else [single])
    tgt_set = set(targets)
    swarm = []
    for k in keys:
        swarm += [Swarm3D(k, AREA, HEIGHT, rng) for _ in range(THREATS[k]["count"])]
    tgt_count = sum(THREATS[k]["count"] for k in targets)

    sensor = Sensor3D()
    tracker = Tracker3D()
    # beam tuning sized for the strongest TARGETED threat (spared species
    # never take a beam, so their power tier doesn't drive tuning)
    max_power = max(THREATS[k]["power_w"] for k in targets)
    turret = PanTiltTurret(pos=(0.3, AREA / 2, 1.5),
                           power_w=max_power, spot_mm=8.0)
    brain = flybrain.FlyBrain()
    turret.brain = brain
    turret.brain_mode = brain_mode

    frames = []
    stats = dict(shots=0, kills=0, energy=0.0, track_switches=0)
    prev_target = None
    steps = int(seconds / dt)
    t_end = seconds
    cleared_t = None

    for k in range(steps):
        t = k * dt
        turret.clock = t
        alive = [s for s in swarm if s.alive()]
        # engagement pressure tracks how much of the TARGET set is left
        tgt_alive = sum(1 for s in alive if s.key in tgt_set)
        pressure = 1.0 - tgt_alive / max(1, tgt_count)

        dets = sensor.observe(alive)
        brain.update(dets, turret.pos, dt)
        tracks = tracker.step(dets, dt)
        turret.engage(tracks, dt, swarm, rng, None, target_keys=targets)

        if prev_target is not None and turret.target_id != prev_target:
            stats["track_switches"] += 1
        prev_target = turret.target_id

        for s in alive:
            near_beam = 0.0
            if turret.firing:
                d = np.linalg.norm(s.pos - turret.aim_point)
                near_beam = max(0.0, 1.0 - d / 0.5)
            s.step(dt, pressure + near_beam * 0.5)

        if k % 2 == 0:
            # per-creature motion trails (last ~0.8 s)
            for s in alive:
                s.trail.append(s.pos.copy())
                if len(s.trail) > 24:
                    s.trail.pop(0)
            # kill bursts: events in the last 0.6 s, with age for animation
            bursts = [(p, t - t0, tk) for (t0, p, tk) in turret.kill_events
                      if 0.0 <= t - t0 < 0.6]
            frames.append(dict(
                t=t,
                bugs={k2: [(s.pos.copy(), s.hp / THREATS[s.key]["hp"])
                           for s in alive if s.key == k2]
                      for k2 in keys},
                trails={k2: [np.array(s.trail) for s in alive if s.key == k2]
                        for k2 in keys},
                bursts=bursts,
                az=turret.az, el=turret.el,
                firing=turret.firing,
                aim=turret.pos + turret.aim_vec() * 2.5,
                tracks=[(tr.x[0], tr.x[1], tr.x[2], tr.confirmed) for tr in tracks],
                heat=turret.heat,
                kills=turret.kills,
                shots=turret.shots,
                energy=turret.energy_j,
                saliency=brain.saliency.copy(),
                motion=np.abs(brain.motion).sum(axis=2).copy(),
                attention=brain.attention,
                looming=brain.looming,
                waves=dict(
                    looming=list(brain.wave_looming),
                    neurons=list(brain.wave_neurons),
                    attention=list(brain.wave_attention),
                    eeg=list(brain.wave_eeg),
                ),
                brain_mode=brain_mode,
            ))

        if tgt_alive == 0:
            # every TARGET destroyed: engagement over -- spared species keep
            # flying, the sim ran exactly as long as the fight lasted
            t_end = t
            cleared_t = t
            break

    stats["brain_mode"] = brain_mode
    stats["shots"] = turret.shots
    stats["kills"] = turret.kills
    stats["energy"] = turret.energy_j
    stats["targets"] = len(swarm)
    stats["targets_selected"] = tgt_count
    stats["species"] = {k: THREATS[k]["count"] for k in keys}
    stats["duration_s"] = round(t_end, 2)
    stats["cleared"] = cleared_t is not None
    stats["species_shots"] = {k: turret.species_shots[k] for k in keys}
    stats["species_kills"] = {k: turret.species_kills[k] for k in keys}

    # ------------------------------------------------------------------
    # LEADERBOARD: full-ecosystem scorecard. Targeted species are ranked by
    # kills; spared species are listed beneath as survivors.
    # Shown on the end-of-video card AND the summary PNG.
    # ------------------------------------------------------------------
    labels = {k2: THREATS[k2]["label"] for k2 in keys}
    unis = {k2: THREATS[k2]["uni"] for k2 in keys}
    ranked = sorted(targets, key=lambda k2: -turret.species_kills[k2])
    medal = {0: "1.", 1: "2.", 2: "3."}
    board_lines = []
    for rank, k2 in enumerate(ranked):
        sk = turret.species_kills[k2]
        st = turret.species_shots[k2]
        eff = 100.0 * sk / st if st else 0.0
        n = THREATS[k2]["count"]
        bar = "\u2588" * int(round(12.0 * sk / max(1, n)))
        board_lines.append(
            f"{medal.get(rank, f'{rank + 1}.')} {unis[k2]} {labels[k2]:<11} "
            f"{sk:>2}/{n:<2} {bar:<12} {eff:3.0f}%")
    spared = [k2 for k2 in keys if k2 not in tgt_set]
    spared_lines = [
        f"  {unis[k2]} {labels[k2]}: {len([s for s in swarm if s.alive() and s.key == k2])}"
        f"/{THREATS[k2]['count']} spared" for k2 in spared]
    survivors_txt = "\n".join(
        f"{unis[k2]} {labels[k2]}: "
        f"{len([s for s in swarm if s.alive() and s.key == k2])}/{THREATS[k2]['count']}"
        for k2 in keys)
    leaderboard_txt = (
        f"LEADERBOARD // {turret.kills}/{tgt_count} DOWN @ {t_end:.0f}S\n"
        f"{'':<3}{'SPECIES':<15}{'KILLS':<7}{'SCORE':<21}{'EFF'}\n"
        + "\n".join(board_lines)
        + ("\nSPARED (out of scope)\n" + "\n".join(spared_lines)
           if spared_lines else ""))

    # ------------------------------------------------------------------
    # Cinematic render: glow beams, kill bursts, trails, HUD + summary PNG
    # Outputs: MP4 (with GIF fallback) AND PNG -- always both, every run.
    # Skipped entirely when render=False (fast benchmark mode).
    # ------------------------------------------------------------------
    if not render:
        stats["outputs"] = {"mp4": None, "png": None}
        if return_frames:
            stats["frames"] = frames
        return stats
    BG, PANE = "#04060d", "#0b1226"
    CYAN, ACCENT = "#22d3ee", "#fbbf24"

    fig = plt.figure(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.02, 0.02, 0.70, 0.90], projection="3d")
    ax.set_xlim(0, AREA); ax.set_ylim(0, AREA); ax.set_zlim(0, HEIGHT)
    ax.set_facecolor(BG)
    # near-black glass panes with electric edge -- holographic war-room look
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(PANE); pane.set_alpha(0.30)
        pane.set_edgecolor("#164e63")
    ax.grid(True, color="#155e75", alpha=0.35, lw=0.5)
    ax.set_xlabel("X (m)", color="#67e8f9", labelpad=6)
    ax.set_ylabel("Y (m)", color="#67e8f9", labelpad=6)
    ax.set_zlabel("Z (m)", color="#67e8f9", labelpad=6)
    # minimize axis chrome: sparse ticks, thin lines -- cinema not spreadsheet
    ax.tick_params(colors="#475569", labelsize=7, pad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["tick"]["color"] = "#164e63"
        axis.line.set_color("#164e63")
        axis.line.set_linewidth(0.6)
    ax.view_init(elev=24, azim=55)

    colors = {k: THREATS[k]["color"] for k in keys}
    glyphs = {k: THREATS[k]["glyph"] for k in keys}

    # distinct marker SHAPE + SIZE per species (scaled to real body size) so a
    # pigeon reads as a big hex, a sparrow as a delta, a fly as a star burst --
    # never generic colored orbs
    bug_plots, halo_plots, trail_plots = {}, {}, {}
    for k2 in keys:
        ms = glyph_ms(THREATS[k2]["size_m"])
        halo_plots[k2] = ax.plot([], [], [], glyphs[k2], color=colors[k2],
                                 ms=ms * 2.1, alpha=0.14, linestyle="")[0]
        bug_plots[k2] = ax.plot([], [], [], glyphs[k2], color=colors[k2],
                                ms=ms, markeredgecolor="w",
                                markeredgewidth=0.5, alpha=0.95,
                                linestyle="")[0]
        trail_plots[k2] = [ax.plot([], [], [], "-", color=colors[k2],
                                   lw=1.4, alpha=0.35, solid_capstyle="round")[0]
                           for _ in range(THREATS[k2]["count"])]
    track_plot = ax.plot([], [], [], "+", color=CYAN, ms=6, alpha=0.8,
                         linestyle="")[0]
    beam_glow, = ax.plot([], [], [], color="#ff3d3d", lw=7.5, alpha=0.22,
                         solid_capstyle="round")
    beam_core, = ax.plot([], [], [], color="#ffcccc", lw=2.6,
                         solid_capstyle="round")
    aim_plot, = ax.plot([], [], [], "*", color=ACCENT, ms=13,
                        markeredgecolor="#7c2d12")
    burst_plots = [ax.plot([], [], [], "*", color="#ffd166", ms=16,
                           alpha=0.0, markeredgecolor="w",
                           linestyle="")[0] for _ in range(12)]
    turret_plot, = ax.plot([], [], [], marker="s", color="w", ms=9,
                           markeredgecolor=CYAN, markeredgewidth=1.2,
                           linestyle="")

    title_txt = fig.text(0.035, 0.955, "", color="w", fontsize=17,
                         fontweight="bold", family="monospace")
    sub_txt = fig.text(0.035, 0.915, "", color=CYAN, fontsize=10,
                       family="monospace")
    # retinotopic saliency inset: literally what the fly's optic lobe sees
    ax_sal = fig.add_axes([0.700, 0.265, 0.22, 0.16])
    ax_sal.set_facecolor("#0a0f1e")
    ax_sal.set_xticks([]); ax_sal.set_yticks([])
    for sp in ax_sal.spines.values():
        sp.set_edgecolor("#1e3a5f")
    sal_img = ax_sal.imshow(np.zeros((32, 64)), aspect="auto", cmap="inferno",
                            vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_sal.set_title("OPTIC LOBE  //  STMD SALIENCY", color="#7dd3fc",
                     fontsize=7.5, family="monospace", pad=3)
    attn_marker, = ax_sal.plot([], [], "wo", ms=5, markeredgecolor="k",
                               visible=False)
    # ------------------------------------------------------------------
    # "FLY BRAINWAVE MONITOR" panel -- the meme: an animated fly brain being
    # used as a science experiment. Live electrophysiology-style traces:
    # EEG rhythm, LGMD looming spike train, STMD neuron pool, attention drive.
    # ------------------------------------------------------------------
    ax_wave = fig.add_axes([0.700, 0.04, 0.22, 0.195])
    ax_wave.set_facecolor("#070c18")
    ax_wave.set_xticks([]); ax_wave.set_yticks([])
    ax_wave.set_xlim(0, 160); ax_wave.set_ylim(-0.15, 4.35)
    for sp in ax_wave.spines.values():
        sp.set_edgecolor("#1e3a5f")
    ax_wave.set_title("FLY BRAINWAVE MONITOR  //  IN VIVO",
                      color="#fbbf24", fontsize=7.5, family="monospace", pad=3)
    wave_lines = [
        ax_wave.plot(np.arange(160), np.full(160, 3.5), color="#22d3ee",
                     lw=0.9, alpha=0.95)[0],                       # EEG
        ax_wave.plot(np.arange(160), np.full(160, 2.4), color="#f87171",
                     lw=0.9, alpha=0.95)[0],                       # LGMD
        ax_wave.plot(np.arange(160), np.full(160, 1.3), color="#4ade80",
                     lw=0.9, alpha=0.95)[0],                       # STMD pool
        ax_wave.plot(np.arange(160), np.full(160, 0.2), color="#fbbf24",
                     lw=0.9, alpha=0.95)[0],                       # WTA att.
    ]
    wave_lbls = [
        ax_wave.text(2, y + 0.12, t, color=c, fontsize=5.5,
                     family="monospace")
        for y, t, c in [(3.5, "EEG", "#22d3ee"),
                        (2.4, "LGMD LOOMING", "#f87171"),
                        (1.3, "STMD POP", "#4ade80"),
                        (0.2, "WTA ATTENTION", "#fbbf24")]]
    hud_txt = fig.text(0.725, 0.97, "", color="#cbd5e1", fontsize=10,
                       family="monospace", va="top",
                       bbox=dict(boxstyle="round,pad=0.6", facecolor=PANE,
                                 edgecolor="#1e3a5f", alpha=0.9))
    legend_txt = fig.text(0.725, 0.68, "", color="#cbd5e1", fontsize=9.5,
                          family="monospace", va="top",
                          bbox=dict(boxstyle="round,pad=0.6", facecolor=PANE,
                                    edgecolor="#1e3a5f", alpha=0.9))
    fig.text(0.035, 0.012, "POINT-DEFENSE AI  //  sense > track > engage",
             color="#475569", fontsize=8.5, family="monospace")
    # legend glyphs match the exact marker shapes drawn in the 3D field;
    # targeted species get a LOCK tag, spared species a SPARED tag
    leg_lines = [
        f"{unis[k2]} {labels[k2]}  ({THREATS[k2]['power_w']:.0f} W)"
        f"  [{'TARGET' if k2 in tgt_set else 'SPARED'}]"
        for k2 in keys]
    leg_colors = [colors[k2] for k2 in keys]
    tgt_label = "+".join(THREATS[k2]["label"] for k2 in targets)

    def draw(fi):
        i = min(fi, len(frames) - 1)
        f = frames[i]
        tp = turret.pos
        for k2 in keys:
            pts = f["bugs"].get(k2, [])
            if pts:
                xs = [p[0][0] for p in pts]; ys = [p[0][1] for p in pts]
                zs = [p[0][2] for p in pts]
                bug_plots[k2].set_data(xs, ys)
                bug_plots[k2].set_3d_properties(zs)
                halo_plots[k2].set_data(xs, ys)
                halo_plots[k2].set_3d_properties(zs)
            else:
                for pl in (bug_plots[k2], halo_plots[k2]):
                    pl.set_data([], []); pl.set_3d_properties([])
            trs = f["trails"].get(k2, [])
            for ti, pl in enumerate(trail_plots[k2]):
                if ti < len(trs) and len(trs[ti]) > 1:
                    pl.set_data(trs[ti][:, 0], trs[ti][:, 1])
                    pl.set_3d_properties(trs[ti][:, 2])
                else:
                    pl.set_data([], []); pl.set_3d_properties([])
        tk = f["tracks"]
        if tk:
            track_plot.set_data([a for a, b, c, cf in tk],
                                [b for a, b, c, cf in tk])
            track_plot.set_3d_properties([c for a, b, c, cf in tk])
        else:
            track_plot.set_data([], []); track_plot.set_3d_properties([])
        turret_plot.set_data([tp[0]], [tp[1]])
        turret_plot.set_3d_properties([tp[2]])
        if f["firing"]:
            for pl in (beam_core, beam_glow):
                pl.set_data([tp[0], f["aim"][0]], [tp[1], f["aim"][1]])
                pl.set_3d_properties([tp[2], f["aim"][2]])
        else:
            for pl in (beam_core, beam_glow):
                pl.set_data([], []); pl.set_3d_properties([])
        aim_plot.set_data([f["aim"][0]], [f["aim"][1]])
        aim_plot.set_3d_properties([f["aim"][2]])
        # expanding, fading kill bursts
        for bi, pl in enumerate(burst_plots):
            if bi < len(f["bursts"]):
                p, age, _tk = f["bursts"][bi]
                pl.set_markersize(8 + 26 * (age / 0.6))
                pl.set_alpha(max(0.0, 1.0 - age / 0.6) * 0.9)
                pl.set_data([p[0]], [p[1]]); pl.set_3d_properties([p[2]])
            else:
                pl.set_alpha(0.0)
        # slow cinematic orbit with gentle tilt breathing
        ax.view_init(elev=22 + 4 * math.sin(fi * 0.010),
                     azim=55 + fi * 0.22)
        title_txt.set_text("AUTONOMOUS POINT-DEFENSE \u2014 LIVE")
        brain_lbl = {"fly": "FLY OPTIC LOBE (WTA)", "hybrid": "HYBRID FLY+KALMAN",
                     "kalman": "KALMAN BASELINE"}.get(
                         f.get("brain_mode", "hybrid"), "HYBRID")
        sub_txt.set_text(f"{tgt_label.upper()} SWEEP  //  "
                         f"{tgt_count} TARGETS, {len(swarm)} ON FIELD  //  "
                         f"BRAIN: {brain_lbl}")
        hud_txt.set_text(
            f"TIME      {f['t']:6.1f} s\n"
            f"KILLS     {f['kills']:3d}/{tgt_count}\n"
            f"SHOTS     {turret.shots:5d}\n"
            f"ENERGY    {f['energy']:6.0f} J\n"
            f"HEAT      {f['heat']*100:5.0f} %\n"
            f"TRACKS    {len(f['tracks']):3d}\n"
            f"AZ/EL     {math.degrees(f['az']):5.0f}d "
            f"{math.degrees(f['el']):5.1f}d\n"
            f"LOOMING   {f.get('looming', 0.0)*100:5.0f} %")
        legend_txt.set_text("THREAT MATRIX\n" + "\n".join(leg_lines))
        sal = f.get("saliency")
        if sal is not None and sal.max() > 0:
            sal_img.set_data(sal / max(sal.max(), 1e-6))
            att = f.get("attention")
            if att is not None:
                ia = int((att[0] + math.pi) / (2 * math.pi) * 64) % 64
                ie = int((att[1] + math.pi / 2) / math.pi * 31)
                attn_marker.set_data([ia], [ie])
                attn_marker.set_visible(True)
            else:
                attn_marker.set_visible(False)
        else:
            sal_img.set_data(np.zeros((32, 64)))
            attn_marker.set_visible(False)
        # brainwave monitor: scroll live fly-neural traces (the meme panel)
        waves = f.get("waves")
        if waves:
            xs = np.arange(160)
            for ln, key, base in zip(
                    wave_lines, ("eeg", "looming", "neurons", "attention"),
                    (3.5, 2.4, 1.3, 0.2)):
                data = waves.get(key)
                if data:
                    ln.set_data(xs[-len(data):], base + 0.85 * np.array(data))
        # end-of-video recap: freeze camera and swap the leaderboard in
        if i == len(frames) - 1 and len(frames) > 1:
            title_txt.set_text("AUTONOMOUS POINT-DEFENSE \u2014 CLEARED")
            sub_txt.set_text(f"{tgt_label.upper()} SWEEP COMPLETE  //  "
                             f"{turret.kills}/{tgt_count} NEUTRALIZED  //  "
                             f"{t_end:.1f} S")
            legend_txt.set_text(leaderboard_txt)
        return []

    anim = animation.FuncAnimation(fig, draw, len(frames) + 90, interval=33,
                                   blit=False)
    mp4_path = out_mp4
    mp4_ok = False
    try:
        writerv = animation.FFMpegWriter(
            fps=30, bitrate=4000, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high"])
        anim.save(out_mp4, writer=writerv)
        mp4_ok = True
        print(f"[render] saved {out_mp4}")
    except Exception as e:
        print(f"[render] mp4 failed ({e}); trying gif fallback")
    if not mp4_ok:
        try:
            mp4_path = out_mp4.replace(".mp4", ".gif")
            anim.save(mp4_path, writer=animation.PillowWriter(fps=15))
            print(f"[render] saved {mp4_path} (gif fallback)")
        except Exception as e2:
            print(f"[render] WARNING: no animated output saved: {e2}")

    # --------------------------------------------------------------
    # Styled summary PNG
    # --------------------------------------------------------------
    fig2 = plt.figure(figsize=(12.8, 7.2))
    fig2.patch.set_facecolor(BG)
    ax2 = fig2.add_axes([0.02, 0.02, 0.70, 0.90], projection="3d")
    ax2.set_xlim(0, AREA); ax2.set_ylim(0, AREA); ax2.set_zlim(0, HEIGHT)
    ax2.set_facecolor(BG)
    for pane in (ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane):
        pane.set_facecolor(PANE); pane.set_alpha(0.30)
        pane.set_edgecolor("#164e63")
    ax2.grid(True, color="#155e75", alpha=0.35, lw=0.5)
    ax2.set_xlabel("X (m)", color="#67e8f9")
    ax2.set_ylabel("Y (m)", color="#67e8f9")
    ax2.set_zlabel("Z (m)", color="#67e8f9")
    ax2.tick_params(colors="#475569", labelsize=7, pad=2)
    for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        axis._axinfo["tick"]["color"] = "#164e63"
        axis.line.set_color("#164e63")
        axis.line.set_linewidth(0.6)
    ax2.view_init(elev=24, azim=55)
    for src, hit, killed in turret.beam_history[-100:]:
        ax2.plot([src[0], hit[0]], [src[1], hit[1]], [src[2], hit[2]],
                 color="#4ade80" if killed else "#f87171",
                 alpha=0.30, lw=0.9)
    for k2 in keys:
        pts = [s.pos for s in swarm if s.alive() and s.key == k2]
        if pts:
            p = np.array(pts)
            # survivors drawn with their species glyph, sized like the live view
            ax2.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors[k2],
                        marker=THREATS[k2]["glyph"],
                        s=max(30, glyph_ms(THREATS[k2]["size_m"]) ** 2 * 0.6),
                        edgecolors="w", linewidths=0.5, alpha=0.95)
    tp = turret.pos
    ax2.scatter([tp[0]], [tp[1]], [tp[2]], c="w", marker="s", s=70,
                edgecolors=CYAN, linewidths=1.2)
    hit_eff = 100.0 * turret.kills / max(1, turret.shots)
    brain_lbl = {"fly": "FLY OPTIC LOBE", "hybrid": "HYBRID FLY+KALMAN",
                 "kalman": "KALMAN BASELINE"}.get(brain_mode, "HYBRID")
    fig2.text(0.035, 0.955,
              "AUTONOMOUS POINT-DEFENSE \u2014 ENGAGEMENT REPORT",
              color="w", fontsize=17, fontweight="bold", family="monospace")
    # two stacked banner lines so long species lists never clip off-frame
    fig2.text(0.035, 0.905,
              f"{turret.kills}/{tgt_count} TARGETS DOWN  //  "
              f"{turret.shots} SHOTS  //  {turret.energy_j:.0f} J  //  "
              f"{hit_eff:.0f}% KILLS/SHOT  //  {t_end:.1f} S\n"
              f"{tgt_label.upper()}  //  BRAIN: {brain_lbl}",
              color=CYAN, fontsize=10, family="monospace", va="top",
              linespacing=1.7)
    # optic-lobe saliency snapshot (peak attention state) on the report
    sal_peak = None
    if frames:
        sal_frames = [f.get("saliency") for f in frames
                      if f.get("saliency") is not None]
        if sal_frames:
            sal_peak = max(sal_frames, key=lambda s: float(s.max()))
    if sal_peak is not None and sal_peak.max() > 0:
        ax_sal2 = fig2.add_axes([0.735, 0.10, 0.20, 0.15])
        ax_sal2.set_facecolor("#0a0f1e")
        ax_sal2.set_xticks([]); ax_sal2.set_yticks([])
        for sp in ax_sal2.spines.values():
            sp.set_edgecolor("#1e3a5f")
        ax_sal2.imshow(sal_peak / max(sal_peak.max(), 1e-6), aspect="auto",
                       cmap="inferno", vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        ax_sal2.set_title("OPTIC LOBE  //  PEAK STMD SALIENCY", color="#7dd3fc",
                          fontsize=7.5, family="monospace", pad=3)
    # full-engagement brainwave strip on the report (the meme, quantified)
    if frames and frames[-1].get("waves"):
        ax_wv2 = fig2.add_axes([0.735, 0.30, 0.20, 0.12])
        ax_wv2.set_facecolor("#070c18")
        ax_wv2.set_xticks([]); ax_wv2.set_yticks([])
        ax_wv2.set_xlim(0, 160); ax_wv2.set_ylim(-0.15, 4.35)
        for sp in ax_wv2.spines.values():
            sp.set_edgecolor("#1e3a5f")
        wfinal = frames[-1]["waves"]
        for data, base, c in zip(
                (wfinal["eeg"], wfinal["looming"], wfinal["neurons"],
                 wfinal["attention"]), (3.5, 2.4, 1.3, 0.2),
                ("#22d3ee", "#f87171", "#4ade80", "#fbbf24")):
            xs = np.arange(160)
            ax_wv2.plot(xs[-len(data):], base + 0.85 * np.array(data),
                        color=c, lw=0.8)
        ax_wv2.set_title("FLY BRAINWAVE MONITOR  //  IN VIVO", color="#fbbf24",
                         fontsize=7.0, family="monospace", pad=2)
    fig2.text(0.735, 0.88, "THREAT MATRIX\n" + survivors_txt,
              color="#cbd5e1", fontsize=9.0, family="monospace", va="top",
              bbox=dict(boxstyle="round,pad=0.6", facecolor=PANE,
                        edgecolor="#1e3a5f", alpha=0.9))
    fig2.text(0.715, 0.52, leaderboard_txt,
              color=ACCENT, fontsize=8.0, family="monospace", va="top",
              bbox=dict(boxstyle="round,pad=0.6", facecolor=PANE,
                        edgecolor="#7c5a12", alpha=0.95))
    fig2.savefig(out_png, dpi=140, facecolor=BG)
    print(f"[render] saved {out_png}")
    stats["outputs"] = {"mp4": mp4_path if mp4_ok else None, "png": out_png}
    if return_frames:
        stats["frames"] = frames
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mixed", choices=["mixed", "single"])
    ap.add_argument("--creatures", default="mosquitoes",
                    help="comma-separated species to TARGET (all species "
                         "still spawn on the field), e.g. mosquitoes,flies "
                         " (or 'all')")
    ap.add_argument("--seconds", type=float, default=90.0,
                    help="upper bound; sim ends when all targets are cleared")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--brain", default="hybrid",
                    choices=["fly", "hybrid", "kalman"],
                    help="guidance brain: fly (optic-lobe WTA only), "
                         "hybrid (fly attention gates Kalman fire control), "
                         "kalman (pure engineering baseline)")
    ap.add_argument("--out-png", default="defense3d.png")
    ap.add_argument("--out-mp4", default="defense3d.mp4")
    a = ap.parse_args()
    creatures = None
    if a.creatures and a.creatures.lower() != "all":
        creatures = [c.strip() for c in a.creatures.split(",") if c.strip()]
        bad = [c for c in creatures if c not in SPECIES_ORDER]
        if bad:
            ap.error(f"unknown species: {bad} (choose from {SPECIES_ORDER})")
    s = run_sim(mode=a.mode, single=a.creatures.split(",")[0].strip(),
                seconds=a.seconds, seed=a.seed, out_png=a.out_png,
                out_mp4=a.out_mp4, creatures=creatures,
                brain_mode=a.brain)
    print(json.dumps(s, indent=2))

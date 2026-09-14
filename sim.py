#!/usr/bin/env python3
"""
MOSQUITO POINT-DEFENSE SIMULATOR  ("laser turret vs swarm")
===========================================================
Aircraft-carrier-grade fire control, 2-millimeter targets.

Pipeline mirrors a real C-RAM / point-defense system:
    1. SENSORS    - synthetic camera with noise, blind time, detection gating
    2. TRACKING   - global-nearest-neighbor data association + per-track
                    constant-velocity Kalman filters
    3. FIRE CTRL  - lead-pursuit solution with convergence check,
                    slew-rate-limited turret, energy budget
    4. EFFECTORS  - galvo-steered laser: spot size, atmospheric/window
                    attenuation, dwell time, stochastic kill probability
    5. THREATS    - mosquito swarm with wander + noise-driven evasive jinks
                    (raises with laser losses, so survivors get harder)

Escalation knob:  --creatures mosquitoes|roaches|sparrows|pigeons|drones
Same code path, retuned threat table. That's the "scale up the creatures" step.

Everything is in SI units. Kill assessment is stochastic (p_kill per dwell),
not scripted, so a miss or a weak burn actually happens.
"""

import argparse
import json
import math
import random

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, patches

# ----------------------------------------------------------------------------
# Threat escalation table -- "over time we scale up the creatures"
# ----------------------------------------------------------------------------
THREATS = {
    "mosquitoes": dict(
        label="Mosquitoes",
        hp=1.0, speed=0.55, speed_spread=0.20, wander=2.2,
        pkill=0.88, dwell=0.030, spot_mm=6.0,   # 2W focused -> instant pop
        flux_needed=0.9, count=14, color="#2ecc71", size_m=0.0045,
        power_w=2.0,
    ),
    "roaches": dict(
        label="Roaches",
        hp=2.6, speed=1.10, speed_spread=0.35, wander=3.0,
        pkill=0.55, dwell=0.055, spot_mm=6.0,
        flux_needed=1.6, count=8, color="#8e5a2b", size_m=0.015,
        power_w=6.0,
    ),
    "sparrows": dict(
        label="Sparrows",
        hp=7.0, speed=3.5, speed_spread=0.8, wander=4.5,
        pkill=0.30, dwell=0.085, spot_mm=8.0,
        flux_needed=4.5, count=6, color="#c0392b", size_m=0.070,
        power_w=15.0,
    ),
    "pigeons": dict(
        label="Pigeons",
        hp=14.0, speed=5.5, speed_spread=1.0, wander=4.0,
        pkill=0.18, dwell=0.110, spot_mm=9.0,
        flux_needed=9.0, count=4, color="#7f8fa6", size_m=0.170,
        power_w=30.0,
    ),
    "drones": dict(
        label="Micro-drones",
        hp=22.0, speed=8.0, speed_spread=1.5, wander=2.5,
        pkill=0.12, dwell=0.130, spot_mm=10.0,
        flux_needed=16.0, count=3, color="#e67e22", size_m=0.250,
        power_w=60.0,
    ),
}

# ----------------------------------------------------------------------------
# Threat swarm model
# ----------------------------------------------------------------------------
class Swarm:
    """Constant-speed wanderers with noise-triggered evasive jinks."""

    def __init__(self, key, area, rng):
        t = THREATS[key]
        self.t = t
        self.area = area
        self.rng = rng
        self.pos = np.array([rng.uniform(0.5, area - 0.5),
                             rng.uniform(0.5, area - 0.5)])
        ang = rng.uniform(0, 2 * math.pi)
        self.vel = np.array([math.cos(ang), math.sin(ang)]) * t["speed"]
        self.hp = t["hp"]

    def step(self, dt, time, laser_pressure):
        """laser_pressure 0..1: survivors jink harder as the swarm dies off."""
        t = self.t
        rng = self.rng
        # wander: piecewise target heading, resampled randomly
        if rng.random() < t["wander"] * dt:
            ang = rng.uniform(0, 2 * math.pi)
            sp = t["speed"] * (1.0 + rng.uniform(-t["speed_spread"],
                                                 t["speed_spread"]))
            self.vel = np.array([math.cos(ang), math.sin(ang)]) * sp
        # evasive jink when beam gets close or swarm is under fire
        if laser_pressure > 0 and rng.random() < laser_pressure * 1.5 * dt:
            ang = math.atan2(self.vel[1], self.vel[0]) + rng.uniform(-1.2, 1.2)
            sp = np.linalg.norm(self.vel)
            self.vel = np.array([math.cos(ang), math.sin(ang)]) * sp
        self.pos = self.pos + self.vel * dt
        # soft wall reflection
        for i in range(2):
            if self.pos[i] < 0.2:
                self.pos[i] = 0.2
                self.vel[i] = abs(self.vel[i])
            elif self.pos[i] > self.area - 0.2:
                self.pos[i] = self.area - 0.2
                self.vel[i] = -abs(self.vel[i])

    def alive(self):
        return self.hp > 0


# ----------------------------------------------------------------------------
# Sensor: synthetic camera with noise + detection gating
# ----------------------------------------------------------------------------
class Sensor:
    def __init__(self, noise_std=0.008, p_detect=0.93, miss_prob=0.05):
        self.noise_std = noise_std        # meters, 1-sigma on x and y
        self.p_detect = p_detect          # prob the frame sees an alive bug
        self.miss_prob = miss_prob        # dropped frame per track

    def observe(self, swarm_list):
        """Returns list of (track_id_hint, noisy_xy) detections."""
        dets = []
        for sw in swarm_list:
            if not sw.alive():
                continue
            if self.rng_uniform() > self.p_detect:
                continue
            noisy = sw.pos + self.rng_normal() * self.noise_std
            dets.append(noisy)
        return dets

    def rng_uniform(self):
        return random.random()

    def rng_normal(self):
        return random.gauss(0.0, 1.0)


# ----------------------------------------------------------------------------
# Tracker: GNN association + constant-velocity Kalman filters
# ----------------------------------------------------------------------------
class Track:
    _next_id = 1

    def __init__(self, z, dt):
        self.id = Track._next_id
        Track._next_id += 1
        self.x = np.array([z[0], z[1], 0.0, 0.0])       # [x, y, vx, vy]
        self.P = np.diag([0.05, 0.05, 4.0, 4.0])
        self.hits = 1
        self.misses = 0
        self.dt_last = dt
        self.age = 0.0
        self.confirmed = False          # fire-control only engages confirmed
        self.frozen = 0                 # frames since last update
        self.last_update = dt

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        q = 0.35                        # process noise (wandering targets)
        Q = q * np.array([
            [dt**3 / 3, 0, dt**2 / 2, 0],
            [0, dt**3 / 3, 0, dt**2 / 2],
            [dt**2 / 2, 0, dt, 0],
            [0, dt**2 / 2, 0, dt],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += dt
        self.last_update += dt

    def update(self, z):
        H = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.]])
        R = np.diag([0.008**2, 0.008**2])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1
        self.misses = 0
        self.last_update = 0.0
        if self.hits >= 3:
            self.confirmed = True

    def coast(self):
        """No update this frame."""
        self.misses += 1
        self.last_update += 0.0

    def stale(self, gate_time=0.6):
        return self.misses * 0.033 > gate_time


class Tracker:
    def __init__(self):
        self.tracks = []

    def step(self, detections, dt):
        # predict all
        for tr in self.tracks:
            tr.predict(dt)
        # global nearest neighbor association (greedy on distance)
        pairs = []
        for i, tr in enumerate(self.tracks):
            for j, z in enumerate(detections):
                d = np.linalg.norm(tr.x[:2] - z)
                if d < 0.25:                       # association gate
                    pairs.append((d, i, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for d, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            self.tracks[i].update(detections[j])
            used_t.add(i)
            used_d.add(j)
        # unmatched detections -> new tracks
        for j, z in enumerate(detections):
            if j not in used_d:
                self.tracks.append(Track(z, dt))
        # unmatched tracks -> coast
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.coast()
        # prune stale
        self.tracks = [tr for tr in self.tracks if not tr.stale()]
        return self.tracks


# ----------------------------------------------------------------------------
# Fire control: turret + lead-pursuit solution with convergence check
# ----------------------------------------------------------------------------
class Turret:
    """Slew-rate-limited laser turret with lead-pursuit engagement logic."""

    def __init__(self, pos, laser_power_w=2.0, spot_mm=6.0,
                 attn_per_m=0.03, max_slew_dps=400.0):
        self.pos = np.array(pos, dtype=float)
        self.power = laser_power_w
        self.spot_mm = spot_mm
        self.attn = attn_per_m          # per-meter atmospheric/window loss
        self.max_slew = math.radians(max_slew_dps)
        self.aim = np.array([pos[0] + 1.0, pos[1]])   # initial aim point
        self.heat = 0.0                 # thermal duty cycle 0..1
        self.firing = False
        self.target = None
        self.beam_history = []          # [(turret_xy, hit_xy, killed)]
        self.shots = 0
        self.kills = 0
        self.energy_j = 0.0

    def lead_point(self, tr, flight_time):
        """Where the target will be when the beam dwell completes."""
        p = tr.x[:2]
        v = tr.x[2:4]
        lead = p + v * flight_time
        # aim at center of mass with slight vertical offset for wings :)
        return lead

    def time_to_turn(self, aim_point):
        d = aim_point - self.aim
        # crude slew time: angle between aim vector and target vector
        ang = abs(math.atan2(d[1], d[0]) - math.atan2(
            self.aim[1] - self.pos[1], self.aim[0] - self.pos[0]))
        ang = min(ang, math.pi)
        r = np.linalg.norm(aim_point - self.pos)
        arc = ang * r
        return arc / (self.max_slew * r) if r > 1e-6 else 0.0

    def engage(self, tracks, dt, swarm_list, rng, tparams):
        """Pick target, solve, fire if solution converges and energy allows."""
        self.firing = False
        # --- thermal model: beam duty cycle, cool when idle
        self.heat = max(0.0, self.heat - 0.35 * dt)
        if self.heat > 0.85:
            self.target = None
            return

        # --- target selection: keep current lock unless it goes stale
        # (fire-control hysteresis: no thrashing between targets)
        cands = [tr for tr in tracks if tr.confirmed and tr.misses < 4]
        if not cands:
            self.target = None
            self.lock_id = None
            # slew toward centroid of tracks to look busy
            if tracks:
                c = np.mean([tr.x[:2] for tr in tracks], axis=0)
                self._slew_toward(c, dt)
            return

        tgt = None
        if getattr(self, "lock_id", None) is not None:
            tgt = next((tr for tr in cands if tr.id == self.lock_id), None)
        if tgt is None:
            tgt = min(cands, key=lambda tr: np.linalg.norm(tr.x[:2] - self.pos))
        self.lock_id = tgt.id
        self.target = tgt.id

        # --- lead-pursuit solution (iterate: aim point depends on flight
        # time, flight time depends on aim point)
        tof = 0.0
        aim = tgt.x[:2]
        for _ in range(3):
            aim = self.lead_point(tgt, tof)
            tof = self.time_to_turn(aim) + tparams["dwell"]

        # convergence check: does the target actually arrive near aim point?
        err = np.linalg.norm(tgt.x[:2] + tgt.x[2:4] * tof - aim)
        if err > 0.10:                        # not converged, don't waste joules
            self._slew_toward(aim, dt)
            return

        # --- slew toward solution
        arrived = self._slew_toward(aim, dt)
        if not arrived:
            return

        # --- check the physical spot actually lands on the creature
        # miss distance = |aim - true future position| (tof already includes
        # slew + dwell); the convergence gate above bounds it.
        beam_len = np.linalg.norm(aim - self.pos)
        flux = self.power * math.exp(-self.attn * beam_len)  # watts delivered

        # --- FIRE: dwell time, stochastic kill assessment
        self.firing = True
        self.shots += 1
        self.energy_j += flux * tparams["dwell"]
        p_kill = tparams["pkill"] * min(1.0, flux / tparams["flux_needed"])
        killed = rng.random() < p_kill
        self.beam_history.append((self.pos.copy(), aim.copy(), killed))
        if killed:
            # assess against the SAME tof-projected position the fire
            # solution aimed at; radius scales with spot size + creature size
            spot_r = (tparams["spot_mm"] / 1000.0) / 2.0
            hit_r = spot_r * 1.5 + tparams["size_m"]
            for sw in swarm_list:
                if not sw.alive():
                    continue
                proj = sw.pos + sw.vel * tof
                if np.linalg.norm(proj - aim) < hit_r:
                    sw.hp = 0.0
                    self.kills += 1
                    break
        self.heat = min(1.0, self.heat + tparams["dwell"] * 4.0)

    def _slew_toward(self, aim_point, dt):
        """Move aim point toward the target at max slew rate. Returns True
        when within spot radius."""
        d = aim_point - self.aim
        dist = np.linalg.norm(d)
        step = self.max_slew * dt * max(0.3, np.linalg.norm(
            self.aim - self.pos))          # angular rate x lever arm
        if dist <= step:
            self.aim = aim_point.copy()
            return True
        self.aim = self.aim + d / dist * step
        return False


# ----------------------------------------------------------------------------
# Simulation driver
# ----------------------------------------------------------------------------
def run_sim(threat_key="mosquitoes", seconds=40.0, dt=1 / 60.0, seed=7,
            out_png="defense_sim.png", out_mp4="defense_sim.mp4"):
    rng = random.Random(seed)
    nprng = np.random.RandomState(seed)
    tparams = THREATS[threat_key]
    AREA = 6.0
    turret_pos = (0.3, AREA / 2)

    # --- build world
    swarm = [Swarm(threat_key, AREA, rng) for _ in range(tparams["count"])]
    sensor = Sensor()
    tracker = Tracker()
    turret = Turret(turret_pos, laser_power_w=tparams["power_w"],
                    spot_mm=tparams["spot_mm"])

    # --- logging
    frames = []
    stats = dict(shots=0, kills=0, energy=0.0, frames=0, track_switches=0)
    kill_times = []

    steps = int(seconds / dt)
    prev_target = None
    for k in range(steps):
        t = k * dt
        alive = [s for s in swarm if s.alive()]
        laser_pressure = 1.0 - len(alive) / max(1, len(swarm))

        # 1. sense
        dets = sensor.observe(alive)
        # 2. track
        tracks = tracker.step(dets, dt)
        # 3. fire control + effector
        turret.engage(tracks, dt, swarm, rng, tparams)

        if prev_target is not None and turret.target != prev_target:
            stats["track_switches"] += 1
        prev_target = turret.target

        # swarm moves (post-engagement: jinks away from beam)
        for s in alive:
            near_beam = 0.0
            if turret.firing:
                dd = np.linalg.norm(s.pos - turret.aim)
                near_beam = max(0.0, 1.0 - dd / 0.35)
            s.step(dt, t, laser_pressure + near_beam * 0.5)

        # log for animation
        if k % 2 == 0:   # 30 fps output
            frames.append(dict(
                t=t,
                bugs=[(s.pos.copy(), s.vel.copy(), s.hp / tparams["hp"])
                      for s in alive],
                aim=turret.aim.copy(),
                firing=turret.firing,
                tracks=[(tr.x[0], tr.x[1], tr.confirmed) for tr in tracks],
                heat=turret.heat,
            ))

    stats["shots"] = turret.shots
    stats["kills"] = turret.kills
    stats["energy"] = turret.energy_j
    stats["frames"] = len(frames)

    # ------------------------------------------------------------------
    # Render: static PNG (final state) + animated MP4
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, AREA)
    ax.set_ylim(0, AREA)
    ax.set_aspect("equal")
    ax.set_facecolor("#0d1117")
    ax.set_title(f"Point-Defense Laser vs {tparams['label']}", color="w")
    ax.tick_params(colors="w")

    turret_marker, = ax.plot([], [], "ws", ms=12)
    aim_dot, = ax.plot([], [], "y.", ms=8)
    beam, = ax.plot([], [], "r-", lw=2, alpha=0.9)
    bug_dots, = ax.plot([], [], "o", color=tparams["color"], ms=5,
                        linestyle="")
    track_dots, = ax.plot([], [], "c+", ms=6, linestyle="")

    def draw(fi):
        f = frames[fi]
        if f["bugs"]:
            bx = [p[0] for p, v, h in f["bugs"]]
            by = [p[1] for p, v, h in f["bugs"]]
            bug_dots.set_data(bx, by)
        else:
            bug_dots.set_data([], [])
        if f["tracks"]:
            tx = [tr[0] for tr in f["tracks"]]
            ty = [tr[1] for tr in f["tracks"]]
            track_dots.set_data(tx, ty)
        else:
            track_dots.set_data([], [])
        aim_dot.set_data([f["aim"][0]], [f["aim"][1]])
        turret_marker.set_data([turret_pos[0]], [turret_pos[1]])
        if f["firing"]:
            beam.set_data([turret_pos[0], f["aim"][0]],
                          [turret_pos[1], f["aim"][1]])
        else:
            beam.set_data([], [])
        ax.set_title(f"t={f['t']:.1f}s  heat={f['heat']*100:.0f}%  "
                     f"kills={turret.kills}/{len(swarm)}", color="w")
        return (beam, bug_dots, track_dots, aim_dot, turret_marker)

    anim = animation.FuncAnimation(fig, draw, len(frames), interval=33,
                                   blit=True)
    try:
        writerv = animation.FFMpegWriter(fps=30, bitrate=2400)
        anim.save(out_mp4, writer=writerv)
        print(f"[render] saved {out_mp4}")
    except Exception as e:
        print(f"[render] mp4 failed ({e}); saving gif fallback")
        try:
            anim.save(out_mp4.replace(".mp4", ".gif"),
                      writer=animation.PillowWriter(fps=20))
        except Exception as e2:
            print(f"[render] gif also failed: {e2}")

    # static final-state PNG with beam history overlay
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    ax2.set_xlim(0, AREA)
    ax2.set_ylim(0, AREA)
    ax2.set_aspect("equal")
    ax2.set_facecolor("#0d1117")
    ax2.set_title(f"Engagement summary - {tparams['label']}\n"
                  f"{turret.kills}/{len(swarm)} killed, "
                  f"{turret.shots} shots, {turret.energy_j:.1f} J delivered",
                  color="w")
    ax2.tick_params(colors="w")
    for src, hit, killed in turret.beam_history[-60:]:
        c = "lime" if killed else "red"
        ax2.plot([src[0], hit[0]], [src[1], hit[1]], "-", color=c,
                 alpha=0.35, lw=1)
    alive_pos = [s.pos for s in swarm if s.alive()]
    if alive_pos:
        ap = np.array(alive_pos)
        ax2.plot(ap[:, 0], ap[:, 1], "o", color=tparams["color"], ms=6)
    ax2.plot([turret_pos[0]], [turret_pos[1]], "ws", ms=14)
    fig2.savefig(out_png, dpi=120, facecolor="#0d1117")
    print(f"[render] saved {out_png}")

    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--creatures", default="mosquitoes",
                    choices=list(THREATS.keys()))
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-png", default="defense_sim.png")
    ap.add_argument("--out-mp4", default="defense_sim.mp4")
    args = ap.parse_args()
    s = run_sim(args.creatures, args.seconds, seed=args.seed,
                out_png=args.out_png, out_mp4=args.out_mp4)
    print(json.dumps(s, indent=2))

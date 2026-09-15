#!/usr/bin/env python3
"""
END-TO-END TEST: everything in code, before any hardware is ordered
===================================================================
Chain under test (one process):

  sim3d Swarm3D (real flight dynamics, seeded)
    -> vision.render_frame   (virtual OV9281 + IR strobe, 6 mm lens)
    -> VisionPipeline        (blobs -> tracks -> mono-range -> 3D dets)
    -> Tracker3D + FlyBrain  (the real slow-loop targeting code)
    -> PanTiltTurret.engage  (the real fire-control logic)
    -> protocol.encode_fire  (the real wire format)
    -> firmware/fire_control.c via ctypes  (the real MCU fast loop)
    -> V/T frames back       (veto + status accounting)

Pass criteria:
  A. vision-fed engagement kills a majority of target mosquitoes in 30 s
  B. firmware core completes every fire frame: no watchdog trips,
     heat stays <= 10000 cP, no unexpected vetoes, stays armed
  C. abort -> disarm -> veto -> re-arm cycle through the real core

Run:  python3 tests/e2e_test.py     (builds the C core first)
"""
import ctypes
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim3d
import vision
import flybrain
from hardware import protocol

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------
# Build the C core as a shared library
# ----------------------------------------------------------------------------
def build_firmware():
    src = os.path.join(ROOT, "firmware", "fire_control.c")
    so = os.path.join(ROOT, "firmware", "libfire_control.so")
    subprocess.run(["cc", "-Wall", "-O2", "-shared", "-fPIC",
                    "-o", so, src], check=True)
    return so


# ----------------------------------------------------------------------------
# ctypes harness for the firmware core
# ----------------------------------------------------------------------------
LASER_CB = ctypes.CFUNCTYPE(None, ctypes.c_int32)
VOID_CB = ctypes.CFUNCTYPE(None)
GALVO_CB = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_int32)
MS_CB = ctypes.CFUNCTYPE(ctypes.c_uint32)
TX_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)


class FirmCore:
    """The actual C fire-control core, driven exactly as the MCU would be."""

    def __init__(self, beam_cw_limit=200):
        lib = ctypes.CDLL(os.path.join(ROOT, "firmware", "libfire_control.so"))
        self.lib = lib
        lib.fc_init.argtypes = [ctypes.c_void_p]
        lib.fc_on_line.argtypes = [ctypes.c_char_p, TX_CB, ctypes.c_void_p]
        lib.fc_tick_1khz.argtypes = []
        lib.fc_heat.argtypes = []
        lib.fc_heat.restype = ctypes.c_int32
        lib.fc_shots.argtypes = []
        lib.fc_shots.restype = ctypes.c_uint32
        lib.fc_armed.restype = ctypes.c_int
        lib.fc_firing.restype = ctypes.c_int

        self._on = LASER_CB(self._laser_on)
        self._off = VOID_CB(self._laser_off)
        self._gv = GALVO_CB(self._galvo)
        self._ms = MS_CB(self._get_ms)
        self._tx = TX_CB(self._on_tx)

        class Hal(ctypes.Structure):
            _fields_ = [("laser_on", LASER_CB), ("laser_off", VOID_CB),
                        ("galvo_to", GALVO_CB), ("ms", MS_CB),
                        ("beam_cw_limit", ctypes.c_int32)]

        self._hal = Hal(self._on, self._off, self._gv, self._ms,
                        beam_cw_limit)
        rc = lib.fc_init(ctypes.byref(self._hal))
        assert rc == 0
        self.tx_lines = []
        self.vetoes = []
        self.beam_ons = 0
        self._ms_val = 0

    # HAL callbacks
    def _laser_on(self, pw): self.beam_ons += 1
    def _laser_off(self): pass
    def _galvo(self, az, el): pass
    def _get_ms(self): return self._ms_val
    def _on_tx(self, buf, n, ud):
        line = buf[:n].decode()
        self.tx_lines.append(line)
        if line.startswith("V,"):
            self.vetoes.append(line)

    # driver API
    def send(self, line: str):
        self.lib.fc_note_rx()
        self.lib.fc_on_line(line.encode(), self._tx, None)

    def tick_ms(self, n):
        for _ in range(n):
            self._ms_val += 1
            self.lib.fc_tick_1khz()


# ----------------------------------------------------------------------------
# Shared rig: real tracker + brain + turret fed by the real vision pipeline
# ----------------------------------------------------------------------------
def make_rig(seed):
    rng = random.Random(seed)
    mosq = [sim3d.Swarm3D("mosquitoes", sim3d.AREA, sim3d.HEIGHT, rng)
            for _ in range(sim3d.THREATS["mosquitoes"]["count"])]
    turret = sim3d.PanTiltTurret(pos=(0.3, sim3d.AREA / 2, 1.5), power_w=2.0)
    turret.brain = flybrain.FlyBrain()
    turret.brain_mode = "hybrid"
    tracker = sim3d.Tracker3D()
    # REAL BOM GEOMETRY: camera co-mounted with the diode (2 cm offset),
    # level pitch so the 0.3-2.6 m swarm band fills the frame
    cam = vision.PixelCamera(pos=(0.3, sim3d.AREA / 2 - 0.02, 1.5),
                             pitch_deg=0)
    vp = vision.VisionPipeline(cam)
    vrng = np.random.default_rng(seed)
    return mosq, turret, tracker, cam, vp, vrng, rng


def run_frame(mosq, turret, tracker, cam, vp, vrng, dt):
    alive = [s for s in mosq if s.alive()]
    frame = vision.render_frame(alive, cam, rng=vrng)   # virtual IR camera
    dets = vp.step(frame, dt)                           # pixels -> 3D
    turret.brain.update(dets, turret.pos, dt)           # optic lobe
    tracks = tracker.step(dets, dt)                     # Kalman tracks
    return alive, dets, tracks


# ----------------------------------------------------------------------------
# A. vision-fed engagement against the real sim dynamics
# ----------------------------------------------------------------------------
def test_vision_engagement():
    mosq, turret, tracker, cam, vp, vrng, _ = make_rig(11)
    sim_rng = random.Random(11)
    dt = 1 / 60.0
    for k in range(int(30 / dt)):
        turret.clock = k * dt
        alive, dets, tracks = run_frame(mosq, turret, tracker, cam, vp, vrng, dt)
        turret.engage(tracks, dt, mosq, sim_rng, target_keys=["mosquitoes"])
        for s in alive:
            s.step(dt, 0.0)

    kills = sum(1 for s in mosq if not s.alive())
    n = sim3d.THREATS["mosquitoes"]["count"]
    print(f"[A] vision-fed 30s engagement: {kills}/{n} mosquitoes killed "
          f"through camera+vision+tracker chain")
    assert kills >= n // 2, f"vision-fed engagement too weak ({kills}/{n})"
    return kills


# ----------------------------------------------------------------------------
# B. every turret shot becomes a real wire frame into the real C core
# ----------------------------------------------------------------------------
def test_firmware_in_loop():
    mosq, turret, tracker, cam, vp, vrng, _ = make_rig(5)
    sim_rng = random.Random(5)
    core = FirmCore(beam_cw_limit=200)
    core.send(protocol.encode_arm(1))

    dt = 1 / 60.0
    seq = 2
    prev_shots = 0
    frames_sent = 0
    for k in range(int(20 / dt)):
        turret.clock = k * dt
        alive, dets, tracks = run_frame(mosq, turret, tracker, cam, vp, vrng, dt)
        turret.engage(tracks, dt, mosq, sim_rng, target_keys=["mosquitoes"])
        for s in alive:
            s.step(dt, 0.0)

        if turret.shots > prev_shots:
            # mirror the turret's actual shot as a wire frame
            aim = turret.aim_point
            rel = np.asarray(aim) - turret.pos
            r = float(np.linalg.norm(rel))
            az = math.atan2(rel[1], rel[0])
            el = math.asin(np.clip(rel[2] / max(r, 0.1), -1, 1))
            dwell_ms = min(50.0, sim3d.THREATS["mosquitoes"]["dwell"] * 1000)
            cmd = protocol.FireCommand(az=az, el=el, dwell_ms=dwell_ms,
                                       power_w=2.0, seq=seq,
                                       expires_s=time.monotonic() + 0.5)
            core.send(protocol.encode_fire(cmd))
            frames_sent += 1
            seq = (seq + 1) % 65536
            prev_shots = turret.shots

        # MCU fast loop runs one frame's worth of ticks + status keepalive
        core.tick_ms(17)
        core.send(protocol.encode_status_request())

    heat = core.lib.fc_heat()
    shots = core.lib.fc_shots()
    armed = core.lib.fc_armed()
    bad_vetoes = [v for v in core.vetoes
                  if "watchdog" in v or "disarmed" in v or "estop" in v]
    print(f"[B] firmware-in-loop: {frames_sent} fire frames -> {shots} shots "
          f"completed by C core, heat {heat}/10000, armed={armed}, "
          f"vetoes={core.vetoes}")
    assert frames_sent > 0, "turret never fired in 20 s"
    assert shots == frames_sent, "firmware did not complete every shot"
    assert heat <= 10000, "firmware heat exceeded 100%"
    assert armed == 1, "firmware disarmed unexpectedly"
    assert not bad_vetoes, f"unexpected vetoes: {bad_vetoes}"


# ----------------------------------------------------------------------------
# C. abort -> disarm -> veto -> re-arm through the real core
# ----------------------------------------------------------------------------
def test_abort_semantics():
    core = FirmCore(200)
    core.send("R,1")
    core.send("F,2,30000,5000,100000,200")       # 100 ms shot
    core.tick_ms(40)
    assert core.lib.fc_firing() == 1
    core.send("A,2")                             # abort mid-dwell
    assert core.lib.fc_firing() == 0
    assert core.lib.fc_armed() == 0
    core.send("F,3,0,0,1000,200")                # fire while disarmed
    assert any("V,disarmed" in v for v in core.vetoes)
    core.send("R,4")                             # re-arm cycle works
    core.send("F,5,0,0,5000,200")
    assert core.lib.fc_firing() == 1
    core.tick_ms(60)
    assert core.lib.fc_firing() == 0
    print("[C] abort/disarm/veto/rearm through real core: OK")


if __name__ == "__main__":
    build_firmware()
    k = test_vision_engagement()
    test_firmware_in_loop()
    test_abort_semantics()
    print(f"\nE2E PASS: vision kills={k}/14, firmware loop clean, "
          f"abort semantics verified -- code-complete before ordering")

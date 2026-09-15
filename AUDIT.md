# laserzquoteunquote — Audit for Real-World Deployment

**Date:** 2026-09-14 · **Scope:** sim3d.py, flybrain.py, benchmark.py
**Goal:** backyard laser point-defense against mosquitoes/pests, built with physical hardware.

---

## 1. Verdict on the tech stack

**Python + NumPy: right brain, wrong reflexes.** The stack is fine for perception,
tracking, fire-control *solutions*, and simulation — but a real system splits into
two loops, and only one of them should be Python:

| Layer | Rate | Runtime | Why |
|---|---|---|---|
| Perception + tracking + target selection | 30–60 Hz | Python + NumPy (Pi 5 or mini-PC) | Matrices, ML models, dev speed. Proven at this rate. |
| Galvo steering + laser firing + interlocks | 1–10 kHz | C/C++ or Rust on a microcontroller (ESP32/STM32) | A mosquito crossing the FOV gives you ~100–300 ms. Frame-latency Python aim-then-fire wastes most of it. Galvos want a hard-real-time analog/PWM signal, not USB packets. |

This is exactly how the two working precedents do it:
- **Rakhmatulin 2021** (arXiv:2105.14190): Raspberry Pi vision → galvanometer + 1 W
  diode laser, neutralized mosquitoes; explicitly recommends a microcontroller for
  the final device.
- **Photonic Fence** (Intellectual Ventures / Global Good): CCD vision + galvo,
  sub-ms beam placement, deployed for malaria-vector mosquitoes.

### Recommended hardware stack
- **Cameras:** 2× global-shutter USB cameras (e.g. Arducam OV9281, ~$50 ea) for stereo
  3D. Rolling shutter + fast small targets = smeared detections; global shutter is
  non-negotiable. The sim currently assumes true 3D points — stereo triangulation
  must supply them.
- **Compute:** Pi 5 or Jetson Orin Nano if you later want a learned detector.
- **Galvos:** cheap Chinese galvo set (analog ±10 V, ~20 kpps) driven by a 12-bit DAC
  from the MCU — or salvaged laser-show galvos. SPI/I2C DAC, not serial.
- **Laser:** the sim's 8–20 W table is fantasy for diodes. Real kill demos used
  1–2 W focused diodes (445 nm blue or 808/980 nm IR). **IR + safety goggles**, or
  the kids/pets/dog never know where the beam is. Blue is visible = inherently safer.
- **MCU:** ESP32-S3 (cheap, fast, has DAC/PWM, does the interlock loop) with UART/WiFi
  link to the Python brain.

### Software additions needed
- `requirements.txt` (numpy, matplotlib — that's it today) and `hardware/` package.
- Structure the core as a library with clean interfaces so the same fire-control
  code runs in sim and on metal.

---

## 2. Code bugs and defects

### Critical (wrong results / blocks real hardware)
1. **No shot cooldown — dwell is decorative** (`sim3d.py` engage loop). Firing
   happens every frame with a fresh stochastic kill roll, so kills/frame-rate
   scale together. `dwell` in THREATS is never used as an exposure time. On
   hardware, dwell IS the design (laser on-target time per shot). Fix: explicit
   `fire_state` (aim → dwell for N ms → assess → cooldown), decoupled from frame
   rate.
2. **Frame-rate-dependent physics everywhere.** `dt` enters slew, heat decay
   (0.55·dt), prediction, jink probability. Fine if dt is constant; but
   benchmark vs. real-camera latency comparisons are meaningless until the fire
   loop is time-quantized like the real 1 kHz MCU loop.
3. **Sensor model is a lie for this use case** (`Sensor3D.observe`): 1 cm iid
   Gaussian on true 3D positions, 93% detection. Real stereo at 3–10 m has
   range²-squared depth error (baseline-limited), miss/merge during swarm
   crossings, and no per-target identity. Fix: simulate 2D detections per
   camera + triangulation, with range-dependent noise — this also finally makes
   FlyBrain's clutter-rejection job real.
4. **Species classification consults ground truth** (`_classify_species`): reads
   the track's true speed/altitude against the THREATS table. On hardware you
   don't know the species table. Acceptable as a sim shortcut, but label it
   `oracle_classify` and add an honest classifier (size + speed) before
   trusting selective-targeting results.

### High (correctness)
5. **Newborn tracks start with `misses=1`** (Tracker3D.step init + same-loop
   coast) — a consistent-detection track accrues phantom misses; off-by-one in
   track lifetime.
6. **WTA azimuth off by half a bin** (`flybrain.py` wta_step): argmax over
   `sin(az_bins)` then decodes `az = asin(...)` — aliases symmetric peaks and
   loses half-bin accuracy. Should keep bin index and decode arctan2 from the
   2D bin grid.
7. **Heat model has no thermal mass:** instant heat-in per frame, linear decay;
   real laser diodes need duty-cycle + thermistor-based derating. Matters when
   sizing a real diode's duty cycle.
8. **Kill assessment samples the target once per frame at `aim_point`** — no
   beam–target relative-motion integration during the (future, real) dwell.
   Once cooldown lands, integrate p_hit over the dwell window.

### Medium (cleanliness / maintainability)
9. Dead code: `engage(..., tparams)` unused; `_classify()` back-compat wrapper
   unused outside tests; `SCRATCH`/`shutil` in benchmark.py unused.
10. Module-level `AREA`/`HEIGHT` globals in sim3d.py are imported by benchmark.py
    — fine now, but hardware config should move to a dataclass/config file, not
    module constants.
11. `Tracker3D` association gate (0.30 m) and `confirmed` threshold (3 hits) are
    tuned to the too-good sensor; revisit after sensor realism fix.
12. No unit tests, no requirements.txt, no README. Tracker, flybrain math, and
    fire-control deserve tests before hardware money is spent.

---

## 3. Sim → hardware fidelity gaps (what to change before building)

| Sim assumption | Reality | Action |
|---|---|---|
| 3D points handed to tracker | Two 2D cameras + triangulation | Rebuild Sensor3D as StereoPair with range-dependent noise |
| Instant slew to solution, 250°/s | Galvo is fast (kHz) but beam *walks*; mechanical turret is slow | Model galvo FOV cone + turret re-pointing as separate stages |
| p_kill roll per frame | Thermal fluence budget on wing/thorax | Integrate fluence during dwell; kill when threshold exceeded |
| Species table drives power | One diode, fixed power | Power becomes a hardware constant; dwell varies per species |
| No safety layer | Mandatory | Interlocks: human/pet-motion veto zone, per-frame exposure check, watchdog, hardware E-stop line to laser enable |
| Perfect sync | Camera exposure + USB latency + MCU round-trip | Timestamp every stage; measure end-to-end latency on rig, feed back into lead-pursuit |

## 4. Recommended build order

1. Fix the fire loop (cooldown/dwell/fluence) in sim — it's the piece the
   hardware inherits directly.
2. Make the sensor honest (stereo 2D → triangulate). Benchmark again; FlyBrain
   vs Kalman numbers will change.
3. Define `hardware/` interfaces: `Camera`, `Galvo`, `LaserDriver`, `Interlock`
   with sim implementations first.
4. Bench rig: single galvo + low-power visible diode + paper target at 2 m,
   closed-loop latency measurement. Then IR + goggles + live insects.
5. Safety module before any outdoor/backyard power: veto zones, watchdog,
   beam-time budget per activation.

## 5. Precedent
- I. Rakhmatulin, "RaspberryPI for mosquito neutralization by power laser,"
  arXiv:2105.14190 (2021) — 1 W laser + galvo + Pi, working prototype.
- Intellectual Ventures "Photonic Fence" — commercial equivalent, malaria focus.
- Photonic Fence videos show the same pipeline this repo simulates:
  detect → track → lead → fire. The physics shown matches the sim's model well.

## 6. Audit rounds 4–6 (code-complete state, all verified by probes)

Round 4 (commit 6a4c17a): VetoReason enum added ESTOP/DISARMED/POWER to match
the exact C strings (serial path previously collapsed them to WATCHDOG);
benchmark HUD heat now on 0..100 scale.

Round 5 (commits after): S-status frame tolerates CRLF terminations; dead
heat_cp/HEAT_DECAY_PER_S removed in favor of the exact heat_hcp accumulator
(5.5 cP/ms decay, zero integer drift vs sim); protocol.py documents that W
is reserved (T-frame carries the watchdog state).

Round 6: veto-zone parity between drivers. The C core now implements the
same cone-veto list as SimInterlock (Z,<az>,<el>,<r> pushes; Z,END commits
only while disarmed; immutable while armed), checked in the F gate before
thermal. Firmware bench: 66/66 checks. E2E: PASS (8/14 vision kills, 20/20
firmware shots, heat 409, no vetoes, abort/rearm OK). Protocol fuzz:
2000/2000. py_compile clean. Hardware-only remainders: flashing the ESP32
board package, mechanical alignment, real-optics calibration (per BOM in
RIG.md).

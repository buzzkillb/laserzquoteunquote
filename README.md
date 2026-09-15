# laserzquoteunquote

A real, working backyard laser point-defense system for flying insects —
mosquitoes and flies — built from a ~$180 bill of materials and validated
end-to-end in simulation before any hardware is ordered.

One 2 W 445 nm diode, one global-shutter camera, one galvo pair, one
microcontroller doing the reflexes, one Raspberry Pi doing the thinking.
Species selection is software; beam power is never tuned per target.

## Architecture: two loops, one wire protocol

| Loop | Rate | Runs on | Does |
|---|---|---|---|
| Slow (sense/track/select) | 30–60 Hz | Raspberry Pi Zero 2 W (Python) | blob detection, tracking, lead-pursuit, species gating |
| Fast (fire control) | 1 kHz | ESP32-S3 (portable C99) | beam on/off, galvo setpoints, interlock vetoes, thermal model |

The two talk over UART @ 921600 using a frozen, fuzz-tested ASCII line
protocol (`hardware/protocol.py`, implemented by `firmware/fire_control.c`):

```
-> F,<seq>,<az_md>,<el_md>,<dwell_us>,<power_cW>   fire command
-> A,<seq> / -> R,<seq> / -> S                      abort / (re)arm / status
-> Z,<az_md>,<el_md>,<r_md> ... Z,END               veto-zone (cone) push
<- T,<seq>,<az_md>,<el_md>,<flags>,<heat_cP>,<shots>,<beam_ms>   status
<- V,<reason>                                       veto notice
```

Every field is an integer (millidegrees, microseconds, centiwatts) so the
MCU parser needs no floating point. The contract is verified by a
2000-case round-trip fuzz and a 66-check firmware state-machine bench.

## Repository map

```
sim3d.py            3D simulation: swarm physics, PanTiltTurret with
                    thermal model + beam-LINE kill physics, species table
flybrain.py         bio-inspired target selection (EMD/STMD/LGMD + WTA),
                    14-deg attention gate — drives the 30 Hz loop
vision.py           single-camera model: 6 mm lens projection, blob
                    detector, 3-frame track confirmation, mono ranging
                    with dual size hypotheses, waterline reflection filter
hardware/
  interfaces.py     abstract device contracts (LaserDriver, Galvo,
                    Interlock, TurretStatus, VetoReason)
  protocol.py       the wire format: encode/decode + constants (frozen)
  sim_devices.py    Python sim implementations (dwell timing, veto cones)
  serial_devices.py pyserial implementations for the real MCU
firmware/
  fire_control.c/.h the portable C99 fast loop (drop into any ESP32 main)
  test_fire_control.c  host bench: 66 checks, -Wall -Wextra clean
tests/
  e2e_test.py       full-chain test: sim -> virtual IR camera -> vision ->
                    tracker -> fly brain -> turret -> wire frames -> the
                    ACTUAL compiled C core (ctypes) -> status back
RIG.md              hardware contract: BOM, geometry, optics, night mode,
                    SKU ladder
BRINGUP.md          the boxes-to-mosquitoes procedure + symptom tree
AUDIT.md            audit findings log (rounds 1–6, all closed)
```

## Quick start

```bash
pip install -r requirements.txt          # numpy, matplotlib

# simulate: 14 mosquitoes, 2 W rig, 60 s
python3 sim3d.py --seconds 60 --seed 7 --laser-w 2

# benchmark: optimized vs fly-brain vs hybrid controllers
python3 benchmark.py --seconds 60 --seeds 7 8 9

# firmware state-machine bench (host C99)
cd firmware && cc -Wall -Wextra -O2 -o test_fw \
    test_fire_control.c fire_control.c && ./test_fw

# full chain, sim to compiled firmware and back
python3 tests/e2e_test.py
```

## Verified results (as of 2026-09-15)

- Sim baselines, perfect sensor: **14/14 mosquitoes** killed in 60 s, all
  three seeds (7, 8, 9); mixed field: 14 mosquitoes + 10 flies = 24.
- Full vision-fed chain (camera noise, mono ranging, water twins):
  **8/14 in 30 s** — the gap to the perfect-sensor baseline is the honest
  cost of single-camera ranging, and it is stable across reruns.
- Firmware core: **66/66** bench checks — arm discipline, veto-chain order
  (estop → disarmed → zones → thermal → power), thermal saturation and
  decay, watchdog, mid-dwell abort truncation, CRLF/garbage-frame
  immunity, and heat agreement with the sim model to < 1% over a 30-shot
  burn (5.5 cP/ms decay via a half-centi-percent accumulator, zero
  integer drift).
- Protocol: 2000/2000 fuzz round-trips, zero failures.
- Vision: worst beam-ray error **0.008°** (≈0.6 mm on the 9 mm spot at
  4 m); blob/track counts stay consistent over 600-frame runs.

## What is hardware-only

Flashing `fire_control.c` into an ESP32 board package, mechanical camera↔
galvo alignment, and real-optics calibration constants. All of it is
covered step-by-step in `BRINGUP.md` (power-on with the diode
disconnected, galvo calibration card, first light at 0.4 W, IR strobe
sync, beam-camera registration, software dry run, dusk field test, and a
symptom-keyed troubleshooting tree).

## Design notes worth knowing

- **Single camera, not stereo.** The galvo's own angle feedback plus a
  fixed 2–6 m depth band replaces triangulation; spherical (az, el,
  range) coordinates are the native frame for both the tracker and the
  beam, so there is no Cartesian round-trip.
- **Beam-LINE kill physics.** The galvo controls angle, not range: the
  beam doses anything on the beam line at its own range, which is what
  makes the parallax between the co-mounted camera and turret harmless.
- **Night mode** = $10 850 nm IR illuminator strobed ~1 ms/frame, which
  requires global shutter (rolling shutter smears small fast targets).
- **Birds/pets/people** are spared by the classifier gates
  (speed/size/altitude bands + angular veto cones), enforced twice: in
  the Python slow loop and independently in the MCU's veto chain.

## Precedent

- I. Rakhmatulin, "RaspberryPI for mosquito neutralization by power
  laser," arXiv:2105.14190 (2021).
- Intellectual Ventures "Photonic Fence" — same pipeline: detect →
  track → lead → fire.

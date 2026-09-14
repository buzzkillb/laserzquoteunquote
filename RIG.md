# laserz backyard rig — "one diode, one spot, one rule"

Design goal: **cheap enough to sell, simple enough to build in a weekend.**
Every dollar of complexity must earn its keep. This file is the hardware
contract the sim (sim3d.py, `--laser-w 2`) and the `hardware/` package
both implement.

## The simplification decision

The sim's original species table implied a tunable multi-watt laser with
per-species power tiers (up to 60 W for drones — that's a mains-powered
device, a laser-safety filing, and a product nobody buys). The backyard
product is the opposite:

> **ONE fixed 2 W diode. Species selection is software. Power is never
> tuned per target.**

- 2 W focused is lethal to insects, harmless to birds at our spot size,
  and below the Class 4 "can burn skin instantly" regime — the safety
  story a consumer product needs.
- One diode = one driver, one thermistor, one duty cycle. No power DAC.
- Birds are spared by the classifier (`speed < ~2.5 m/s OR size > 30 mm
  → do not fire`), not by changing beam power.
- Drones are simply out of scope for v1 (they're fast and don't damage
  anything if spared; they get a firmware gate, not a laser upgrade).

## Bill of materials (~$210 single-unit; ~$150 at 100 units)

| Item | Choice | ~Cost | Notes |
|---|---|---|---|
| Compute | Raspberry Pi Zero 2 W | $15 | Runs vision+tracker at 30 Hz headless. Zero 2 W keeps the product silent and under 5 W total. |
| Camera | Arducam OV9281 (global shutter, mono) ×1 | $40 | ONE camera, not stereo — see geometry below. |
| Laser | 2 W 445 nm blue diode module + TTL driver | $35 | Visible = inherently safer; the beam position is always knowable. |
| Galvo | 20 kpps analog galvo set (±10 V) | $45 | The only "exotic" part; proven in laser-show hardware. |
| DAC | MCP4728 quad I²C DAC | $3 | Drives galvo X/Y; the MCU streams setpoints. |
| MCU | ESP32-S3 | $6 | The 1 kHz fast loop (protocol.py frames); owns interlocks. |
| Beam combiner | dichroic or half-mirror | $8 | Aligns the (invisible) aiming indicator with the kill beam. |
| Aiming aid | 650 nm 5 mW red dot | $2 | Visual feedback for setup/debug; helps users trust the box. |
| Power | 5 V 4 A supply + buck for diode | $12 | Whole device < 5 W idle, ~12 W firing. |
| Mechanics | printed enclosure + galvo mount | $5 | |
| Misc | optics, wiring, fasteners | $9 | |
| **Total** | | **~$180** | Retail ~$350–400 is a sane price point. |

## Geometry: how one camera gives 3D

Stereo would double camera cost and calibration pain. Instead:

- The galvo's own **position feedback pots give beam angles to ~0.1°**.
- Distance comes from a **VL53L4CX ToF sensor on the turret** (short-range,
  multi-object, ~$8) — or simply: fire only inside a **fixed depth band**
  (targets in a 2–6 m slab; anything closer/farther is vetoed).
- The tracker runs on (az, el, range) in spherical coords — which is
  actually the *native* frame for galvo aiming, so no Cartesian round-trip.

This drops the BOM by ~$40 and the calibration complexity by half vs.
stereo, and matches the veto-cone interlock model in
`hardware/interfaces.py` (zones are angular cones — exactly what a
single-camera + depth-band system computes naturally).

## The three-rate architecture (unchanged, now cheap)

| Loop | Rate | Hardware |
|---|---|---|
| Sense/track/select | 30 Hz | Pi Zero 2 W, Python, this repo's sim3d pipeline |
| Galvo/DAC streaming | 1 kHz | ESP32-S3, Rust or C, `hardware/protocol.py` frames |
| Safety veto | on-demand + watchdog | MCU-local; can always say no |

## Optics & night operation (validated)

**Lens: 6 mm** on the OV9281 (34° HFOV, ~3 m × 1.9 m coverage at 5 m standoff).
Pixels-on-target math (2.9 µm pitch, 1280 px):

| Target | Detectable (≥1.5 px) |
|---|---|
| Mosquito (5 mm) | ~7 m |
| Fly (8 mm) | ~11 m |
| Moth (20 mm) | ~28 m |

**The camera always out-ranges the 2 W beam (~6–8 m kill envelope).**
Design invariant: detection range ≥ engagement range at every distance.
Beyond ~8 m the beam can't build kill fluence (0.6 W/cm² at 12 m), so the
depth-band veto is physics, not just software policy.

**Night mode:** OV9281 is natively near-IR sensitive. Add a **$10 850 nm IR
illuminator, strobed ~1 ms/frame** — motion-frozen dots at 30 fps, no smear.
Global shutter (already in BOM) is required for the strobe trick. Dusk/night
is when mosquitoes are most active; high-contrast dots on dark background is
exactly the regime the FlyBrain STMD pipeline is tuned for. IR LED at 850 nm
is invisible to humans; the red aiming dot stays visible for setup only.

## SKU ladder

| | Backyard (v1) | Pro |
|---|---|---|
| Laser | 2 W 445 nm, ~6–8 m kill envelope | 3–5 W tight-collimation, ~10–15 m |
| Lens | 6 mm (34° FOV) | 12 mm (18° FOV, tracks to ~14 m mosq / ~22 m fly) |
| Range tactic | engage in 2–6 m slab | **track at range, engage on entry** — maintain tracks in the far cone, fire the moment targets cross the kill envelope |
| BOM | ~$180 | ~$230 |

The tracker already supports track-then-engage (confirmed tracks persist
across the `misses<4` coast window); the Pro SKU is mostly a lens + laser
module change, not new software.

## What the sim now models faithfully

- `--laser-w 2` → the one-diode rig; species auto-sizing is legacy
  (`--laser-w 0`).
- Shot cooldown (50 ms) in `PanTiltTurret` == the MCU fire-loop cadence.
- Dwell per species from THREATS table == MCU-enforced exposure time.
- Selective targeting == the classifier, not power tuning.
- Veto cones == `hardware/interfaces.py` Interlock (angular zones).

## What's still sim-vs-honesty gap (fine for v1 tuning, not for production claims)

1. Sensor is still "true 3D + 1 cm noise" — the single-camera + ToF/depth-band
   rig will have range-dependent error; tune `pkill` down 10–20% when the real
   sensor lands.
2. No wind (backyard reality): add a lateral drift term to swarm motion before
   publishing kill-rate numbers.
3. Outdoor range: flux attenuation `exp(-0.03·R)` is indoor-ish; re-measure on
   the bench at 3–6 m before quoting effectiveness.

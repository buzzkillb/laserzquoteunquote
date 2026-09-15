# BRINGUP.md -- from boxes to bugs
Procedures for the first bench build and dusk field test. The sim,
firmware core, vision pipeline, and protocol are already code-verified
(see tests/e2e_test.py); this file covers the parts that only exist
once hardware does.

## 0. What this does NOT cover
- ESP32 board support package install (use the Arduino IDE or ESP-IDF
  with fire_control.c as the loop core -- it is HAL-free C99 and
  compiles unchanged).
- Mechanical mounting. Aim for: camera and diode co-mounted within 2 cm,
  camera level (the sim's validated rig geometry), galvo after the
  beam expander.

## 1. Power-on, no beam (safe)
1. Bench supply at 5 V, current limit 2 A. Wire ESP32 + galvo driver +
   camera, leave the laser diode UNCONNECTED.
2. Flash firmware. Open serial at 115200. Send `S` -- expect
   `T,0,0,0,0,0,0,0` (disarmed, heat 0).
3. Send `R,1` then `S` -- flags should now show armed (bit 0 = 1).
4. Send `F,2,0,0,1000,100` -- galvo should visibly deflect to az=0,
   el=0 and return. Beam stays off (diode disconnected). After 1 s,
   `S` shows shots=1.

## 2. Galvo calibration (still no beam)
1. Tape a target card 3 m from the galvo mirror.
2. Send `F,3,0,0,50000,0` -- power 0 cW, so you can watch the galvo
   without any light. Wait -- with power 0 the frame is vetoed? No:
   power_cw=0 <= limit, so the shot programs and the galvo moves while
   the diode emits nothing. Mark the spot.
3. Sweep az in steps of 5000 md (5 deg) and mark. The spots should be
   evenly spaced; if not, the galvo driver gain needs trimming. Record
   the mapping -- this is your beam-pointing calibration table.
4. Repeat for el.

## 3. First light (low power)
1. Wear laser glasses rated for 445 nm. Clear the beam path. Nobody
   eye-level with the bench.
2. Connect the diode through its driver with the current set to 20% of
   rating. Send `F,4,0,0,20000,40` (0.4 W, 20 ms). A dim blue dot
   should appear at the az=0/el=0 mark from step 2.
3. If the dot is not at the mark: do NOT chase it with software yet.
   Fix mechanical alignment until zero-command = zero-mark. Software
   offsets hide mechanical drift.
4. Repeat at a few galvo angles. Verify the dot lands on each
   calibration mark within ~2 mm at 3 m (~0.7 mrad pointing error,
   plenty for a 9 mm spot).

## 4. Camera bring-up
1. Enable the camera (raspi-config), confirm `libcamera-hello --list-
   cameras` shows the OV9281.
2. Capture a frame pointed at the target card with room lights on:
   `libcamera-still -o t.png --width 1280 --height 800`. Confirm the
   card and the (off) galvo mirror are visible and the framing matches
   the sim's geometry (level camera, swarm band centered).
3. IR strobe: wire the 850 nm illuminator to a GPIO via MOSFET. Sync it
   to the exposure window (global shutter: flash during integration).
   Capture with lights OFF -- card should still be visible in the PNG.
4. Run vision.py's BlobDetector on a live frame with a fly-sized speck
   on the card: confirm one blob at the right (u, v).

## 5. Beam-camera registration
The camera and diode are co-mounted, so registration is a fixed affine
offset: for a test point at (az, el) measured by the camera, the beam
command that lands the dot on it is (az + daz, el + del). Measure daz,
del once with the card at 3 m and again at 5 m; they should agree. Put
the offsets in the Pi's config, not the firmware.

## 6. Software dry run (no beam, full loop)
1. Run the Pi loop against real camera frames but with the diode
   disconnected: tracker locks the calibration speck, turret produces
   fire frames, MCU (still connected) counts shots. This validates the
   whole chain with zero optical risk -- it is tests/e2e_test.py but
   with `render_frame` swapped for the real camera read.

## 7. Dusk field test (first live fire at bugs)
1. Site: pool deck or patio, turret 3-5 m from the swarm zone, mount
   height ~1.5 m, camera level.
2. Depth-band veto: confirm it is armed at 1.5-9 m (config), so the
   beam can never dwell on anything near the property line.
3. Lights off, IR strobe on. Watch the tracker console: mosquitoes over
   the water should appear as confirmed tracks in the 0.3-2.6 m band.
   Waterline filter: set cam.waterline_v from the deck horizon.
4. Arm (`R`), enable targeting, start with dwell 30 ms / 2.0 W. Success
   = audible/visible mosquito drop within a few seconds of track lock.
5. Expect 50-70% of the sim's kill rate on night one (sim: 8-10/14
   through the vision chain). If 0 kills with many shots: the miss is
   registration (step 5) or dwell; check beam_ms vs heat in `S` frames.

## 8. Known-good numbers to compare against
- Fire loop: 60 Hz commands, 1 kHz MCU tick, 50 ms cooldown
- Shot: 30 ms @ 2 W for mosquitoes (sim dwell table)
- Heat: 1200 cP per full-power shot, decays 5500 cP/s (firmware core
  matches sim3d.py's thermal model to <1%)
- Camera: 1280x800 @ 30-60 fps, 6 mm lens, mosquito dot 2-4 px at 3-7 m
- E2E sim chain: 8-10 kills / 14 mosquitoes / 30 s through camera+
  vision+tracker (13-14/14 with a perfect sensor)

## 9. When something does not match the sim
1. Beam misses left/right -> galvo calibration table (step 2)
2. Beam misses up/down -> camera registration (step 5)
3. Tracks but no shots -> species gate: check speed/altitude against
   THREATS table (a mosquito flying like a fly gets treated like one)
4. Shots but no kills -> dwell too short for real wing/wingbeat flux;
   raise dwell in 10 ms steps while watching heat
5. Watchdog trips in the field -> Pi loop overrunning 500 ms; check for
   camera stalls; the MCU is doing its job

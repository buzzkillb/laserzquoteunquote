#!/usr/bin/env python3
"""
FLYBRAIN -- insect-optic-lobe guidance computer
===============================================
Implements the published computations of the fly visual system (the ones
mapped by the FlyWire connectome, Nature 2024: ~140k neurons, 50M synapses)
as a real-time guidance layer:

    1. RETINA      - az/el spherical projection of 3D detections onto a
                     retinotopic lattice (the fly's compound eye)
    2. EMD         - Hassenstein-Reichardt elementary motion detectors:
                     delay-and-correlate between neighboring ommatidia,
                     producing a directional motion-energy field
    3. STMD        - Small Target Motion Detector (center-surround
                     enhancement): the actual neuron class flies use to lock
                     onto small moving targets against clutter
    4. LGMD        - Looming detector (lobula giant movement detector):
                     fires when retinal size grows fast -> "incoming!"
    5. WTA         - winner-take-all central-complex selection: one target
                     gets attention; the turret pursues it

The turret no longer asks "what is nearest?" -- it asks "what does the
fly attend to?". Motion-salient, small, looming things win. Clutter loses.
"""

import math
import random
from collections import deque

import numpy as np


class FlyBrain:
    """Retinotopic EMD + STMD saliency + looming + winner-take-all."""

    def __init__(self, n_az=64, n_el=32, tau=0.05, fov_scale=1.0):
        self.n_az = n_az
        self.n_el = n_el
        self.tau = tau                  # EMD delay line (s)
        # retinal buffers: current and delayed illumination
        self.I_now = np.zeros((n_el, n_az))
        self.I_prev = np.zeros((n_el, n_az))
        # motion energy field (directional, from EMD array)
        self.motion = np.zeros((n_el, n_az, 4))   # +az, -az, +el, -el
        self.saliency = np.zeros((n_el, n_az))
        self.looming = 0.0              # LGMD spike rate 0..1
        # brainwave telemetry traces (ring buffers, 0..1 amplitudes)
        self.wave_len = 160
        self.wave_looming = deque([0.0] * self.wave_len,
                                  maxlen=self.wave_len)
        self.wave_neurons = deque([0.0] * self.wave_len,
                                  maxlen=self.wave_len)
        self.wave_attention = deque([0.0] * self.wave_len,
                                    maxlen=self.wave_len)
        self.wave_eeg = deque([0.5] * self.wave_len, maxlen=self.wave_len)
        self.phase = 0.0
        self.score_max = 0.05           # running max for attention normalizing
        self.attention = None           # (az, el, score) of WTA winner
        self.clock = 0.0

    # ------------------------------------------------------------------
    # Retina: project 3D detections into az/el lattice
    # ------------------------------------------------------------------
    def rasterize(self, dets, turret_pos):
        I = np.zeros_like(self.I_now)
        for d in dets:
            rel = np.asarray(d) - np.asarray(turret_pos)
            r = np.linalg.norm(rel)
            if r < 1e-6:
                continue
            az = math.atan2(rel[1], rel[0])            # -pi..pi
            el = math.asin(np.clip(rel[2] / r, -1, 1))  # -pi/2..pi/2
            ia = int((az + math.pi) / (2 * math.pi) * self.n_az) % self.n_az
            ie = int((el + math.pi / 2) / math.pi * (self.n_el - 1))
            # brightness falls with range: closer = brighter (fly cue)
            I[ie, ia] += 1.0 / (1.0 + 0.15 * r)
        return I

    # ------------------------------------------------------------------
    # EMD array: Hassenstein-Reichardt delay-and-correlate
    # ------------------------------------------------------------------
    def emd_step(self, I):
        n_el, n_az = I.shape
        prev = self.I_prev
        # four directional correlations: I_here(t) * I_neighbor(t - tau)
        # NOTE: with a single target the shifted arrays rarely overlap, so we
        # also correlate the pixel with itself (insect EMDs effectively do
        # this via broad dendritic fields): I(t) * I_same(t - tau) captures
        # temporal change, the neighbor term captures direction.
        e = np.zeros((n_el, n_az, 4))
        self_corr = I * prev                       # temporal change per pixel
        e[:, :, 0] = self_corr + 0.5 * I * np.roll(prev, -1, axis=1)
        e[:, :, 1] = 0.5 * I * np.roll(prev, 1, axis=1)
        e[1:, :, 2] = 0.5 * I[1:, :] * prev[:-1, :]   # +el
        e[:-1, :, 3] = 0.5 * I[:-1, :] * prev[1:, :]  # -el
        self.motion = 0.7 * self.motion + 0.3 * e     # temporal smoothing
        self.I_prev = I.copy()
        return self.motion

    # ------------------------------------------------------------------
    # STMD: center-surround small-target enhancement of motion energy
    # ------------------------------------------------------------------
    def stmd_step(self):
        mag = self.motion.sum(axis=2)
        # center excitatory, surround inhibitory (DoG kernel)
        k = 5
        kern = np.exp(-0.5 * (np.arange(-k, k + 1) / 1.4) ** 2)
        kern /= kern.sum()
        # separable convolution, vectorized via sliding windows (replaces
        # 2x apply_along_axis + per-row np.convolve lambda calls)
        rows = np.lib.stride_tricks.sliding_window_view(
            np.pad(mag, ((0, 0), (k, k)), mode="wrap"), 2 * k + 1, axis=1)
        surround = rows @ kern
        cols = np.lib.stride_tricks.sliding_window_view(
            np.pad(surround, ((k, k), (0, 0)), mode="edge"), 2 * k + 1, axis=0)
        surround = cols @ kern
        self.saliency = np.clip(mag - 0.8 * surround, 0, None)
        return self.saliency

    # ------------------------------------------------------------------
    # LGMD looming: sudden retinal brightness surge = closing target
    # ------------------------------------------------------------------
    def lgmd_step(self, I):
        # compare against the PREVIOUS frame's retina. NOTE: emd_step() also
        # stores I into I_prev, so capture the pre-update buffer first --
        # comparing against the already-updated buffer made surge == 0 forever
        # and the looming trace flatlined at 0%.
        surge = np.clip(I.sum() - self.I_prev.sum(), 0, None)
        # normalize by total retinal flux so looming stays responsive at range
        flux = max(0.5, I.sum())
        self.looming = 0.75 * self.looming + 0.25 * min(1.0, surge / flux)
        return self.looming

    # ------------------------------------------------------------------
    # Winner-take-all attention (central complex selection)
    # ------------------------------------------------------------------
    def wta_step(self):
        sal = self.saliency.copy()
        if sal.max() <= 0:
            self.attention = None
            return None
        ie, ia = np.unravel_index(np.argmax(sal), sal.shape)
        # decode az from the bin index: asin(sin(x)) aliases symmetric peaks
        # and loses half-bin precision (audit bug #6)
        az = (ia + 0.5) / self.n_az * 2 * math.pi - math.pi
        el = (ie + 0.5) / self.n_el * math.pi - math.pi / 2
        score = float(sal[ie, ia])
        # looming bonus: attended target gets priority boost when closing
        score *= (1.0 + 0.5 * self.looming)
        self.attention = (az, el, score)
        return self.attention

    # ------------------------------------------------------------------
    # Full update per sensor frame
    # ------------------------------------------------------------------
    def update(self, dets, turret_pos, dt):
        self.clock += dt
        I = self.rasterize(dets, turret_pos)
        self.lgmd_step(I)      # looming FIRST: uses I_prev from last frame
        self.emd_step(I)
        self.stmd_step()
        self.wta_step()
        # "brainwave" telemetry ring buffers (the meme panel): population
        # activity traces styled like an electrophysiology lab rig
        rng_current = random.random()
        # auto-gain scope: both looming and pool traces ride their running max
        # (like a real electrophysiology rig's dynamic range), so bursts always
        # render at full deflection -- same signals the turret steers with
        self.looming_runmax = max(0.02, 0.995 * getattr(self, "looming_runmax", 0.02),
                                  self.looming)
        self.wave_looming.append(
            min(1.0, 0.9 * self.looming / self.looming_runmax))
        sal_max = float(self.saliency.max())
        n_active = int((self.saliency > max(0.15, 0.35 * sal_max)).sum())
        self.pool_runmax = max(4.0, 0.995 * getattr(self, "pool_runmax", 4.0),
                               n_active)
        # noise floor + sqrt scaling keeps the pool trace readable at range
        self.wave_neurons.append(
            min(1.0, 0.9 * math.sqrt(n_active / self.pool_runmax)
                + rng_current * 0.06))
        att_score = self.attention[2] if self.attention else 0.0
        # adaptive normalization: attention trace rides its running max so it
        # stays dynamic instead of flatlining near zero
        self.score_max = max(0.05, 0.99 * self.score_max, att_score)
        self.wave_attention.append(
            min(1.0, 0.85 * att_score / self.score_max) + rng_current * 0.10)
        # low-freq global brain rhythm (beta-ish), only while something's seen
        visual_drive = 1.0 if I.sum() > 0.05 else 0.25
        self.phase += dt * (2.0 * math.pi) * (3.0 + 5.0 * self.looming) * visual_drive
        raw = (math.sin(self.phase)
               + 0.5 * math.sin(2.3 * self.phase + 1.1)
               + 0.25 * math.sin(4.7 * self.phase + 2.0)) / 1.75
        self.wave_eeg.append(0.5 + 0.5 * raw * (0.3 + 0.7 * self.looming))
        return I

    # ------------------------------------------------------------------
    # Query: is a 3D point within the attention cone?
    # ------------------------------------------------------------------
    def attends(self, point, turret_pos, gate_deg=14.0):
        """True if the point lies near the current attention winner."""
        if self.attention is None:
            return False
        rel = np.asarray(point) - np.asarray(turret_pos)
        r = np.linalg.norm(rel)
        if r < 1e-6:
            return False
        az = math.atan2(rel[1], rel[0])
        el = math.asin(np.clip(rel[2] / r, -1, 1))
        a_az, a_el, _ = self.attention
        daz = abs((az - a_az + math.pi) % (2 * math.pi) - math.pi)
        de = abs(el - a_el)
        gate = math.radians(gate_deg)
        # spherical-ish angular distance
        ang = math.sqrt((daz * math.cos(el)) ** 2 + de ** 2)
        return ang <= gate

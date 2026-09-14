"""
Simulated device implementations.

These mirror sim3d.py behavior but through the hardware interfaces, so
the fire-control logic that will run against real metal is exercised
in the sim first. The kill-physics remains sim3d's stochastic model;
what changes vs. sim3d's current turret is:
  * dwell is a real timed exposure (fixes frame-rate-dependent kills)
  * interlock veto is enforced on every shot
"""

import math
import time
from typing import List, Optional

from hardware.interfaces import (
    FireCommand,
    InterlockState,
    TurretStatus,
    VetoReason,
    Galvo,
    LaserDriver,
    Interlock,
)


class SimGalvo(Galvo):
    """Point-to-point slew at max_slew deg/s, same model as sim3d."""

    def __init__(self, max_slew_dps: float = 250.0, tolerance_rad: float = 0.005):
        self.max_slew = math.radians(max_slew_dps)
        self.tol = tolerance_rad
        self.az = 0.0
        self.el = 0.0
        self._target_az = 0.0
        self._target_el = 0.0
        self._last_t = time.monotonic()

    def aim(self, az: float, el: float) -> None:
        self._integrate()
        self._target_az, self._target_el = az, el

    def ready(self) -> bool:
        self._integrate()
        daz = abs((self._target_az - self.az + math.pi) % (2 * math.pi) - math.pi)
        de = abs(self._target_el - self.el)
        return daz <= self.tol and de <= self.tol

    def _integrate(self) -> None:
        now = time.monotonic()
        # integrate over ALL elapsed time since the last poll (wall-clock
        # motion, like a real galvo), clamped only to swallow pauses
        dt = min(1.0, now - self._last_t)
        self._last_t = now
        daz = (self._target_az - self.az + math.pi) % (2 * math.pi) - math.pi
        de = self._target_el - self.el
        step = self.max_slew * dt
        self.az += max(-step, min(step, daz))
        self.el += max(-step, min(step, de))


class SimLaser(LaserDriver):
    """Timed-dwell laser with cooldown, thermal heat, and stochastic
    kill callback. Kill assessment is injected (sim3d's p_kill model)
    via on_shot_complete so device code stays physics-free."""

    def __init__(self, max_power_w: float = 2.0, cooldown_ms: float = 40.0):
        self.max_power_w = max_power_w
        self.cooldown_s = cooldown_ms / 1000.0
        self._shot_end = 0.0        # monotonic time current shot ends
        self._next_ready = 0.0      # monotonic time cooldown lifts
        self._pending: Optional[FireCommand] = None
        self._status = TurretStatus()
        self.on_shot_complete = None    # callable(cmd, elapsed_s) -> None
        self.on_interlock_block = None  # callable(cmd, state) -> None

    def fire(self, dwell_ms: float, power_w: float) -> None:
        now = time.monotonic()
        if now < self._next_ready or now < self._shot_end:
            return  # MCU would simply refuse; slow loop reads status()
        self._pending = FireCommand(az=0.0, el=0.0, dwell_ms=dwell_ms,
                                    power_w=power_w, expires_s=now + 0.5,
                                    seq=self._status.last_cmd_seq + 1)
        self._shot_end = now + dwell_ms / 1000.0
        self._next_ready = self._shot_end + self.cooldown_s
        self._status.firing = True
        self._status.shots += 1
        self._status.last_cmd_seq = self._pending.seq

    def stop(self) -> None:
        if self._status.firing:
            self._status.firing = False
            self._pending = None

    def status(self) -> TurretStatus:
        now = time.monotonic()
        if self._status.firing and now >= self._shot_end:
            self._status.firing = False
            if self._pending is not None:
                self._status.beam_time_ms += self._pending.dwell_ms
                if self.on_shot_complete is not None:
                    self.on_shot_complete(self._pending,
                                          self._pending.dwell_ms / 1000.0)
                self._pending = None
        return self._status


class SimInterlock(Interlock):
    """Veto zones as angular cones around the turret + watchdog."""
    veto: Optional[VetoReason] = None

    def __init__(self, watchdog_s: float = 0.5):
        self._zones: List[tuple] = []   # (az, el, radius_rad)
        self._watchdog_s = watchdog_s
        self._last_heartbeat = time.monotonic()
        self._armed = False

    def set_veto_zones(self, zones) -> None:
        assert not self._armed, "veto zones are immutable while armed"
        self._zones = list(zones)

    def arm(self) -> None:
        self._last_heartbeat = time.monotonic()
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def evaluate(self, cmd: FireCommand) -> InterlockState:
        now = time.monotonic()
        if not self._armed:
            return InterlockState(ok_to_fire=False, veto=VetoReason.ARMED_OFF,
                                  armed=False)
        if cmd.is_stale(now):
            return InterlockState(ok_to_fire=False, veto=VetoReason.WATCHDOG,
                                  armed=True)
        if now - self._last_heartbeat > self._watchdog_s:
            return InterlockState(ok_to_fire=False, veto=VetoReason.WATCHDOG,
                                  armed=True)
        # beam-corridor check: any veto cone within corridor blocks the shot
        for (zaz, zel, rad) in self._zones:
            daz = abs((cmd.az - zaz + math.pi) % (2 * math.pi) - math.pi)
            de = abs(cmd.el - zel)
            if math.hypot(daz, de) <= rad:
                return InterlockState(ok_to_fire=False,
                                      veto=VetoReason.HUMAN_IN_BEAM,
                                      armed=True)
        return InterlockState(ok_to_fire=True, veto=VetoReason.NONE, armed=True)

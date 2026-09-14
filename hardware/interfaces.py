"""
Abstract device interfaces for the laserz fire-control boundary.

Design rules (from AUDIT.md):
  1. The slow loop (Python) NEVER fires a laser directly. It issues
     FireCommand intents; the fast loop (MCU) executes or vetoes them.
  2. Sim and hardware implementations are interchangeable so the same
     tracker/fire-control code runs against both.
  3. All geometry is SI (meters, radians, watts, seconds) except where
     the protocol says otherwise (millimeters, millidegrees, centiwatts
     for integer transport -- see protocol.py).
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple


class VetoReason(Enum):
    NONE = "none"
    HUMAN_IN_BEAM = "human_in_beam"      # motion veto zone occupied
    PET_IN_BEAM = "pet_in_beam"          # separate pet zone (collars beacon)
    WATCHDOG = "watchdog"                # no heartbeat from slow loop
    THERMAL = "thermal"                  # diode thermistor derating
    BEAM_BUDGET = "beam_budget"          # dwell-time budget exhausted
    E_STOP = "e_stop"                    # hardware emergency stop line
    ARMED_OFF = "armed_off"              # system not armed


@dataclass
class FireCommand:
    """One engagement intent, issued by the slow loop at ~60 Hz.

    az/el      : absolute aim angles (rad) in the turret frame
    dwell_ms   : on-target exposure time for THIS shot (the real dwell --
                 fixes the sim bug where dwell was decorative)
    power_w    : commanded optical power (MCU clamps to hardware max)
    target_id  : slow-loop track id, for telemetry only
    expires_s  : monotonic-clock deadline; stale commands are dropped
    """
    az: float
    el: float
    dwell_ms: float
    power_w: float
    target_id: int = -1
    expires_s: float = 0.0
    seq: int = 0

    def is_stale(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        # 0.0 = no deadline; anything else (including past times, which can
        # be near zero on monotonic clocks that start at 0) is compared.
        return self.expires_s != 0.0 and now > self.expires_s


@dataclass
class InterlockState:
    """Veto authority snapshot. Computed in the fast loop; the slow loop
    reads it to understand refusals but cannot override it."""
    ok_to_fire: bool = False
    veto: VetoReason = VetoReason.NONE
    armed: bool = False
    beam_time_used_ms: float = 0.0     # rolling window budget
    beam_time_budget_ms: float = 0.0
    diode_temp_c: float = 0.0
    last_shot_ms_ago: float = 0.0


@dataclass
class TurretStatus:
    """Telemetry from the fast loop, consumed by the slow loop."""
    az: float = 0.0
    el: float = 0.0
    firing: bool = False
    heat: float = 0.0
    last_cmd_seq: int = -1
    shots: int = 0
    beam_time_ms: float = 0.0
    interlock: InterlockState = field(default_factory=InterlockState)


class Camera(ABC):
    """One synchronized frame set from every camera in the rig.
    Implementations: sim (perfect truth + noise model), V4L2/Spinnaker
    (real global-shutter cams). Frames carry timestamps so end-to-end
    latency can be measured and fed into lead-pursuit."""

    @abstractmethod
    def grab(self) -> Tuple[float, List]:
        """Return (capture_timestamp_monotonic, frames) where frames is
        one image per camera, pixel-aligned and software-synced."""


class Galvo(ABC):
    """Beam steering. The sim implementation integrates point-to-point
    moves at max slew; the real one streams DAC setpoints at >=1 kHz."""

    @abstractmethod
    def aim(self, az: float, el: float) -> None:
        """Command beam angle (rad). Non-blocking."""

    @abstractmethod
    def ready(self) -> bool:
        """True when beam is on-target within tolerance."""


class LaserDriver(ABC):
    """Laser power control + dwell execution. The MCU owns timing so
    dwell accuracy does not depend on the Python frame clock."""

    @abstractmethod
    def fire(self, dwell_ms: float, power_w: float) -> None:
        """Begin a dwell shot. Returns immediately; progress is polled
        via status(). The MCU enforces its own cooldown between shots."""

    @abstractmethod
    def stop(self) -> None:
        """Immediate beam-off (not the E-stop; that is hardwired)."""

    @abstractmethod
    def status(self) -> TurretStatus:
        """Latest fast-loop telemetry snapshot."""


class Interlock(ABC):
    """Safety veto layer. The ONLY component allowed to say no, and it
    must do so locally in the fast loop. Backyard rule: people, pets,
    and the property line are veto zones; the veto list is set at arm
    time and cannot be modified mid-flight from the slow loop."""

    @abstractmethod
    def evaluate(self, cmd: FireCommand) -> InterlockState:
        """Full pre-fire check: stale command, veto zones along the beam
        corridor, watchdog freshness, thermal derate, beam budget."""

    @abstractmethod
    def set_veto_zones(self, zones: Sequence) -> None:
        """(az, el, angular_radius) cones that block firing. Only
        callable while disarmed."""

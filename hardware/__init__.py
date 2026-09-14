"""
laserz hardware abstraction layer
=================================

Boundary contract between the two loops:

    SLOW LOOP (Python, 30-60 Hz, this process)
        cameras -> detection -> tracking -> target selection
        -> FireCommand(az, el, dwell_ms, power_w) sent down

    FAST LOOP (MCU: Rust/embassy or C, 1-10 kHz, on ESP32/STM32)
        executes FireCommands against the galvo DAC + laser PWM
        and owns LOCAL veto authority via InterlockState.

The MCU must be able to refuse to fire without asking the Pi:
human/pet-in-beam veto, watchdog timeout, and beam-time budget are
enforced in the fast loop (InterlockState), never only in Python.

Every device is defined by an abstract interface in interfaces.py,
with a sim implementation (sim_devices.py) used to keep sim3d.py and
the bench rig code running against identical logic, and a serial
implementation (serial_devices.py) that speaks protocol.py framing
to the real MCU.
"""

from hardware.interfaces import (
    FireCommand,
    InterlockState,
    TurretStatus,
    VetoReason,
    Camera,
    Galvo,
    LaserDriver,
    Interlock,
)
from hardware.protocol import (
    CMD_FIRE,
    CMD_ABORT,
    CMD_STATUS,
    RSP_STATUS,
    RSP_VETO,
    encode_fire,
    decode_fire,
    encode_abort,
    encode_status_request,
)
from hardware.sim_devices import SimGalvo, SimLaser, SimInterlock

__all__ = [
    "FireCommand", "InterlockState", "TurretStatus", "VetoReason",
    "Camera", "Galvo", "LaserDriver", "Interlock",
    "CMD_FIRE", "CMD_ABORT", "CMD_STATUS", "RSP_STATUS", "RSP_VETO",
    "encode_fire", "decode_fire", "encode_abort", "encode_status_request",
]

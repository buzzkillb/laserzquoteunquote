"""
Wire protocol between the slow loop (Python) and the fast loop (MCU).

Transport: UART @ 921600 8N1 (or USB-CDC). ASCII lines, LF-terminated,
for debuggability with a plain terminal. All numeric fields are
integers; units are chosen so values fit int32 with headroom:

    az, el      : milliradians of angle? No -- millidegrees (md), range
                  +/-180000, because galvo calibrators speak degrees
    dwell       : microseconds (us), max 1_000_000
    power       : centiwatts (cW), max 2 W -> 200
    seq         : uint16 rolling command counter
    flags       : bitmask (bit0 armed, bit1 firing, bit2 estop, bit3 thermal)

Frames:
    -> F,<seq>,<az_md>,<el_md>,<dwell_us>,<power_cW>   fire command
    -> A,<seq>                                          abort/stop (disarms)
    -> R,<seq>                                          (re)arm
    -> S                                               status request
    <- T,<seq>,<az_md>,<el_md>,<flags>,<heat_cP>,<shots>,<beam_ms>
    <- V,<reason>                                       veto notice
    <- W,<reason>                                       watchdog trip notice

The MCU ACKs nothing; the next T frame with matching seq is the ack.
Commands unacked for 3 status periods are retried by the slow loop,
and the MCU's watchdog fires if no valid frame arrives for 500 ms
(the laser then needs re-arm, it does not self-recover).
"""

import re
from typing import Optional, Tuple

from hardware.interfaces import FireCommand

CMD_FIRE = "F"
CMD_ABORT = "A"
CMD_ARM = "R"
CMD_STATUS = "S"
RSP_STATUS = "T"
RSP_VETO = "V"

_TIMEOUT_S = 0.5  # must match the MCU watchdog constant


def encode_fire(cmd: FireCommand) -> str:
    az_md = round(math_deg(cmd.az) * 1000)
    el_md = round(math_deg(cmd.el) * 1000)
    dwell_us = round(cmd.dwell_ms * 1000)
    power_cw = round(cmd.power_w * 100)
    return (f"{CMD_FIRE},{cmd.seq % 65536},{az_md},{el_md},"
            f"{dwell_us},{power_cw}\n")


def encode_abort(seq: int = 0) -> str:
    return f"{CMD_ABORT},{seq % 65536}\n"


def encode_arm(seq: int = 0) -> str:
    """(Re)arm the MCU. The core boots disarmed; abort/watchdog disarm.
    Arm discipline lives in the MCU so a crashed Pi can only leave the
    beam off."""
    return f"{CMD_ARM},{seq % 65536}\n"


def encode_status_request() -> str:
    return f"{CMD_STATUS}\n"


_FIRE_RE = re.compile(
    r"^F,(\d+),(-?\d+),(-?\d+),(\d+),(\d+)\n$")


def decode_fire(line: str, now: Optional[float] = None) -> Optional[FireCommand]:
    m = _FIRE_RE.match(line)
    if not m:
        return None
    seq, az_md, el_md, dwell_us, power_cw = (int(g) for g in m.groups())
    # sanity clamps: the MCU must never trust the wire blindly
    dwell_ms = min(dwell_us, 1_000_000) / 1000.0
    power_w = min(power_cw, 100_000) / 100.0
    import time as _t
    now = _t.monotonic() if now is None else now
    return FireCommand(az=rad(az_md / 1000.0), el=rad(el_md / 1000.0),
                       dwell_ms=dwell_ms, power_w=power_w, seq=seq,
                       expires_s=now + _TIMEOUT_S)


def math_deg(r: float) -> float:
    return r * 180.0 / 3.141592653589793


def rad(d: float) -> float:
    return d * 3.141592653589793 / 180.0

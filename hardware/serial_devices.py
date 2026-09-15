"""
Serial transport to the real MCU fast loop (ESP32/STM32 running the
Rust/C fire loop). Implements the same LaserDriver/Galvo/Interlock
interfaces as sim_devices so the slow-loop code is unchanged.

pyserial is only needed for real hardware; import is deferred so the
sim-only path stays dependency-free.
"""

import time
from typing import Optional

from hardware.interfaces import (
    FireCommand,
    InterlockState,
    TurretStatus,
    VetoReason,
    LaserDriver,
    Interlock,
)
from hardware import protocol


class SerialLink:
    """Framed UART connection with watchdog awareness."""

    def __init__(self, port: str, baud: int = 921600, timeout_s: float = 0.05):
        import serial  # deferred: pyserial
        self.ser = serial.Serial(port, baud, timeout=timeout_s)
        self._last_rx = 0.0

    def send(self, line: str) -> None:
        self.ser.write(line.encode("ascii"))

    def readline(self) -> str:
        raw = self.ser.readline()
        if raw:
            self._last_rx = time.monotonic()
        return raw.decode("ascii", errors="replace")

    def link_age_s(self) -> float:
        return time.monotonic() - self._last_rx


class McuLaserDriver(LaserDriver):
    """Fire commands are sent to the MCU, which owns dwell timing,
    cooldown, and the veto decision. status() is refreshed from T
    frames; refusals arrive as V frames."""

    def __init__(self, link: SerialLink):
        self.link = link
        self._seq = 0
        self._status = TurretStatus()
        self._last_veto: Optional[VetoReason] = None
        self.link.send(protocol.encode_arm(self._seq))   # arm on connect
        self._seq = (self._seq + 1) % 65536

    def fire(self, dwell_ms: float, power_w: float) -> None:
        cmd = FireCommand(az=self._status.az, el=self._status.el,
                          dwell_ms=dwell_ms, power_w=power_w,
                          seq=self._seq, expires_s=time.monotonic() + 0.5)
        self.link.send(protocol.encode_fire(cmd))
        self._seq = (self._seq + 1) % 65536

    def stop(self) -> None:
        self.link.send(protocol.encode_abort(self._seq))

    def status(self) -> TurretStatus:
        self.link.send(protocol.encode_status_request())
        self._pump()
        return self._status

    def _pump(self, max_lines: int = 16) -> None:
        for _ in range(max_lines):
            line = self.link.readline()
            if not line:
                break
            if line.startswith(protocol.RSP_STATUS + ","):
                self._parse_status(line)
            elif line.startswith(protocol.RSP_VETO + ","):
                reason = line.split(",")[1].strip()
                try:
                    self._last_veto = VetoReason(reason)
                except ValueError:
                    self._last_veto = VetoReason.WATCHDOG

    def _parse_status(self, line: str) -> None:
        # T,<seq>,<az_md>,<el_md>,<flags>,<heat_cP>,<shots>,<beam_ms>
        parts = line.strip().split(",")
        if len(parts) != 8:
            return
        self._status.last_cmd_seq = int(parts[1])
        self._status.az = protocol.rad(int(parts[2]) / 1000.0)
        self._status.el = protocol.rad(int(parts[3]) / 1000.0)
        flags = int(parts[4])
        self._status.firing = bool(flags & 0x2)
        self._status.heat = int(parts[5]) / 100.0
        self._status.shots = int(parts[6])
        self._status.beam_time_ms = float(parts[7])
        self._status.interlock = InterlockState(
            ok_to_fire=not self._status.firing and self._last_veto is None,
            veto=self._last_veto or VetoReason.NONE,
            armed=bool(flags & 0x1))


class McuInterlock(Interlock):
    """Veto zones are PUSHED to the MCU while disarmed; evaluation
    happens in the fast loop. evaluate() here is only a local echo for
    slow-loop logging -- the MCU's answer is authoritative."""

    def __init__(self, link: SerialLink):
        self.link = link
        self._zones = []

    def set_veto_zones(self, zones) -> None:
        self._zones = list(zones)
        # zone push frames: Z,<az_md>,<el_md>,<radius_md> per zone, then Z,END
        for (zaz, zel, rad_) in self._zones:
            self.link.send(f"Z,{round(protocol.math_deg(zaz) * 1000)},"
                           f"{round(protocol.math_deg(zel) * 1000)},"
                           f"{round(protocol.math_deg(rad_) * 1000)}\n")
        self.link.send("Z,END\n")

    def evaluate(self, cmd: FireCommand) -> InterlockState:
        # authoritative state comes from McuLaserDriver.status() V/T frames;
        # this echo exists so shared fire-control code has one interface.
        return InterlockState(ok_to_fire=True, veto=VetoReason.NONE,
                              armed=True, beam_time_budget_ms=0.0)

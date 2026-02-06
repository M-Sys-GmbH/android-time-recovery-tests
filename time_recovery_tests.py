#!/usr/bin/env python3
"""
Android Time Recovery Test Runner (ADB-based, no device-side scripts)

Correctness policy (hybrid):
1) If server delta exists (from connectivity probe), use it:
      abs(server_minus_device_sec) <= threshold_sec
2) Otherwise, fall back to host-year comparison with New Year tolerance:
      abs(device_year - host_year) <= year_tolerance

Notes:
- Setting device time usually requires root / working `su`.
- Disabling mobile data does NOT necessarily stop cellular time (NITZ).
  To avoid time snap-back during "offline" baselines, this script keeps auto_time=0
  until the moment you enable Wi‑Fi/mobile for correction phases.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple


# -----------------------------
# Configuration
# -----------------------------
PHASE_DESC: Dict[str, str] = {
    "TC1_BadTime_NoNetwork":
        "Simulate empty RTC (force year 2000) and disable Wi‑Fi & mobile; verify time stays wrong while offline.",
    "TC2_EnableWifi_WaitCorrection":
        "Enable Wi‑Fi and wait for time correction (NTP) once network becomes available/validated.",
    "TC3_EnableMobile_WaitCorrection":
        "Enable mobile data only and wait for time correction (NITZ and/or NTP over mobile).",
    "TC4_Reboot_NetOff_PersistCheck":
        "Reboot with networks off; verify corrected time persists (RTC rewritten).",
}

TRANSIENT_ADB_ERRORS = (
    "device not found",
    "no devices/emulators found",
    "device offline",
    "unauthorized",
    "error: closed",
)

BAD_TIME_DEFAULT = "010100002000.00"  # MMDDhhmmYYYY.SS -> 2000-01-01 00:00:00


# -----------------------------
# Logging
# -----------------------------
LOG = logging.getLogger("time_recovery")


def configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# -----------------------------
# Process execution helpers
# -----------------------------
@dataclass(frozen=True)
class CmdResult:
    rc: int
    out: str


class ProcessRunner:
    """Small wrapper around subprocess to capture output consistently."""
    @staticmethod
    def run(args: List[str], check: bool = False) -> CmdResult:
        LOG.debug("Running command: %s", " ".join(args))
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = (p.stdout or "").strip()
        if check and p.returncode != 0:
            raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(args)}\n{out}")
        return CmdResult(p.returncode, out)


# -----------------------------
# ADB wrapper
# -----------------------------
class Adb:
    def __init__(self, serial: str):
        self.serial = serial.strip()

    def _base(self) -> List[str]:
        return ["adb"] + (["-s", self.serial] if self.serial else [])

    def shell(self, cmd: str, retries: int = 30, delay_sec: float = 1.0) -> str:
        """
        Run an adb shell command with retry on transient ADB errors.
        Raises RuntimeError if transient errors persist to final attempt.
        """
        for attempt in range(1, retries + 1):
            res = ProcessRunner.run(self._base() + ["shell", cmd], check=False)
            lower = res.out.lower()
            if any(e in lower for e in TRANSIENT_ADB_ERRORS):
                LOG.debug("ADB transient error (attempt %d/%d): %s", attempt, retries, res.out)
                if attempt == retries:
                    raise RuntimeError(f"ADB shell failed after {retries} retries: {res.out}")
                time.sleep(delay_sec)
                continue
            return res.out
        raise RuntimeError("Unreachable: adb.shell retry loop")

    def reboot(self) -> None:
        ProcessRunner.run(self._base() + ["reboot"], check=False)

    def wait_for_device(self) -> None:
        ProcessRunner.run(self._base() + ["wait-for-device"], check=False)

    @staticmethod
    def detect_single_serial() -> str:
        res = ProcessRunner.run(["adb", "devices"], check=False)
        devs: List[str] = []
        for line in res.out.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices"):
                continue
            m = re.match(r"^(\S+)\s+device$", line)
            if m:
                devs.append(m.group(1))

        if len(devs) == 1:
            return devs[0]
        if len(devs) == 0:
            raise RuntimeError("No adb device in 'device' state found. Run: adb devices")
        raise RuntimeError(f"Multiple devices connected: {devs}. Please specify --serial.")


# -----------------------------
# Device actions
# -----------------------------
def year_from_date_line(date_line: str) -> Optional[int]:
    m = re.search(r"(\d{4})\s*$", date_line.strip())
    return int(m.group(1)) if m else None


def bad_year_from_bad_time(bad_time: str) -> int:
    return int(bad_time[8:12])


def set_auto_time(adb: Adb, on: bool) -> None:
    v = "1" if on else "0"
    adb.shell(f"settings put global auto_time {v}")
    adb.shell(f"settings put global auto_time_zone {v}")
    LOG.info("Set auto_time=%s auto_time_zone=%s", v, v)


def disable_networks(adb: Adb) -> None:
    adb.shell("svc wifi disable")
    adb.shell("svc data disable")
    LOG.info("Disabling Wi‑Fi and mobile data")


def enable_wifi(adb: Adb) -> None:
    adb.shell("svc wifi enable")
    LOG.info("Enabling Wi‑Fi")


def enable_mobile_data(adb: Adb) -> None:
    adb.shell("svc data enable")
    LOG.info("Enabling mobile data")


def force_bad_time(adb: Adb, bad_time: str, reenable_auto_time: bool) -> None:
    LOG.info("Forcing device time to bad value: %s", bad_time)
    set_auto_time(adb, False)
    bad_year = bad_year_from_bad_time(bad_time)

    # Use candidates compatible with your device (su root works, su -c does not)
    candidates = [
        f"su root date {bad_time}",
        f"su 0 date {bad_time}",
        f"date {bad_time}",
    ]

    last_out: str = ""
    applied = False

    for cmd in candidates:
        try:
            out = adb.shell(cmd)
            last_out = out.strip()
            device_date = adb.shell("date").strip()
            y = year_from_date_line(device_date)
            LOG.debug("After '%s' -> date='%s' year=%s", cmd, device_date, y)
            if y is not None and y == bad_year:
                applied = True
                LOG.info("Bad time applied successfully: %s", device_date)
                break
        except Exception as e:
            last_out = str(e).strip()

    if not applied:
        raise RuntimeError(
            "Failed to set bad time. Usually means root/`su` not available or denied.\n"
            f"Tried commands: {candidates}\n"
            f"Last output/error:\n{last_out}"
        )

    if reenable_auto_time:
        set_auto_time(adb, True)


# -----------------------------
# Parsing helpers (alarm/connectivity)
# -----------------------------
def parse_alarm(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "nowRTC_ms": None,
        "nowELAPSED_ms": None,
        "time_change_events": None,
        "ntp_poll_present": False,
        "ntp_poll_past_due": False,
    }
    m = re.search(r"nowRTC=(\d+)=.*?nowELAPSED=(\d+)", text, re.S)
    if m:
        out["nowRTC_ms"] = int(m.group(1))
        out["nowELAPSED_ms"] = int(m.group(2))

    m = re.search(r"Num time change events:\s+(\d+)", text)
    if m:
        out["time_change_events"] = int(m.group(1))

    if "NetworkTimeUpdateService.action.POLL" in text:
        out["ntp_poll_present"] = True
        if "Past-due" in text:
            out["ntp_poll_past_due"] = True
    return out


def try_parse_http_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    s = re.sub(r"\s+", " ", date_str.strip()).replace("\u200b", "").replace("\ufeff", "")
    try:
        dt = parsedate_to_datetime(s)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def parse_connectivity(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "active_default_none": ("Active default network: none" in text),
        "server_date_hdr": None,
        "received_ms": None,
        "server_minus_device_sec": None,
    }

    idx = text.rfind("PROBE_HTTP")
    while idx != -1:
        nxt = text.find("PROBE_", idx + 1)
        block = text[idx:] if nxt == -1 else text[idx:nxt]
        m_date = re.search(r"Date=\[([^\]]+)\]", block)
        m_recv = re.search(r"X-Android-Received-Millis=\[(\d+)\]", block)

        if m_date and m_recv:
            out["server_date_hdr"] = m_date.group(1).strip()
            out["received_ms"] = int(m_recv.group(1))
            server_dt = try_parse_http_date(out["server_date_hdr"])
            if server_dt:
                device_dt = datetime.fromtimestamp(out["received_ms"] / 1000.0, tz=timezone.utc)
                out["server_minus_device_sec"] = (server_dt - device_dt).total_seconds()
            break

        idx = text.rfind("PROBE_HTTP", 0, idx)
    return out


# -----------------------------
# DUT collection helpers
# -----------------------------
def _battery_status_text(code: Optional[int]) -> str:
    return {
        1: "Unknown",
        2: "Charging",
        3: "Discharging",
        4: "Not charging",
        5: "Full",
    }.get(code, f"Unknown({code})" if code is not None else "Unknown")


def _battery_health_text(code: Optional[int]) -> str:
    return {
        1: "Unknown",
        2: "Good",
        3: "Overheat",
        4: "Dead",
        5: "Over voltage",
        6: "Unspecified failure",
        7: "Cold",
    }.get(code, f"Unknown({code})" if code is not None else "Unknown")


def parse_battery_dumpsys(text: str) -> Dict[str, Any]:
    def pick_bool(key: str) -> Optional[bool]:
        m = re.search(rf"^\s*{re.escape(key)}:\s*(true|false)\s*$", text, re.M)
        return {"true": True, "false": False}.get(m.group(1)) if m else None

    def pick_int(key: str) -> Optional[int]:
        m = re.search(rf"^\s*{re.escape(key)}:\s*(-?\d+)\s*$", text, re.M)
        return int(m.group(1)) if m else None

    def pick_str(key: str) -> Optional[str]:
        m = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", text, re.M)
        return m.group(1).strip() if m else None

    status = pick_int("status")
    health = pick_int("health")
    temp_tenths_c = pick_int("temperature")
    temp_c = (temp_tenths_c / 10.0) if temp_tenths_c is not None else None

    return {
        "ac_powered": pick_bool("AC powered"),
        "usb_powered": pick_bool("USB powered"),
        "wireless_powered": pick_bool("Wireless powered"),
        "status_code": status,
        "status_text": _battery_status_text(status),
        "health_code": health,
        "health_text": _battery_health_text(health),
        "present": pick_bool("present"),
        "level": pick_int("level"),
        "scale": pick_int("scale"),
        "voltage_mv": pick_int("voltage"),
        "temperature_tenths_c": temp_tenths_c,
        "temperature_c": temp_c,
        "technology": pick_str("technology"),
        "charge_counter": pick_int("Charge counter"),
        "max_charging_current": pick_int("Max charging current"),
        "max_charging_voltage": pick_int("Max charging voltage"),
    }


def parse_default_network_line(connectivity_text: str) -> Dict[str, Any]:
    m = re.search(r"Active default network:\s*(.+)\s*$", connectivity_text, re.M)
    val = m.group(1).strip() if m else None
    return {
        "active_default_network_raw": val,
        "active_default_none": (val == "none"),
        "active_default_network_id": int(val) if (val and val.isdigit()) else None,
    }


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def adb_read_file_best_effort(adb: Adb, path: str, try_su: bool = True) -> Dict[str, Any]:
    """
    Try to read a sysfs file.
    - normal cat
    - if permission denied: try 'su root' then 'su 0'
    Never treats 'su:' errors as valid values.
    """
    def classify(out: str) -> Dict[str, Any]:
        o = (out or "").strip()
        lower = o.lower()

        if lower.startswith("su:") or "invalid uid/gid" in lower:
            return {"ok": False, "value": None, "error": o}

        if "permission denied" in lower:
            return {"ok": False, "value": None, "error": "permission denied"}

        if "no such file" in lower or "not found" in lower:
            return {"ok": False, "value": None, "error": "not found"}

        if o == "":
            return {"ok": False, "value": None, "error": "empty"}

        return {"ok": True, "value": o, "error": None}

    quoted = sh_quote(path)

    out = adb.shell(f"cat {quoted}").strip()
    res = classify(out)
    if res["ok"]:
        res["source"] = "cat"
        return res

    if res["error"] == "not found":
        res["source"] = "cat"
        return res

    if try_su and res["error"] == "permission denied":
        candidates = [
            f"su root cat {quoted}",
            f"su 0 cat {quoted}",
        ]
        last = out
        for cmd in candidates:
            out2 = adb.shell(cmd).strip()
            last = out2
            r2 = classify(out2)
            if r2["ok"]:
                r2["source"] = cmd.split()[:2]  # ['su','root'] or ['su','0']
                return r2
        return {"ok": False, "value": None, "error": "permission denied (even with su)", "source": "root", "last": last}

    res["source"] = "cat"
    return res


def adb_read_any_path_best_effort(adb: Adb, paths: List[str], try_su: bool = True) -> Dict[str, Any]:
    last = {"ok": False, "value": None, "error": "not found", "source": None, "path": None}
    for p in paths:
        r = adb_read_file_best_effort(adb, p, try_su=try_su)
        r["path"] = p
        if r.get("ok"):
            return r
        last = r
        last["path"] = p
    return last


def collect_dut_info(adb: Adb) -> Dict[str, Any]:
    props = {
        "manufacturer": adb.shell("getprop ro.product.manufacturer").strip(),
        "model": adb.shell("getprop ro.product.model").strip(),
        "device": adb.shell("getprop ro.product.device").strip(),
        "android_release": adb.shell("getprop ro.build.version.release").strip(),
        "sdk": adb.shell("getprop ro.build.version.sdk").strip(),
        "fingerprint": adb.shell("getprop ro.build.fingerprint").strip(),
        "security_patch": adb.shell("getprop ro.build.version.security_patch").strip(),
        "kernel": adb.shell("uname -a").strip(),
    }

    time_cfg = {
        "auto_time": adb.shell("settings get global auto_time").strip(),
        "auto_time_zone": adb.shell("settings get global auto_time_zone").strip(),
        "timezone": adb.shell("getprop persist.sys.timezone").strip(),
        "device_date_now": adb.shell("date").strip(),
    }

    battery = parse_battery_dumpsys(adb.shell("dumpsys battery"))
    default_net = parse_default_network_line(adb.shell("dumpsys connectivity"))
    alarm = parse_alarm(adb.shell("dumpsys alarm"))

    sysfs = {
        "cycle_count": adb_read_any_path_best_effort(adb, [
            "/sys/class/power_supply/battery/cycle_count",
            "/sys/class/power_supply/bms/cycle_count",
            "/sys/class/power_supply/bq27541-0/cycle_count",
        ]),
        "charge_full": adb_read_any_path_best_effort(adb, [
            "/sys/class/power_supply/battery/charge_full",
            "/sys/class/power_supply/bms/charge_full",
            "/sys/class/power_supply/bq27541-0/charge_full",
        ]),
        "charge_full_design": adb_read_any_path_best_effort(adb, [
            "/sys/class/power_supply/bms/charge_full_design",
            "/sys/class/power_supply/bq27541-0/charge_full_design",
            "/sys/class/power_supply/battery/charge_full_design",
        ]),
    }

    return {
        "props": props,
        "time": time_cfg,
        "battery": battery,
        "battery_sysfs": sysfs,
        "connectivity": default_net,
        "alarm": {
            "time_change_events": alarm.get("time_change_events"),
            "ntp_poll_present": alarm.get("ntp_poll_present"),
            "ntp_poll_past_due": alarm.get("ntp_poll_past_due"),
            "nowRTC_ms": alarm.get("nowRTC_ms"),
            "nowELAPSED_ms": alarm.get("nowELAPSED_ms"),
        },
    }


# -----------------------------
# Correctness logic
# -----------------------------
def is_time_correct(
    device_year: Optional[int],
    server_minus_device_sec: Optional[float],
    threshold_sec: int,
    year_tolerance: int
) -> bool:
    if device_year is None:
        return False

    host_year = datetime.now().year
    if abs(device_year - host_year) > year_tolerance:
        return False

    if server_minus_device_sec is not None:
        return abs(server_minus_device_sec) <= threshold_sec

    return True


# -----------------------------
# Boot wait helper
# -----------------------------
def wait_for_boot_completed(adb: Adb, timeout_sec: int = 240) -> bool:
    adb.wait_for_device()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        val = adb.shell("getprop sys.boot_completed", retries=5, delay_sec=1.0).strip()
        if val == "1":
            return True
        time.sleep(2)
    return False


# -----------------------------
# Reporting structures
# -----------------------------
@dataclass
class Sample:
    phase: str
    sample: int
    host_time: str
    device_date: str
    device_year: Optional[int]
    uptime: str
    auto_time: str
    auto_time_zone: str
    timezone: str
    nowRTC_ms: Optional[int]
    nowELAPSED_ms: Optional[int]
    time_change_events: Optional[int]
    ntp_poll_present: bool
    ntp_poll_past_due: bool
    active_default_none: bool
    server_date_hdr: Optional[str]
    received_ms: Optional[int]
    server_minus_device_sec: Optional[float]
    rtc_minus_elapsed_deviation_ms: Optional[int]
    notes: str


@dataclass
class PhaseResult:
    phase: str
    description: str
    passed: bool
    time_to_correct_seconds: Optional[int]
    start_host_time: str
    end_host_time: str
    notes: str


# -----------------------------
# Snapshot + sampling
# -----------------------------
def take_snapshot(adb: Adb,
                  phase: str,
                  sample_idx: int,
                  notes: str,
                  raw_dir: Path,
                  save_raw: bool,
                  prev_alarm: Optional[Dict[str, Any]]) -> Tuple[Sample, Dict[str, Any]]:
    host_time = datetime.now().isoformat(timespec="seconds")

    device_date = adb.shell("date").strip()
    device_year = year_from_date_line(device_date)

    uptime = adb.shell("uptime").strip()
    auto_time = adb.shell("settings get global auto_time").strip()
    auto_tz = adb.shell("settings get global auto_time_zone").strip()
    tz_prop = adb.shell("getprop persist.sys.timezone").strip()

    alarm_txt = adb.shell("dumpsys alarm")
    conn_txt = adb.shell("dumpsys connectivity")

    alarm = parse_alarm(alarm_txt)
    conn = parse_connectivity(conn_txt)

    deviation = None
    if (
        prev_alarm
        and alarm["nowRTC_ms"] and alarm["nowELAPSED_ms"]
        and prev_alarm["nowRTC_ms"] and prev_alarm["nowELAPSED_ms"]
    ):
        drtc = alarm["nowRTC_ms"] - prev_alarm["nowRTC_ms"]
        delp = alarm["nowELAPSED_ms"] - prev_alarm["nowELAPSED_ms"]
        deviation = drtc - delp

    if save_raw:
        safe_phase = re.sub(r"[^A-Za-z0-9_]+", "_", phase)
        stamp = f"{safe_phase}_{sample_idx:03d}"
        (raw_dir / f"{stamp}-alarm.txt").write_text(alarm_txt, encoding="utf-8")
        (raw_dir / f"{stamp}-connectivity.txt").write_text(conn_txt, encoding="utf-8")

    sample = Sample(
        phase=phase,
        sample=sample_idx,
        host_time=host_time,
        device_date=device_date,
        device_year=device_year,
        uptime=uptime,
        auto_time=auto_time,
        auto_time_zone=auto_tz,
        timezone=tz_prop,
        nowRTC_ms=alarm["nowRTC_ms"],
        nowELAPSED_ms=alarm["nowELAPSED_ms"],
        time_change_events=alarm["time_change_events"],
        ntp_poll_present=alarm["ntp_poll_present"],
        ntp_poll_past_due=alarm["ntp_poll_past_due"],
        active_default_none=conn["active_default_none"],
        server_date_hdr=conn["server_date_hdr"],
        received_ms=conn["received_ms"],
        server_minus_device_sec=conn["server_minus_device_sec"],
        rtc_minus_elapsed_deviation_ms=deviation,
        notes=notes,
    )

    LOG.info(
        "[%s] #%d date='%s' default_none=%s server_delta_s=%s dev_ms=%s",
        phase, sample_idx, device_date, conn["active_default_none"],
        conn["server_minus_device_sec"], deviation
    )
    return sample, alarm


def wait_for_correction(adb: Adb,
                        phase: str,
                        threshold_sec: int,
                        year_tolerance: int,
                        timeout_sec: int,
                        interval_sec: int,
                        raw_dir: Path,
                        save_raw: bool,
                        prev_alarm: Optional[Dict[str, Any]],
                        samples_out: List[Sample]) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    start = time.time()
    deadline = start + timeout_sec
    sample_idx = 1

    while time.time() < deadline:
        sample, prev_alarm = take_snapshot(adb, phase, sample_idx, "poll", raw_dir, save_raw, prev_alarm)
        samples_out.append(sample)

        server_delta = sample.server_minus_device_sec
        if sample.active_default_none:
            server_delta = None

        if is_time_correct(sample.device_year, server_delta, threshold_sec, year_tolerance):
            elapsed = int(time.time() - start)
            LOG.info(">>> TIME CORRECTED in %s after %ds", phase, elapsed)
            return True, elapsed, prev_alarm

        time.sleep(interval_sec)
        sample_idx += 1

    LOG.warning(">>> TIME NOT corrected in %s within %ds", phase, timeout_sec)
    return False, None, prev_alarm


def sample_baseline(adb: Adb,
                    phase: str,
                    samples_baseline: int,
                    interval_sec: int,
                    raw_dir: Path,
                    save_raw: bool,
                    prev_alarm: Optional[Dict[str, Any]],
                    samples_out: List[Sample],
                    predicate_fail: Callable[[Sample], bool],
                    note: str = "baseline") -> Tuple[bool, Optional[Dict[str, Any]]]:
    passed = True
    for i in range(1, samples_baseline + 1):
        s, prev_alarm = take_snapshot(adb, phase, i, note, raw_dir, save_raw, prev_alarm)
        samples_out.append(s)
        if predicate_fail(s):
            passed = False
        time.sleep(interval_sec)
    return passed, prev_alarm


# -----------------------------
# HTML report generation
# -----------------------------
def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _format_server_offset_cell(s: Sample, stale_threshold_sec: int = 86400) -> str:
    if s.server_minus_device_sec is None:
        return "<td></td>"

    offset = float(s.server_minus_device_sec)
    is_stale = s.active_default_none or abs(offset) > stale_threshold_sec

    if is_stale:
        return (
            "<td title='Offset likely stale (offline or unusually large).'>"
            f"{_esc(offset)} <span style='color:#888'>(stale)</span>"
            "</td>"
        )

    return f"<td>{_esc(offset)}</td>"


def _note_meaning(sample: Sample, run_info: Dict[str, Any]) -> str:
    year_tol = int(run_info.get("parameters", {}).get("year_tolerance", 1))
    host_year = datetime.now().year
    looks_correct_by_year = (
        sample.device_year is not None and abs(sample.device_year - host_year) <= year_tol
    )
    tag = (sample.notes or "").strip().lower()

    if tag == "baseline":
        return "⛔ Offline baseline — time must stay wrong"
    if tag == "prep":
        return "⛔ Offline prep — time must stay wrong"
    if tag == "persist":
        return "🔁 Post‑reboot — verify time persists offline"

    if tag == "poll":
        if looks_correct_by_year:
            if sample.server_minus_device_sec is not None:
                return "✅ Time corrected (server‑aligned)"
            return "✅ Time corrected"
        return "⏳ Waiting for time correction"

    return sample.notes


def _render_run_info(run_info: Dict[str, Any]) -> str:
    pretty = _esc(json.dumps(run_info, indent=2))
    return (
        "<h2>Run Info</h2>"
        "<details>"
        "<summary style='cursor:pointer; font-weight:600;'>Click to expand</summary>"
        f"<pre>{pretty}</pre>"
        "</details>"
    )


def format_uah_as_mah(value_str: Optional[str]) -> str:
    if not value_str:
        return ""
    try:
        uah = int(value_str.strip())
        mah = uah / 1000.0
        return f"{uah} µAh (~{mah:.0f} mAh)"
    except Exception:
        return value_str


def _render_dut_info(dut: Dict[str, Any]) -> str:
    if not dut:
        return "<h2>Device Under Test (DUT)</h2><p><i>No DUT information captured.</i></p>"

    props = dut.get("props", {})
    time_cfg = dut.get("time", {})
    bat = dut.get("battery", {})
    net = dut.get("connectivity", {})
    alarm = dut.get("alarm", {})
    sysfs = dut.get("battery_sysfs", {})

    def yesno(v: Any) -> str:
        if v is True:
            return "Yes"
        if v is False:
            return "No"
        return ""

    def sysfs_field(name: str, format_as_uah: bool = False) -> str:
        item = sysfs.get(name) or {}
        if item.get("ok"):
            val = item.get("value")
            if format_as_uah:
                val = format_uah_as_mah(val)
            extra = ""
            if item.get("path"):
                extra = f" <span style='color:#888'>({item.get('path')})</span>"
            return _esc(val) + extra
        err = item.get("error")
        return _esc(f"Not available ({err})") if err else "Not available"

    powered = []
    if bat.get("ac_powered"): powered.append("AC")
    if bat.get("usb_powered"): powered.append("USB")
    if bat.get("wireless_powered"): powered.append("Wireless")
    powered_str = ", ".join(powered) if powered else "No"

    default_net_raw = net.get("active_default_network_raw")
    if default_net_raw is None:
        default_net_disp = "Unknown"
    elif net.get("active_default_none"):
        default_net_disp = "none (offline)"
    else:
        default_net_disp = f"{default_net_raw} (online)"

    temp_c = bat.get("temperature_c")
    temp_disp = f"{temp_c:.1f} °C" if isinstance(temp_c, (int, float)) else ""

    rows: List[str] = []
    rows.append("<h2>Device Under Test (DUT)</h2>")

    rows.append("<h3>Device & Build</h3>")
    rows.append("<table border='1' cellpadding='6' cellspacing='0'>")
    rows.append("<tr><th>Field</th><th>Value</th></tr>")
    rows.append(f"<tr><td>Manufacturer</td><td>{_esc(props.get('manufacturer'))}</td></tr>")
    rows.append(f"<tr><td>Model</td><td>{_esc(props.get('model'))}</td></tr>")
    rows.append(f"<tr><td>Device</td><td>{_esc(props.get('device'))}</td></tr>")
    rows.append(f"<tr><td>Android</td><td>{_esc(props.get('android_release'))} (SDK {_esc(props.get('sdk'))})</td></tr>")
    rows.append(f"<tr><td>Security patch</td><td>{_esc(props.get('security_patch'))}</td></tr>")
    rows.append(f"<tr><td>Build fingerprint</td><td><code>{_esc(props.get('fingerprint'))}</code></td></tr>")
    rows.append(f"<tr><td>Kernel</td><td><code>{_esc(props.get('kernel'))}</code></td></tr>")
    rows.append("</table>")

    rows.append("<h3>Time Configuration</h3>")
    rows.append("<table border='1' cellpadding='6' cellspacing='0'>")
    rows.append("<tr><th>Field</th><th>Value</th></tr>")
    rows.append(f"<tr><td>Auto time</td><td>{_esc(time_cfg.get('auto_time'))}</td></tr>")
    rows.append(f"<tr><td>Auto time zone</td><td>{_esc(time_cfg.get('auto_time_zone'))}</td></tr>")
    rows.append(f"<tr><td>Time zone</td><td>{_esc(time_cfg.get('timezone'))}</td></tr>")
    rows.append(f"<tr><td>Device date (start)</td><td>{_esc(time_cfg.get('device_date_now'))}</td></tr>")
    rows.append("</table>")

    rows.append("<h3>Battery</h3>")
    rows.append("<table border='1' cellpadding='6' cellspacing='0'>")
    rows.append("<tr><th>Field</th><th>Value</th></tr>")
    rows.append(f"<tr><td>Level</td><td>{_esc(bat.get('level'))}%</td></tr>")
    rows.append(f"<tr><td>Status</td><td>{_esc(bat.get('status_text'))} ({_esc(bat.get('status_code'))})</td></tr>")
    rows.append(f"<tr><td>Health</td><td>{_esc(bat.get('health_text'))} ({_esc(bat.get('health_code'))})</td></tr>")
    rows.append(f"<tr><td>Temperature</td><td>{_esc(temp_disp)}</td></tr>")
    rows.append(f"<tr><td>Voltage</td><td>{_esc(bat.get('voltage_mv'))} mV</td></tr>")
    rows.append(f"<tr><td>Powered by</td><td>{_esc(powered_str)}</td></tr>")
    rows.append(f"<tr><td>Technology</td><td>{_esc(bat.get('technology'))}</td></tr>")
    rows.append(f"<tr><td>Cycle count</td><td>{sysfs_field('cycle_count')}</td></tr>")
    rows.append(f"<tr><td>Charge full</td><td>{sysfs_field('charge_full', format_as_uah=True)}</td></tr>")
    rows.append(f"<tr><td>Charge full design</td><td>{sysfs_field('charge_full_design', format_as_uah=True)}</td></tr>")
    rows.append("</table>")

    rows.append("<h3>Network & Time Observations</h3>")
    rows.append("<table border='1' cellpadding='6' cellspacing='0'>")
    rows.append("<tr><th>Field</th><th>Value</th></tr>")
    rows.append(f"<tr><td>Active default network</td><td>{_esc(default_net_disp)}</td></tr>")
    rows.append(f"<tr><td>Time change events</td><td>{_esc(alarm.get('time_change_events'))}</td></tr>")
    rows.append(f"<tr><td>NTP poll present</td><td>{_esc(yesno(alarm.get('ntp_poll_present')))}</td></tr>")
    rows.append(f"<tr><td>NTP poll past-due</td><td>{_esc(yesno(alarm.get('ntp_poll_past_due')))}</td></tr>")
    if "rtc_persistence_observed" in dut:
        rows.append(f"<tr><td>RTC persistence observed (TC4)</td><td>{_esc('Yes' if dut.get('rtc_persistence_observed') else 'No')}</td></tr>")
    rows.append("</table>")

    return "\n".join(rows)


def _render_glossary() -> str:
    items = [
        ("Host time",
         "Timestamp on the PC when the sample was taken (ISO format; easy to sort)."),
        ("Device date",
         "Raw output of `adb shell date` from the device (format depends on device locale/timezone)."),
        ("Network",
         "Offline means 'Active default network: none'. Online means a default network exists."),
        ("Server time offset (s)",
         "Estimated server_time − device_time from Android HTTP probe. Near 0 means device time is close to server time."),
        ("Wall‑clock jump (ms)",
         "Difference between wall clock progress (RTC) and monotonic time (ELAPSED) since the previous sample. Large values usually mean time correction."),
        ("Meaning",
         "Human-friendly explanation of why/when the sample was taken."),
    ]
    li = "\n".join(f"<li><b>{_esc(k)}:</b> {_esc(v)}</li>" for k, v in items)
    return "<h2>Glossary</h2>\n<ul>" + li + "</ul>"


def _render_summary_table(phase_results: List[PhaseResult]) -> str:
    rows = [
        "<h2>Summary</h2>",
        "<table border='1' cellpadding='6' cellspacing='0'>",
        "<tr><th>Phase</th><th>Description</th><th>Result</th><th>Time to correct (s)</th><th>Notes</th></tr>",
    ]
    for r in phase_results:
        rows.append(
            "<tr>"
            f"<td>{_esc(r.phase)}</td>"
            f"<td>{_esc(r.description)}</td>"
            f"<td><b>{'PASS' if r.passed else 'FAIL'}</b></td>"
            f"<td>{_esc(r.time_to_correct_seconds)}</td>"
            f"<td>{_esc(r.notes)}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _render_phase_details(run_info: Dict[str, Any], samples: List[Sample], last_n_per_phase: int) -> str:
    # Only render phases that were executed (from run_info parameters).
    selected = run_info.get("parameters", {}).get("selected_phases", [])
    if not selected:
        selected = list(PHASE_DESC.keys())

    by_phase: Dict[str, List[Sample]] = {}
    for s in samples:
        by_phase.setdefault(s.phase, []).append(s)

    out: List[str] = ["<h2>Details (last samples per phase)</h2>"]
    for phase in selected:
        desc = PHASE_DESC.get(phase, "")
        out.append(f"<h3>{_esc(phase)}</h3>")
        out.append(f"<p>{_esc(desc)}</p>")

        phase_samples = by_phase.get(phase, [])
        if not phase_samples:
            out.append("<p><i>No samples for this phase.</i></p>")
            continue

        tail = phase_samples[-last_n_per_phase:]

        out.append("<table border='1' cellpadding='6' cellspacing='0'>")
        out.append(
            "<tr>"
            "<th title='Sample index within this phase.'>#</th>"
            "<th title='Timestamp on the host PC when the sample was taken (ISO 8601).'>Host time</th>"
            "<th title='Raw output of `adb shell date` from the device.'>Device date</th>"
            "<th title='Year parsed from Device date.'>Year</th>"
            "<th title=\"Offline means 'Active default network: none'. Online means a default network exists.\">Network</th>"
            "<th title='Estimated server_time − device_time from HTTP probe. Labeled (stale) when offline or unusually large.'>Server time offset (s)</th>"
            "<th title='RTC minus ELAPSED deviation between samples.'>Wall‑clock jump (ms)</th>"
            "<th title='Tester-friendly meaning of this snapshot.'>Meaning</th>"
            "</tr>"
        )
        for s in tail:
            net_state = "Offline" if s.active_default_none else "Online"
            out.append(
                "<tr>"
                f"<td>{_esc(s.sample)}</td>"
                f"<td>{_esc(s.host_time)}</td>"
                f"<td>{_esc(s.device_date)}</td>"
                f"<td>{_esc(s.device_year)}</td>"
                f"<td title='Derived from dumpsys connectivity default network state.'>{_esc(net_state)}</td>"
                + _format_server_offset_cell(s) +
                f"<td>{_esc(s.rtc_minus_elapsed_deviation_ms)}</td>"
                f"<td title='Meaning of this snapshot within the test flow.'>{_esc(_note_meaning(s, run_info))}</td>"
                "</tr>"
            )
        out.append("</table>")
    return "\n".join(out)


def render_html_report(run_info: Dict[str, Any],
                       phase_results: List[PhaseResult],
                       samples: List[Sample],
                       out_path: Path,
                       last_n_per_phase: int = 15) -> None:
    doc = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Android Time Recovery Test Report</title>",
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;} pre{background:#f6f6f6;padding:10px;} "
        "table{border-collapse:collapse;} th{background:#efefef;}"
        "</style>",
        "</head><body>",
        "<h1>Android Time Recovery Test Report</h1>",
        _render_run_info(run_info),
        _render_dut_info(run_info.get("dut", {})),
        _render_glossary(),
        _render_summary_table(phase_results),
        _render_phase_details(run_info, samples, last_n_per_phase),
        "</body></html>",
    ]
    out_path.write_text("\n".join(doc), encoding="utf-8")


# -----------------------------
# Phase runners
# -----------------------------
def phase_tc1(adb: Adb, args: argparse.Namespace, raw_dir: Path,
              prev_alarm: Optional[Dict[str, Any]], all_samples: List[Sample]) -> Tuple[PhaseResult, Optional[Dict[str, Any]]]:
    phase = "TC1_BadTime_NoNetwork"

    disable_networks(adb)
    force_bad_time(adb, args.bad_time, reenable_auto_time=False)

    bad_year = bad_year_from_bad_time(args.bad_time)

    def fail_if_time_changed(s: Sample) -> bool:
        return s.device_year is not None and s.device_year != bad_year

    start_host = datetime.now().isoformat(timespec="seconds")
    passed, prev_alarm = sample_baseline(
        adb=adb,
        phase=phase,
        samples_baseline=args.samples_baseline,
        interval_sec=args.interval_sec,
        raw_dir=raw_dir,
        save_raw=args.save_raw,
        prev_alarm=prev_alarm,
        samples_out=all_samples,
        predicate_fail=fail_if_time_changed,
        note="baseline"
    )

    end_host = datetime.now().isoformat(timespec="seconds")
    result = PhaseResult(
        phase=phase,
        description=PHASE_DESC[phase],
        passed=passed,
        time_to_correct_seconds=None,
        start_host_time=start_host,
        end_host_time=end_host,
        notes="PASS means time stayed wrong while offline (expected).",
    )
    return result, prev_alarm


def phase_tc2(adb: Adb, args: argparse.Namespace, raw_dir: Path,
              prev_alarm: Optional[Dict[str, Any]], all_samples: List[Sample]) -> Tuple[PhaseResult, Optional[Dict[str, Any]]]:
    phase = "TC2_EnableWifi_WaitCorrection"

    disable_networks(adb)
    force_bad_time(adb, args.bad_time, reenable_auto_time=False)
    enable_wifi(adb)
    set_auto_time(adb, True)

    start_host = datetime.now().isoformat(timespec="seconds")
    ok, elapsed, prev_alarm = wait_for_correction(
        adb=adb,
        phase=phase,
        threshold_sec=args.threshold_sec,
        year_tolerance=args.year_tolerance,
        timeout_sec=args.timeout_sec,
        interval_sec=args.interval_sec,
        raw_dir=raw_dir,
        save_raw=args.save_raw,
        prev_alarm=prev_alarm,
        samples_out=all_samples,
    )

    end_host = datetime.now().isoformat(timespec="seconds")
    result = PhaseResult(
        phase=phase,
        description=PHASE_DESC[phase],
        passed=ok,
        time_to_correct_seconds=elapsed,
        start_host_time=start_host,
        end_host_time=end_host,
        notes="PASS means time aligned to server threshold (preferred) or host-year fallback when probe missing.",
    )
    return result, prev_alarm


def phase_tc3(adb: Adb, args: argparse.Namespace, raw_dir: Path,
              prev_alarm: Optional[Dict[str, Any]], all_samples: List[Sample]) -> Tuple[PhaseResult, Optional[Dict[str, Any]]]:
    phase_prep = "TC3_PreMobile_BadTime_NoNetwork"
    phase = "TC3_EnableMobile_WaitCorrection"

    disable_networks(adb)
    force_bad_time(adb, args.bad_time, reenable_auto_time=False)

    prep_count = max(1, args.samples_baseline // 3)
    for i in range(1, prep_count + 1):
        s, prev_alarm = take_snapshot(adb, phase_prep, i, "prep", raw_dir, args.save_raw, prev_alarm)
        all_samples.append(s)
        time.sleep(args.interval_sec)

    set_auto_time(adb, True)
    enable_mobile_data(adb)

    start_host = datetime.now().isoformat(timespec="seconds")
    ok, elapsed, prev_alarm = wait_for_correction(
        adb=adb,
        phase=phase,
        threshold_sec=args.threshold_sec,
        year_tolerance=args.year_tolerance,
        timeout_sec=args.timeout_sec,
        interval_sec=args.interval_sec,
        raw_dir=raw_dir,
        save_raw=args.save_raw,
        prev_alarm=prev_alarm,
        samples_out=all_samples,
    )

    end_host = datetime.now().isoformat(timespec="seconds")
    result = PhaseResult(
        phase=phase,
        description=PHASE_DESC[phase],
        passed=ok,
        time_to_correct_seconds=elapsed,
        start_host_time=start_host,
        end_host_time=end_host,
        notes="PASS means time aligned to server threshold (preferred) or host-year fallback when probe missing.",
    )
    return result, prev_alarm


def phase_tc4(adb: Adb, args: argparse.Namespace, raw_dir: Path,
              prev_alarm: Optional[Dict[str, Any]], all_samples: List[Sample]) -> Tuple[PhaseResult, Optional[Dict[str, Any]]]:
    phase = "TC4_Reboot_NetOff_PersistCheck"

    disable_networks(adb)

    start_host = datetime.now().isoformat(timespec="seconds")
    adb.reboot()
    time.sleep(5)

    boot_ok = wait_for_boot_completed(adb, timeout_sec=args.boot_timeout_sec)
    if not boot_ok:
        LOG.warning("sys.boot_completed not reached within %ds; continuing anyway.", args.boot_timeout_sec)
    time.sleep(5)

    disable_networks(adb)

    def fail_if_time_looks_bad(s: Sample) -> bool:
        if s.device_year is None:
            return True
        host_year = datetime.now().year
        return abs(s.device_year - host_year) > args.year_tolerance

    passed, prev_alarm = sample_baseline(
        adb=adb,
        phase=phase,
        samples_baseline=args.samples_baseline,
        interval_sec=args.interval_sec,
        raw_dir=raw_dir,
        save_raw=args.save_raw,
        prev_alarm=prev_alarm,
        samples_out=all_samples,
        predicate_fail=fail_if_time_looks_bad,
        note="persist"
    )

    end_host = datetime.now().isoformat(timespec="seconds")
    result = PhaseResult(
        phase=phase,
        description=PHASE_DESC[phase],
        passed=passed,
        time_to_correct_seconds=None,
        start_host_time=start_host,
        end_host_time=end_host,
        notes="PASS means corrected time persisted across reboot while offline (checked via host-year tolerance).",
    )
    return result, prev_alarm


# -----------------------------
# Main / CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Android time recovery tests via ADB (JSON + HTML report, no charts, no device scripts)"
    )
    ap.add_argument("--serial", default="", help="ADB device serial (optional if one device connected)")

    ap.add_argument("--samples-baseline", type=int, default=5,
                    help="Number of baseline samples in phases that sample N times (default: 5)")
    ap.add_argument("--interval-sec", type=int, default=1, help="Seconds between samples (default: 1)")
    ap.add_argument("--timeout-sec", type=int, default=60, help="Max seconds to wait for time correction phases (default: 60)")
    ap.add_argument("--threshold-sec", type=int, default=300,
                    help="Max allowed server-device delta (seconds) when server date is available (default: 300)")
    ap.add_argument("--year-tolerance", type=int, default=1,
                    help="Fallback year tolerance when server delta is unavailable (default: 1, for New Year)")
    ap.add_argument("--boot-timeout-sec", type=int, default=240,
                    help="Max seconds to wait for sys.boot_completed after reboot")
    ap.add_argument("--bad-time", default=BAD_TIME_DEFAULT,
                    help=f"Bad time used in 'date' format MMDDhhmmYYYY.SS (default {BAD_TIME_DEFAULT})")

    ap.add_argument("--no-wifi", dest="run_wifi", action="store_false", default=True,
                    help="Skip Wi‑Fi correction test (TC2)")
    ap.add_argument("--no-mobile", dest="run_mobile", action="store_false", default=True,
                    help="Skip mobile correction test (TC3)")
    ap.add_argument("--no-reboot", dest="run_reboot", action="store_false", default=True,
                    help="Skip reboot persistence test (TC4)")
    ap.add_argument("--no-raw", dest="save_raw", action="store_false", default=True,
                    help="Do not store raw dumpsys outputs")

    ap.add_argument("--html-last-n", type=int, default=15, help="How many last samples per phase to show in HTML")
    ap.add_argument("--phases", default="", help="Comma-separated phases to run (e.g. TC1,TC2). Empty means run default set.")
    ap.add_argument("--list-phases", action="store_true", help="List available phases and exit")
    ap.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    ap.add_argument("--quiet", action="store_true", help="Only show warnings/errors")

    return ap.parse_args()


def resolve_phase_selection(args: argparse.Namespace) -> List[str]:
    all_phases = [
        "TC1_BadTime_NoNetwork",
        "TC2_EnableWifi_WaitCorrection",
        "TC3_EnableMobile_WaitCorrection",
        "TC4_Reboot_NetOff_PersistCheck",
    ]

    if args.phases.strip():
        requested = [p.strip() for p in args.phases.split(",") if p.strip()]
        normalized: List[str] = []
        for p in requested:
            if p in ("TC1", "TC2", "TC3", "TC4"):
                mapping = {
                    "TC1": "TC1_BadTime_NoNetwork",
                    "TC2": "TC2_EnableWifi_WaitCorrection",
                    "TC3": "TC3_EnableMobile_WaitCorrection",
                    "TC4": "TC4_Reboot_NetOff_PersistCheck",
                }
                normalized.append(mapping[p])
            else:
                normalized.append(p)

        unknown = [p for p in normalized if p not in all_phases]
        if unknown:
            raise RuntimeError(f"Unknown phases requested: {unknown}. Use --list-phases.")
        return normalized

    selected = ["TC1_BadTime_NoNetwork"]
    if args.run_wifi:
        selected.append("TC2_EnableWifi_WaitCorrection")
    if args.run_mobile:
        selected.append("TC3_EnableMobile_WaitCorrection")
    if args.run_reboot:
        selected.append("TC4_Reboot_NetOff_PersistCheck")
    return selected


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose, args.quiet)

    if args.list_phases:
        print("Available phases:")
        for k in [
            "TC1_BadTime_NoNetwork",
            "TC2_EnableWifi_WaitCorrection",
            "TC3_EnableMobile_WaitCorrection",
            "TC4_Reboot_NetOff_PersistCheck",
        ]:
            print(f"- {k}: {PHASE_DESC[k]}")
        return

    base_dir = Path(__file__).resolve().parent
    runs_dir = base_dir / "runs"

    serial = args.serial.strip() or Adb.detect_single_serial()
    adb = Adb(serial)

    selected_phases = resolve_phase_selection(args)

    LOG.info("=== TEST PLAN ===")
    for k in selected_phases:
        LOG.info("- %s: %s", k, PHASE_DESC.get(k, "(no description)"))

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = runs_dir / run_stamp
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    report_json = out_dir / "report.json"
    report_html = out_dir / "report.html"

    LOG.info("Run output: %s", out_dir)
    LOG.info("Report JSON: %s", report_json)
    LOG.info("Report HTML: %s\n", report_html)

    LOG.info("Collecting DUT info (device/build/battery/time)...")
    dut_info = collect_dut_info(adb)

    run_info = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "serial": serial,
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "parameters": {
            "samples_baseline": args.samples_baseline,
            "interval_sec": args.interval_sec,
            "timeout_sec": args.timeout_sec,
            "threshold_sec": args.threshold_sec,
            "year_tolerance": args.year_tolerance,
            "boot_timeout_sec": args.boot_timeout_sec,
            "bad_time": args.bad_time,
            "run_wifi": args.run_wifi,
            "run_mobile": args.run_mobile,
            "run_reboot": args.run_reboot,
            "save_raw": args.save_raw,
            "selected_phases": selected_phases,
        },
        "dut": dut_info,
        "output_paths": {
            "report_json": str(report_json),
            "report_html": str(report_html),
            "raw_dir": str(raw_dir) if args.save_raw else None,
        },
    }

    all_samples: List[Sample] = []
    phase_results: List[PhaseResult] = []
    prev_alarm: Optional[Dict[str, Any]] = None

    runners: Dict[str, Callable[..., Tuple[PhaseResult, Optional[Dict[str, Any]]]]] = {
        "TC1_BadTime_NoNetwork": phase_tc1,
        "TC2_EnableWifi_WaitCorrection": phase_tc2,
        "TC3_EnableMobile_WaitCorrection": phase_tc3,
        "TC4_Reboot_NetOff_PersistCheck": phase_tc4,
    }

    tc4_pass_value: Optional[bool] = None

    for phase_name in selected_phases:
        LOG.info("\n=== %s ===", phase_name)
        LOG.info("%s", PHASE_DESC.get(phase_name, ""))

        result, prev_alarm = runners[phase_name](adb, args, raw_dir, prev_alarm, all_samples)
        phase_results.append(result)

        if phase_name == "TC4_Reboot_NetOff_PersistCheck":
            tc4_pass_value = bool(result.passed)

    if tc4_pass_value is not None:
        run_info["dut"]["rtc_persistence_observed"] = tc4_pass_value

    report = {
        "run": run_info,
        "phases": [asdict(r) for r in phase_results],
        "samples": [asdict(s) for s in all_samples],
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    render_html_report(run_info, phase_results, all_samples, report_html, last_n_per_phase=args.html_last_n)

    LOG.info("\n=== SUMMARY ===")
    for r in phase_results:
        metric = f" time_to_correct={r.time_to_correct_seconds}s" if r.time_to_correct_seconds is not None else ""
        LOG.info("- %s: %s%s", r.phase, "PASS" if r.passed else "FAIL", metric)
        LOG.info("  %s", r.description)
        LOG.info("  %s\n", r.notes)

    # Restore connectivity for convenience after tests
    enable_mobile_data(adb)
    enable_wifi(adb)

    LOG.info("\n=== DONE ===")
    LOG.info("Results: %s", out_dir)
    LOG.info("Report JSON: %s", report_json)
    LOG.info("Report HTML: %s", report_html)


if __name__ == "__main__":
    main()
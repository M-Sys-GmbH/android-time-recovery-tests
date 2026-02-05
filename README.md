# Android Time Recovery Test Runner (ADB-based)

This repository contains a single Python script that automates a time-recovery test plan on an Android device via **ADB**.

It is designed for **manual execution** by testers and produces:
- a machine-readable **JSON report**
- a human-readable **HTML report** (with a glossary and tooltips)
- optional raw dumpsys output for debugging

## What the script tests

The script runs the following phases (test cases):

- **TC1_BadTime_NoNetwork**  
  Force a “bad” wall-clock time (default: Jan 1, 2000) and disable networks.  
  Verify the time stays wrong while offline.

- **TC2_EnableWifi_WaitCorrection**  
  Enable Wi‑Fi and wait for time correction.

- **TC3_EnableMobile_WaitCorrection**  
  Enable mobile data only and wait for time correction.

- **TC4_Reboot_NetOff_PersistCheck**  
  Reboot with networks off and verify corrected time persists (RTC rewritten).

## Correctness policy (hybrid)

The script determines “time is correct” using this policy:

1. **If a server time offset is available** (from Android connectivity HTTP probe):
   - PASS when `abs(server_minus_device_sec) <= threshold_sec`
2. **Otherwise** (if no server offset is available):
   - PASS when device year is within `host_year ± year_tolerance` (New Year tolerance)

Additionally, the script includes a safety gate so that **obviously wrong years** (e.g., 2000) can never be treated as correct.

## Prerequisites

### Host requirements
- Python 3.9+ (works with Python 3.13)
- `adb` in your PATH (Android Platform Tools installed)
- A USB connection or network ADB access to the device

Verify:
```bash
adb version
adb devices
````

### Device requirements

*   USB debugging enabled
*   You must be able to set the device time to a past value. This usually requires:
    *   root access and a working `su`, OR
    *   privileged shell access on engineering builds

If you cannot set the device time, TC1/TC3 cannot work.

## Setup

Place the script in a folder. The script will create output under:

    runs/<YYYY-MM-DD_HH-MM-SS>/
      report.json
      report.html
      raw/                (optional, if --no-raw is NOT used)

No additional device scripts are required.

## Usage

Basic run (auto-detect one connected device):

```bash
python3 time_recovery_tests.py
```

If more than one device is connected:

```bash
python3 time_recovery_tests.py --serial <DEVICE_SERIAL>
```

List phases and exit:

```bash
python3 time_recovery_tests.py --list-phases
```

Run only selected phases:

```bash
python3 time_recovery_tests.py --phases TC1,TC2
# or full names:
python3 time_recovery_tests.py --phases TC1_BadTime_NoNetwork,TC2_EnableWifi_WaitCorrection
```

Enable verbose debug logs:

```bash
python3 time_recovery_tests.py --verbose
```

Show only warnings/errors:

```bash
python3 time_recovery_tests.py --quiet
```

## Common options

*   `--samples-baseline`  
    Number of samples taken in baseline/persistence phases (default: 5).

*   `--interval-sec`  
    Seconds between samples (default: 1).

*   `--timeout-sec`  
    How long to wait for correction in TC2/TC3 (default: 60 seconds).

*   `--threshold-sec`  
    Allowed server offset in seconds when a server probe exists (default: 300).

*   `--year-tolerance`  
    Fallback tolerance for device year vs host year (default: 1, to handle New Year boundary).

*   `--bad-time`  
    The forced “bad time” string in Android `date` format: `MMDDhhmmYYYY.SS`  
    Default: `010100002000.00` (Jan 1, 2000 00:00:00).

*   `--no-raw`  
    Do not save raw dumpsys outputs in `runs/<stamp>/raw`.

*   `--no-wifi`, `--no-mobile`, `--no-reboot`  
    Skip individual phases.

## Interpreting the reports

### Summary section

Shows each phase, PASS/FAIL, and time-to-correct (if applicable).

### Details section

Shows the last samples per phase (controlled by `--html-last-n`).

### Glossary / columns (HTML report)

The HTML report includes a Glossary. Key columns:

*   **Host time**  
    Timestamp on your PC when sample was taken (ISO format).

*   **Device date**  
    Raw output of `adb shell date` from the device.

*   **No default network (offline)**  
    Derived from `dumpsys connectivity`.  
    `Offline` means: Android reports “Active default network: none”.

*   **Server time offset (s)**  
    Derived from Android’s HTTP connectivity probe (Date header vs received time).  
    Near 0 means device time aligns closely with server time.  
    The report may label this value as **(stale)** when:
    *   the device is offline, or
    *   the absolute value is unusually large (e.g., > 1 day),
        because probe info can be retained or inconsistent during clock jumps.

*   **Wall‑clock jump (ms)**  
    `RTC - ELAPSED` deviation between samples.  
    Large values indicate a time jump/correction event.

*   **Notes**
    A tester-friendly description of what the sample represents, e.g.:
    *   “⛔ Offline baseline — time must stay wrong”
    *   “⏳ Waiting for time correction”
    *   “✅ Time corrected”
    *   “🔁 Post‑reboot — verify time persists offline”

## Troubleshooting

### “Failed to set bad time” / time does not change

*   Root/`su` may be missing, denied, or not functional.
*   Try running:
    ```bash
    adb shell su -c id
    adb shell su 0 id
    adb shell su root id
    ```
*   If none work, the device likely can’t set time via shell.

### Time corrects even while mobile data is disabled

Some devices can receive time via cellular (e.g., NITZ) even when mobile data is disabled.
This script mitigates this by keeping `auto_time=0` during offline baseline phases, and only enabling `auto_time=1` right before enabling Wi‑Fi/mobile for correction phases.

### “Server time offset” appears while offline

`dumpsys connectivity` can retain old probe information. The HTML report may label it as **(stale)** to reduce confusion.

### Device goes “offline” in ADB / transient ADB errors

ADB can be unstable during reboot or USB reconnects. The script retries common transient ADB failures (device offline, unauthorized, etc.). If it persists:

*   Reconnect cable
*   Re-authorize USB debugging
*   Run `adb devices` and verify state is `device`

## Safety / disclaimer

This script changes system time and toggles network states on the device. Use only on test devices.

## License

This project is licensed under the Apache License, Version 2.0.

You may use, modify, and distribute this software in compliance with the License.
A copy of the License is available in the [LICENSE](LICENSE) file or at:

https://www.apache.org/licenses/LICENSE-2.0

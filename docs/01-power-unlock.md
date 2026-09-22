# 01 — Unlocking a Laptop GPU's Full Power Limit on Linux (80 W → 175 W)

**Applicable to: MSI laptops with NVIDIA RTX GPUs on Linux (tested on MSI Raider A18 HX, RTX 4080 Laptop).** The mechanism is vendor-generic, so other brands' laptops may need analogous vendor-EC tooling, but the `nvidia-powerd` part applies to every NVIDIA laptop.

## The symptom

```
$ nvidia-smi -q -d POWER
    Current Power Limit    : 80.00 W     ← the base TGP
    Default Power Limit    : 80.00 W
    Max Power Limit        : 175.00 W    ← the hardware is capable of 175 W
$ sudo nvidia-smi -pl 175
Changing power management limit is not supported for GPU ...   ← always refused on laptops
```

GPU clocks plateau around **1230–1335 MHz** under full load (the chip's boost ceiling is over 3000 MHz), and `SW Power Capping` accumulates seconds of throttling time. The GPU is not hot — it is *power-starved*.

The same machine under Windows reports `Current Power Limit : 175.00 W` out of the box, so this is purely a Linux-side configuration gap.

## Root cause A — `nvidia-powerd` is installed but never enabled

On Ubuntu the driver package ships the systemd unit and the D-Bus policy **inside its documentation directory** and does **not** install them:

```bash
$ systemctl status nvidia-powerd
Unit nvidia-powerd.service could not be found.      # ← nothing installed
$ ls /usr/share/doc/nvidia-kernel-common-580/nvidia-powerd.service
$ ls /usr/share/doc/nvidia-driver-580/nvidia-dbus.conf
```

(Directory versions vary with your driver version — adjust `580` to match.)

### Fix (both files are mandatory)

```bash
# 1. Install the service unit
sudo cp /usr/share/doc/nvidia-kernel-common-580/nvidia-powerd.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-powerd

# 2. Install the D-Bus policy — without it the daemon starts but its
#    D-Bus connection fails and Dynamic Boost does not fully engage
sudo cp /usr/share/doc/nvidia-driver-580/nvidia-dbus.conf /etc/dbus-1/system.d/nvidia-powerd.conf
sudo systemctl restart dbus
sudo systemctl restart nvidia-powerd
```

**Success markers**:

```bash
$ journalctl -u nvidia-powerd -n 5
    nvidia-powerd version:2.0 (build 1)
    DBus Connection is established          ← exactly this line

$ nvidia-smi -q -d POWER | grep "Current Power Limit"
    Current Power Limit    : 105.00 W       ← 80 W + 25 W Dynamic Boost
```

**Symptom if you skip step 2**:
```
Error requesting D-Bus name (Connection ":1.29" is not allowed to own the service
"nvidia.powerd.server" due to security policies in the configuration file)
Failed to acquire D-Bus name ((null))
Error setting up DBus connection
```

## Root cause B — the EC performance profile has no Linux driver

Even with `nvidia-powerd` running, the limit sat at **105 W** (base + 25 W Dynamic Boost), not 175 W. The remaining gap is the vendor's **EC (embedded controller) performance profile** — on MSI laptops the "User Scenario" (Extreme Performance / Balanced / Silent) that Windows' MSI Center sets.

On Linux the FN hotkey reaches the kernel but nothing acts on it:

```
$ dmesg | tail
atkbd serio0: Unknown key released (translated set 2, code 0x75 ...)
atkbd serio0: Use 'setkeycodes 75 <keycode>' to make it known.
```

The fix is the community [`msi-ec`](https://github.com/BeardOverflow/msi-ec) kernel module, which exposes the EC as sysfs attributes:

```bash
git clone https://github.com/BeardOverflow/msi-ec.git
cd msi-ec
make                       # needs linux-headers for your running kernel
sudo make dkms-install     # persistent across kernel updates

# after loading:
ls /sys/devices/platform/msi-ec/
# available_fan_modes  available_shift_modes  cooler_boost  fan_mode
# fw_release_date  fw_version  shift_mode  super_battery  webcam_block ...

cat /sys/devices/platform/msi-ec/fw_version      # e.g. 182KIMS1.113
cat /sys/devices/platform/msi-ec/available_shift_modes   # turbo eco comfort

echo turbo | sudo tee /sys/devices/platform/msi-ec/shift_mode
sudo systemctl restart nvidia-powerd
```

**Result so far: 105 W → 150 W.**

## The last step to 175 W: a sustained load

The final negotiation happens under load. On our machine, and matching reports from other vendors' laptops:

```bash
# run something GPU-heavy for 1–2 minutes (we used repeated long-context prefills)
$ nvidia-smi -q -d POWER | grep -E "Current|Max"
    Current Power Limit    : 175.00 W      ← the ceiling, reached
    Max Power Limit        : 175.00 W
```

Measured under sustained load:

| Metric | Before (80 W) | After (175 W) | Change |
|---|---|---|---|
| Power draw | 79.7–80.2 W | **145–148 W** | +83 % |
| SM clock (steady) | 1230–1335 MHz | **2400 MHz** | +95 % |
| Temperature | 54–61 °C | 65–72 °C | still healthy |

> Note: after the kernel upgrade to 7.0 the 175 W ceiling engaged **immediately at boot** with no load step. On the older 6.8 kernel the load step was required once.

## Fan behaviour — you do *not* need the "cooler boost" hammer

`cooler_boost=on` forces maximum fan RPM. Measured head-to-head under identical load:

| Mode | Peak temp | Peak power | SM clock |
|---|---|---|---|
| `cooler_boost on` (forced max) | 78 °C | 166 W | 2400 MHz |
| **`cooler_boost off` (EC automatic)** | **76 °C** | 147 W | 2385 MHz |

The EC's automatic fan curve is as good or better — leave it alone:

```bash
echo off  | sudo tee /sys/devices/platform/msi-ec/cooler_boost
echo auto | sudo tee /sys/devices/platform/msi-ec/fan_mode
```

## Persisting everything across reboots

```bash
# 1. msi-ec: DKMS handles the module
sudo make dkms-install          # in the msi-ec source tree

# 2. Boot-time EC profile:
sudo tee /usr/local/bin/msi-ec-setup.sh >/dev/null <<'EOF'
#!/bin/bash
for i in $(seq 1 10); do [ -d /sys/devices/platform/msi-ec ] && break; sleep 1; done
if [ -d /sys/devices/platform/msi-ec ]; then
    echo turbo > /sys/devices/platform/msi-ec/shift_mode   2>/dev/null
    echo off   > /sys/devices/platform/msi-ec/cooler_boost 2>/dev/null
    echo auto  > /sys/devices/platform/msi-ec/fan_mode     2>/dev/null
fi
EOF
sudo chmod +x /usr/local/bin/msi-ec-setup.sh

sudo tee /etc/systemd/system/msi-ec-setup.service >/dev/null <<'EOF'
[Unit]
Description=MSI EC performance profile
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/local/bin/msi-ec-setup.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable msi-ec-setup.service

# 3. CPU governor (the MoE hybrid engine leans on CPU too)
sudo tee /etc/systemd/system/cpu-performance.service >/dev/null <<'EOF'
[Unit]
Description=Set CPU governor to performance
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable cpu-performance.service
```

## What is *not* possible (measured, don't waste your time)

| Attempt | Result |
|---|---|
| `nvidia-smi -pl <anything>` | Always refused on laptop GPUs ("not supported") |
| Memory overclock `nvidia-smi -lmc 9500` | Command *reports success* but the clock stays at 9001 MHz — silently ignored |
| Core-clock lock `nvidia-smi -lgc 2400,3105` | **Works**, but costs idle power (7 W → 35 W idle); the GPU already boosts to 2400 on demand, so we reverted it |
| Vendor "User Scenario" already set to performance | Does nothing for the *GPU* budget on this model — only the EC path above did |

## Kernel version matters

We found the whole chain much better behaved on kernel **7.0.0-31** (Ubuntu 24.04 HWE) than on 6.8:
- 6.8: needed the community `msi-ec` DKMS module (mainline rejected the EC firmware as unsupported)
- 7.0: mainline ships `msi-ec` (`/lib/modules/<ver>/kernel/drivers/platform/x86/msi-ec.ko`), and the 175 W ceiling engaged without the load step

Upgrading the LTS kernel is one command (`apt install linux-generic-hwe-24.04`) and far less work than reinstalling a newer distro — the rest of your tuned stack stays untouched. See the notes in [docs/04-methodology.md](04-methodology.md) for the upgrade pitfalls we hit (nohup for long apt runs, GRUB timeout, DKMS rebuild).

## Verification one-liner

```bash
nvidia-smi -q -d POWER | grep "Current Power Limit" | head -1
# under load it should read 175.00 W; idle it may show 150.00 W — both correct
```

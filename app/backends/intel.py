"""IntelBackend — sysfs/fdinfo-based GPU monitoring for i915 and xe drivers."""
import glob
import os
import subprocess
import time
from typing import Optional

from app.backends.base import GpuBackend


class IntelBackend(GpuBackend):
    """GPU monitoring backend using Linux sysfs and fdinfo for Intel GPUs."""

    XE_ENGINE_MAP = {
        "rcs": "Render/3D",
        "vcs": "Video",
        "vecs": "VideoEnhance",
        "bcs": "Blitter",
        "ccs": "Compute",
    }

    I915_ENGINE_MAP = {
        "render": "Render/3D",
        "video": "Video",
        "video-enhance": "VideoEnhance",
        "copy": "Blitter",
    }

    def __init__(self) -> None:
        self._drm_base: str = "/sys/class/drm"
        self._proc_path: str = "/proc"
        self._prev_counters: dict = {}
        self._prev_time: Optional[float] = None
        self._prev_rc6_ms: dict = {}
        self._prev_energy_uj: dict = {}
        self._driver_cache: dict = {}

    # ── Sysfs helpers ────────────────────────────────────────────────

    def _read_sysfs(self, path: str) -> Optional[str]:
        """Read a sysfs file, return stripped content or None."""
        try:
            with open(path) as f:
                return f.read().strip()
        except (OSError, IOError):
            return None

    def _detect_driver(self, card: str) -> Optional[str]:
        """Detect the kernel driver for *card*, caching the result."""
        if card in self._driver_cache:
            return self._driver_cache[card]
        driver_path = os.path.join(self._drm_base, card, "device", "driver")
        try:
            target = os.readlink(driver_path)
            driver = os.path.basename(target)
        except OSError:
            return None
        self._driver_cache[card] = driver
        return driver

    def _detect_gpu_name(self, card: str) -> str:
        """Return a human-readable GPU name for *card*."""
        product = self._read_sysfs(
            os.path.join(self._drm_base, card, "device", "product_name")
        )
        if product:
            return product

        # Fallback: try lspci
        try:
            pci_addr_path = os.path.join(self._drm_base, card, "device")
            pci_addr = os.path.basename(os.readlink(pci_addr_path))
            result = subprocess.run(
                ["lspci", "-s", pci_addr, "-mm"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('"')[5]
        except Exception:
            pass

        return "Intel GPU"

    # ── Sysfs reads: frequency, RC6, power ─────────────────────────

    def _read_frequency(self, card: str, driver: str) -> dict:
        """Read actual and requested GPU frequency from sysfs."""
        if driver == "xe":
            base = os.path.join(
                self._drm_base, card, "device", "tile0", "gt0", "freq0"
            )
            actual = self._read_sysfs(os.path.join(base, "act_freq"))
            requested = self._read_sysfs(os.path.join(base, "cur_freq"))
        else:  # i915
            base = os.path.join(self._drm_base, card)
            actual = self._read_sysfs(os.path.join(base, "gt_act_freq_mhz"))
            requested = self._read_sysfs(os.path.join(base, "gt_cur_freq_mhz"))
        return {
            "actual": float(actual) if actual else 0.0,
            "requested": float(requested) if requested else 0.0,
        }

    def _read_rc6_ms(self, card: str, driver: str) -> Optional[float]:
        """Read RC6/idle residency in milliseconds."""
        if driver == "xe":
            path = os.path.join(
                self._drm_base, card,
                "device", "tile0", "gt0", "gtidle", "idle_residency_ms",
            )
        else:  # i915
            path = os.path.join(
                self._drm_base, card, "gt", "gt0", "rc6_residency_ms",
            )
        val = self._read_sysfs(path)
        return float(val) if val is not None else None

    def _find_hwmon(self, card: str) -> Optional[str]:
        """Find the hwmon directory under the card's device."""
        hwmon_base = os.path.join(self._drm_base, card, "device", "hwmon")
        if not os.path.isdir(hwmon_base):
            return None
        entries = sorted(os.listdir(hwmon_base))
        for entry in entries:
            candidate = os.path.join(hwmon_base, entry)
            if os.path.isdir(candidate):
                return candidate
        return None

    def _read_energy_uj(self, card: str) -> Optional[float]:
        """Read energy counter (microjoules) from hwmon."""
        hwmon = self._find_hwmon(card)
        if hwmon is None:
            return None
        val = self._read_sysfs(os.path.join(hwmon, "energy1_input"))
        return float(val) if val is not None else None

    # ── fdinfo parsing ─────────────────────────────────────────────

    _MEMORY_STATS = frozenset({
        "total", "shared", "resident", "active", "purgeable",
    })

    def _parse_fdinfo(self, content: str, driver: str) -> Optional[dict]:
        """Parse a single fdinfo file.

        Returns ``{"client_id": str, "engines": {...}, "memory": {...}}``
        or *None* if the file does not belong to *driver*.
        """
        engines: dict = {}
        memory: dict = {}
        client_id: Optional[str] = None
        found_driver = False

        for line in content.splitlines():
            if ":\t" not in line:
                continue
            key, _, value = line.partition(":\t")
            value = value.strip()

            if key == "drm-driver":
                if value != driver:
                    return None
                found_driver = True
                continue

            if key == "drm-client-id":
                client_id = value
                continue

            # ── i915 engine lines: drm-engine-<name>: <value> ns ─────
            if key.startswith("drm-engine-") and not key.startswith("drm-engine-capacity-"):
                engine_name = key[len("drm-engine-"):]
                ns_str = value.replace(" ns", "")
                engines.setdefault(engine_name, {})["ns"] = int(ns_str)
                continue

            # ── xe cycle lines ───────────────────────────────────────
            if key.startswith("drm-total-cycles-"):
                engine_name = key[len("drm-total-cycles-"):]
                engines.setdefault(engine_name, {})["total_cycles"] = int(value)
                continue

            if key.startswith("drm-cycles-"):
                engine_name = key[len("drm-cycles-"):]
                engines.setdefault(engine_name, {})["cycles"] = int(value)
                continue

            if key.startswith("drm-engine-capacity-"):
                engine_name = key[len("drm-engine-capacity-"):]
                engines.setdefault(engine_name, {})["capacity"] = int(value)
                continue

            # ── memory lines: drm-{stat}-{region}: <value> ──────────
            if key.startswith("drm-"):
                rest = key[4:]  # strip "drm-"
                # Try to split into stat-region
                for stat in self._MEMORY_STATS:
                    prefix = stat + "-"
                    if rest.startswith(prefix):
                        region = rest[len(prefix):]
                        memory.setdefault(region, {})[stat] = int(value)
                        break

        if not found_driver:
            return None

        return {
            "client_id": client_id,
            "engines": engines,
            "memory": memory,
        }

    def _map_engine_name(self, name: str, driver: str) -> str:
        """Map a raw engine name to a human-readable name."""
        if driver == "xe":
            return self.XE_ENGINE_MAP.get(name, name)
        return self.I915_ENGINE_MAP.get(name, name)

    # ── Public API ───────────────────────────────────────────────────

    def discover_devices(self) -> list[dict]:
        """Enumerate DRM card devices backed by Intel GPUs."""
        devices: list[dict] = []
        pattern = os.path.join(self._drm_base, "card*")
        for card_path in sorted(glob.glob(pattern)):
            card = os.path.basename(card_path)
            # Skip renderD nodes that might match card* glob
            if not card.startswith("card"):
                continue
            # Skip renderD-style names (e.g. "card" prefix but has extra chars
            # beyond digits — shouldn't happen, but be safe)
            suffix = card[4:]
            if not suffix.isdigit():
                continue

            vendor = self._read_sysfs(
                os.path.join(card_path, "device", "vendor")
            )
            if vendor != "0x8086":
                continue

            driver = self._detect_driver(card)
            if driver is None:
                continue

            name = self._detect_gpu_name(card)
            devices.append({
                "device": card,
                "name": name,
                "driver": driver,
            })
        return devices

    def read_sample(self, device: str) -> dict:
        raise NotImplementedError

    def cleanup(self) -> None:
        """Clear all internal state."""
        self._prev_counters = {}
        self._prev_time = None
        self._prev_rc6_ms = {}
        self._prev_energy_uj = {}
        self._driver_cache = {}

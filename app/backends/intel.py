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

    # ── Utilization computation ─────────────────────────────────────

    def _compute_utilization(
        self,
        prev: dict,
        curr: dict,
        wall_time_s: float,
        driver: str,
    ) -> dict:
        """Compute per-engine utilization percentages from counter deltas.

        *prev* and *curr* are dicts keyed by client_id, each with an
        ``"engines"`` sub-dict mapping engine names to counter dicts.

        Returns ``{client_id: {"engines": {engine_name: busy_pct}}}``
        for every client_id present in *curr*.
        """
        result: dict = {}
        for cid, cdata in curr.items():
            engines_out: dict = {}
            pdata = prev.get(cid)
            for engine, counters in cdata.get("engines", {}).items():
                if pdata is None:
                    # New client — no previous data, report 0%
                    engines_out[engine] = 0.0
                    continue

                prev_counters = pdata.get("engines", {}).get(engine)
                if prev_counters is None:
                    engines_out[engine] = 0.0
                    continue

                if driver == "xe":
                    busy_pct = self._xe_busy_pct(prev_counters, counters)
                else:
                    busy_pct = self._i915_busy_pct(
                        prev_counters, counters, wall_time_s,
                    )

                engines_out[engine] = max(0.0, min(busy_pct, 100.0))

            result[cid] = {"engines": engines_out}
        return result

    @staticmethod
    def _i915_busy_pct(prev: dict, curr: dict, wall_time_s: float) -> float:
        delta_ns = curr.get("ns", 0) - prev.get("ns", 0)
        if wall_time_s <= 0:
            return 0.0
        return delta_ns / (wall_time_s * 1e9) * 100.0

    @staticmethod
    def _xe_busy_pct(prev: dict, curr: dict) -> float:
        delta_cycles = curr.get("cycles", 0) - prev.get("cycles", 0)
        delta_total = curr.get("total_cycles", 0) - prev.get("total_cycles", 0)
        if delta_total <= 0:
            return 0.0
        return delta_cycles / delta_total * 100.0

    # ── fdinfo scanning ──────────────────────────────────────────────

    def _scan_fdinfo(self, proc_path: str, driver: str) -> list[dict]:
        """Scan /proc/*/fdinfo/* for DRM clients matching *driver*.

        Returns a list of dicts, each with keys: client_id, engines,
        memory, pid, name.  Deduplicates by drm-client-id (first wins).
        """
        seen_client_ids: dict[str, dict] = {}
        try:
            entries = os.listdir(proc_path)
        except OSError:
            return []

        for pid_name in entries:
            if not pid_name.isdigit():
                continue
            fdinfo_dir = os.path.join(proc_path, pid_name, "fdinfo")
            try:
                fd_entries = os.listdir(fdinfo_dir)
            except OSError:
                continue

            fd_dir = os.path.join(proc_path, pid_name, "fd")
            for fd_name in fd_entries:
                # Fast path: check if this fd points to a DRM device
                # before reading the full fdinfo file. readlink is a
                # single syscall vs open+read+close for fdinfo.
                try:
                    target = os.readlink(os.path.join(fd_dir, fd_name))
                except OSError:
                    continue
                if "/dev/dri/" not in target:
                    continue

                fd_path = os.path.join(fdinfo_dir, fd_name)
                try:
                    with open(fd_path) as f:
                        content = f.read()
                except OSError:
                    continue

                if "drm-driver:" not in content:
                    continue

                parsed = self._parse_fdinfo(content, driver)
                if parsed is None:
                    continue

                client_id = parsed["client_id"]
                if client_id in seen_client_ids:
                    continue

                # Read process name from comm
                comm_path = os.path.join(proc_path, pid_name, "comm")
                try:
                    with open(comm_path) as f:
                        name = f.read().strip()
                except OSError:
                    name = "unknown"

                parsed["pid"] = pid_name
                parsed["name"] = name
                seen_client_ids[client_id] = parsed

        return list(seen_client_ids.values())

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
        now = time.time()
        driver = self._detect_driver(device) or "i915"
        wall_time_s = now - self._prev_time if self._prev_time and self._prev_time > 0 else 1.0

        # Frequency
        frequency = self._read_frequency(device, driver)

        # RC6 / GPU busy
        rc6_ms = self._read_rc6_ms(device, driver)
        gpu_busy = None
        prev_rc6 = self._prev_rc6_ms.get(device)
        if rc6_ms is not None and prev_rc6 is not None and wall_time_s > 0:
            delta_rc6_ms = rc6_ms - prev_rc6
            rc6_pct = (delta_rc6_ms / (wall_time_s * 1000)) * 100
            gpu_busy = max(0.0, min(100.0, 100.0 - rc6_pct))
        self._prev_rc6_ms[device] = rc6_ms

        # Power
        energy_uj = self._read_energy_uj(device)
        power_watts = None
        prev_energy = self._prev_energy_uj.get(device)
        if energy_uj is not None and prev_energy is not None and wall_time_s > 0:
            delta_uj = energy_uj - prev_energy
            power_watts = delta_uj / (wall_time_s * 1_000_000)
        self._prev_energy_uj[device] = energy_uj

        # Per-process fdinfo
        raw_clients = self._scan_fdinfo(self._proc_path, driver)

        # Build counter snapshot for delta computation
        curr_counters: dict = {}
        for client in raw_clients:
            curr_counters[client["client_id"]] = {"engines": client["engines"]}

        # Compute utilization deltas
        util = self._compute_utilization(self._prev_counters, curr_counters, wall_time_s, driver)
        self._prev_counters = curr_counters
        self._prev_time = now

        # Device-level engine utilization (aggregate from clients)
        engine_totals: dict[str, float] = {}
        for client_id, client_util in util.items():
            for engine, busy_pct in client_util["engines"].items():
                display_name = self._map_engine_name(engine, driver)
                key = f"{display_name}/0"
                engine_totals[key] = engine_totals.get(key, 0.0) + busy_pct
        engines = {key: {"busy": min(total, 100.0)} for key, total in engine_totals.items()}

        # GPU busy fallback: derive from max engine utilization if no RC6
        if gpu_busy is None and engines:
            gpu_busy = max(e["busy"] for e in engines.values())
        elif gpu_busy is None:
            gpu_busy = 0.0

        # Build clients dict
        clients: dict = {}
        for client in raw_clients:
            client_id = client["client_id"]
            client_util = util.get(client_id, {}).get("engines", {})
            engine_classes: dict = {}
            for engine, busy_pct in client_util.items():
                display_name = self._map_engine_name(engine, driver)
                engine_classes[display_name] = {"busy": round(busy_pct, 1)}

            memory: dict = {}
            if client.get("memory"):
                memory = {"system": client["memory"].get("system", {})}
                for region, data in client["memory"].items():
                    if region != "system":
                        memory[region] = data

            client_entry: dict = {
                "pid": client["pid"],
                "name": client["name"],
                "engine-classes": engine_classes,
            }
            if memory:
                client_entry["memory"] = memory
            clients[client_id] = client_entry

        power = {"GPU": round(power_watts, 1)} if power_watts is not None else None

        return {
            "period": {"duration": wall_time_s * 1000},
            "frequency": frequency,
            "gpu_busy": round(gpu_busy, 1),
            "power": power,
            "engines": engines,
            "clients": clients,
        }

    def cleanup(self) -> None:
        """Clear all internal state."""
        self._prev_counters = {}
        self._prev_time = None
        self._prev_rc6_ms = {}
        self._prev_energy_uj = {}
        self._driver_cache = {}

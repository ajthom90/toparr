(function () {
  "use strict";

  // --- Config ---
  var HISTORY_MAX = 300;
  var ENGINE_COLORS = {
    "Render/3D": "#60a5fa",
    "Video": "#34d399",
    "VideoEnhance": "#a855f7",
    "Blitter": "#fbbf24",
  };

  // --- State ---
  var history = [];
  var engineNames = [];
  var tdpWatts = 60;
  var gpuName = "";
  var startTime = Date.now();
  var connected = false;

  // --- DOM refs ---
  function $(id) { return document.getElementById(id); }

  // --- Init ---
  async function init() {
    await fetchStatus();
    connectSSE();
  }

  // --- Fetch initial status ---
  async function fetchStatus() {
    try {
      var resp = await fetch("/api/status");
      var data = await resp.json();
      gpuName = data.gpu_name || "Intel GPU";
      $("gpu-name").textContent = gpuName;
      startTime = Date.now() - data.uptime_seconds * 1000;
      if (data.tdp_watts) tdpWatts = data.tdp_watts;

      if (data.history && data.history.length > 0) {
        history = data.history.slice(-HISTORY_MAX);
        discoverEngines(history[history.length - 1]);
        render(history[history.length - 1]);
      }
    } catch (e) {
      console.error("Failed to fetch status:", e);
    }
  }

  // --- SSE ---
  function connectSSE() {
    var eventSource = new EventSource("/api/stream");
    var hasConnectedBefore = false;

    eventSource.addEventListener("gpu_data", function (e) {
      connected = true;
      setConnectionStatus("live");
      hideError();
      var data = JSON.parse(e.data);
      history.push(data);
      if (history.length > HISTORY_MAX) history.shift();
      discoverEngines(data);
      render(data);
    });

    eventSource.addEventListener("status", function (e) {
      var data = JSON.parse(e.data);
      if (data.status === "waiting") {
        setConnectionStatus("warning", "GPU Unavailable");
        showError(data.error || "Waiting for GPU data...");
      }
    });

    eventSource.addEventListener("open", function () {
      connected = true;
      setConnectionStatus("live");
      // On reconnect (not first connect), backfill missed history
      if (hasConnectedBefore) {
        fetchStatus();
      }
      hasConnectedBefore = true;
    });

    eventSource.addEventListener("error", function () {
      connected = false;
      setConnectionStatus("disconnected", "Reconnecting...");
      // EventSource auto-reconnects; backfill happens in the open handler
    });
  }

  // --- Connection status ---
  function setConnectionStatus(state, text) {
    var el = $("connection-status");
    el.className = "status";
    if (state === "live") {
      el.querySelector("span").textContent = "Live \u2014 updating every 1s";
    } else if (state === "warning") {
      el.classList.add("warning");
      el.querySelector("span").textContent = text || "Warning";
    } else if (state === "disconnected") {
      el.classList.add("disconnected");
      el.querySelector("span").textContent = text || "Disconnected";
    }
  }

  function showError(msg) {
    $("error-banner").style.display = "block";
    $("error-message").textContent = msg;
  }

  function hideError() {
    $("error-banner").style.display = "none";
  }

  // --- Discover engines ---
  function discoverEngines(sample) {
    if (!sample || !sample.engines) return;
    var names = Object.keys(sample.engines).map(function (k) {
      return k.replace(/\/\d+$/, "");
    });
    if (JSON.stringify(names) !== JSON.stringify(engineNames)) {
      engineNames = names;
    }
  }

  function engineColor(name) {
    if (ENGINE_COLORS[name]) return ENGINE_COLORS[name];
    for (var key in ENGINE_COLORS) {
      if (name.startsWith(key)) return ENGINE_COLORS[key];
    }
    return "#888";
  }

  // --- Render ---
  function render(sample) {
    if (!sample) return;
    renderGpuBusy(sample);
    renderFrequency(sample);
    renderPower(sample);
    renderEngineBars(sample);
    renderSparklines();
    renderClients(sample);
    renderFooter();
  }

  function renderGpuBusy(sample) {
    var rc6 = sample.rc6 ? sample.rc6.value : 0;
    var busy = Math.max(0, Math.min(100, 100 - rc6));
    var circumference = 213.6;
    var offset = circumference - (busy / 100) * circumference;
    $("gpu-busy-arc").style.strokeDashoffset = offset;
    $("gpu-busy-pct").textContent = busy.toFixed(0) + "%";

    var irq = sample.interrupts ? sample.interrupts.count : 0;
    $("interrupts-val").textContent = Math.round(irq).toLocaleString();
  }

  function renderFrequency(sample) {
    var freq = sample.frequency;
    if (!freq) return;
    $("freq-actual").textContent = Math.round(freq.actual);
    $("freq-requested").textContent = Math.round(freq.requested);
    $("freq-actual-sub").textContent = Math.round(freq.actual);
  }

  function renderPower(sample) {
    var power = sample.power;
    if (!power) {
      $("power-gpu").textContent = "N/A";
      $("power-bar").style.width = "0%";
      return;
    }
    var gpu = power.GPU || 0;
    $("power-gpu").textContent = gpu.toFixed(1);
    var pct = Math.min(100, (gpu / tdpWatts) * 100);
    $("power-bar").style.width = pct + "%";
    $("power-tdp-label").textContent = tdpWatts + "W TDP";
  }

  function renderEngineBars(sample) {
    if (!sample.engines) return;
    var container = $("engine-bars");
    var html = "";
    for (var i = 0; i < engineNames.length; i++) {
      var name = engineNames[i];
      var key = null;
      var keys = Object.keys(sample.engines);
      for (var j = 0; j < keys.length; j++) {
        if (keys[j].startsWith(name)) { key = keys[j]; break; }
      }
      var busy = key ? sample.engines[key].busy : 0;
      var color = engineColor(name);
      var shortName = name.replace("/3D", "");
      html += '<div class="bar-row">' +
        '<div class="bar-label">' + shortName + '</div>' +
        '<div class="bar-track"><div class="bar-fill" style="width:' + busy + '%;background:' + color + '"></div></div>' +
        '<div class="bar-value" style="color:' + color + '">' + busy.toFixed(0) + '%</div>' +
        '</div>';
    }
    container.innerHTML = html;
  }

  function renderSparklines() {
    if (engineNames.length === 0) return;
    var container = $("sparklines");
    var html = "";
    for (var i = 0; i < engineNames.length; i++) {
      var name = engineNames[i];
      var color = engineColor(name);
      var shortName = name.replace("/3D", "");
      var values = history.map(function (s) {
        if (!s.engines) return 0;
        var keys = Object.keys(s.engines);
        for (var j = 0; j < keys.length; j++) {
          if (keys[j].startsWith(name)) return s.engines[keys[j]].busy;
        }
        return 0;
      });
      var current = values.length > 0 ? values[values.length - 1] : 0;
      var points = sparklinePoints(values, 300, 30);
      html += '<div class="sparkline-row">' +
        '<div class="sparkline-label" style="color:' + color + '">' + shortName + '</div>' +
        '<div class="sparkline"><svg width="100%" height="30" preserveAspectRatio="none" viewBox="0 0 300 30">' +
        '<polyline fill="none" stroke="' + color + '" stroke-width="1.5" points="' + points + '"/>' +
        '</svg></div>' +
        '<div class="sparkline-val" style="color:' + color + '">' + current.toFixed(0) + '%</div>' +
        '</div>';
    }
    container.innerHTML = html;
  }

  function sparklinePoints(values, width, height) {
    if (values.length === 0) return "";
    var step = width / Math.max(values.length - 1, 1);
    var parts = [];
    for (var i = 0; i < values.length; i++) {
      var x = i * step;
      var y = height - (values[i] / 100) * height;
      parts.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    return parts.join(" ");
  }

  function renderClients(sample) {
    var tbody = $("clients-body");
    var thead = $("clients-head");
    if (!sample.clients || Object.keys(sample.clients).length === 0) {
      thead.innerHTML = '<tr><th>PID</th><th>Name</th></tr>';
      tbody.innerHTML = '<tr><td colspan="2" class="placeholder-text">No active clients</td></tr>';
      return;
    }

    var engineClasses = {};
    var clients = Object.values(sample.clients);
    for (var i = 0; i < clients.length; i++) {
      var ec = clients[i]["engine-classes"];
      if (ec) {
        var ecKeys = Object.keys(ec);
        for (var j = 0; j < ecKeys.length; j++) {
          engineClasses[ecKeys[j]] = true;
        }
      }
    }
    var ecList = Object.keys(engineClasses).sort();

    var headerHtml = '<tr><th>PID</th><th>Name</th>';
    for (var i = 0; i < ecList.length; i++) {
      headerHtml += '<th>' + ecList[i].replace("/3D", "") + '</th>';
    }
    headerHtml += '</tr>';
    thead.innerHTML = headerHtml;

    var rows = "";
    for (var i = 0; i < clients.length; i++) {
      var client = clients[i];
      var name = client.name || "unknown";
      var pid = client.pid || "?";
      rows += '<tr><td>' + pid + '</td><td style="color:#60a5fa">' + name + '</td>';
      for (var j = 0; j < ecList.length; j++) {
        var data = client["engine-classes"] ? client["engine-classes"][ecList[j]] : null;
        var busy = data ? parseFloat(data.busy) : 0;
        rows += '<td>' + busy.toFixed(0) + '%</td>';
      }
      rows += '</tr>';
    }
    tbody.innerHTML = rows;
  }

  function renderFooter() {
    var elapsed = (Date.now() - startTime) / 1000;
    $("uptime").textContent = formatUptime(elapsed);
    $("footer-status").textContent = connected
      ? "Refreshing via SSE every ~1s"
      : "Disconnected";
  }

  function formatUptime(seconds) {
    var d = Math.floor(seconds / 86400);
    var h = Math.floor((seconds % 86400) / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    if (d > 0) return d + "d " + h + "h " + m + "m";
    if (h > 0) return h + "h " + m + "m";
    return m + "m";
  }

  init();
})();

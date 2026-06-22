// AI Anomaly Detection Dashboard Logic
let streamInterval = null;
let liveChart = null;
let lossChart = null;
let currentDataset = "industrial";
let isStreaming = false;
let trainingPollInterval = null;
let currentFaults = new Set();

// Keep track of last anomaly state to detect transitions
let wasAnomalous = false;

// Sensor metadata
const sensorMeta = {
    "temperature": { name: "Temperature", unit: "°C", icon: "fa-temperature-high" },
    "pressure": { name: "Pressure", unit: "PSI", icon: "fa-gauge" },
    "vibration": { name: "Vibration", unit: "g", icon: "fa-shake" },
    "power": { name: "Power Load", unit: "kW", icon: "fa-bolt" },
    "value": { name: "Reading Value", unit: "units", icon: "fa-chart-line" }
};

document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    setupEventListeners();
    loadDatasetMeta();
    
    // Start streaming automatically
    toggleStream();
});

// Setup Events
function setupEventListeners() {
    // Dataset select
    document.getElementById("dataset-select").addEventListener("change", (e) => {
        changeDataset(e.target.value);
    });

    // Stream Toggle
    document.getElementById("stream-toggle").addEventListener("click", () => {
        toggleStream();
    });

    // Train button
    document.getElementById("train-btn").addEventListener("click", () => {
        startTraining();
    });

    // Reset simulator buttons
    document.getElementById("reset-sim-btn").addEventListener("click", () => resetSimulator());
    document.getElementById("reset-single-btn").addEventListener("click", () => resetSimulator());

    // Fault injection buttons
    document.querySelectorAll(".inject-btn[data-fault]").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const faultType = e.currentTarget.getAttribute("data-fault");
            injectFault(faultType, e.currentTarget);
        });
    });

    // Clear log
    document.getElementById("clear-log-btn").addEventListener("click", () => {
        const body = document.getElementById("anomaly-log-body");
        body.innerHTML = `
            <tr>
                <td colspan="5" class="empty-log-msg">
                    <i class="fa-solid fa-shield-halved"></i> No anomalies detected. System running normally.
                </td>
            </tr>
        `;
    });
}

// Initialize charts
function initCharts() {
    const liveCtx = document.getElementById("live-chart").getContext("2d");
    liveChart = new Chart(liveCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 0, 0, 0.1)' },
                    ticks: { color: '#ff3333', font: { family: 'Outfit', weight: 'bold' } }
                },
                y: {
                    grid: { color: 'rgba(255, 0, 0, 0.1)' },
                    ticks: { color: '#ff3333', font: { family: 'Outfit', weight: 'bold' } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    const lossCtx = document.getElementById("loss-chart").getContext("2d");
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#ff0000',
                    backgroundColor: 'rgba(255, 0, 0, 0.15)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#880000',
                    backgroundColor: 'rgba(136, 0, 0, 0.05)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: false },
                y: {
                    grid: { color: 'rgba(255, 0, 0, 0.08)' },
                    ticks: { color: '#ff3333', font: { size: 9, family: 'Outfit', weight: 'bold' } }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#121212', font: { size: 10, family: 'Outfit', weight: 'bold' } }
                }
            }
        }
    });
}

// Load dataset structure & load historical database
async function loadDatasetMeta() {
    try {
        const res = await fetch("/api/datasets");
        const datasets = await res.json();
        const currentMeta = datasets.find(d => d.id === currentDataset);
        
        // Update badge UI
        updateAIBadge(currentMeta.trained);
        
        // Generate Cards UI shell
        const container = document.getElementById("sensor-cards-container");
        container.innerHTML = "";
        
        currentMeta.features.forEach(feat => {
            const m = sensorMeta[feat] || sensorMeta["value"];
            const cardHtml = `
                <div id="card-${feat}" class="sensor-card glass-card">
                    <div class="sensor-card-header">
                        <div class="sensor-info">
                            <h4>${m.name}</h4>
                            <span id="status-${feat}" class="sensor-status-label status-normal">Normal</span>
                        </div>
                        <div class="sensor-icon-wrapper">
                            <i class="fa-solid ${m.icon}"></i>
                        </div>
                    </div>
                    <div class="sensor-value-display">
                        <span id="value-${feat}" class="sensor-value">--.-</span>
                        <span class="sensor-unit">${m.unit}</span>
                    </div>
                    <div class="sensor-gauges">
                        <div class="gauge-item">
                            <div class="gauge-circle-container">
                                <svg class="gauge-svg" width="60" height="60">
                                    <circle class="gauge-circle-bg" cx="30" cy="30" r="26"></circle>
                                    <circle id="health-ring-${feat}" class="gauge-circle-fill" cx="30" cy="30" r="26" stroke="#ff0000" stroke-dasharray="163" stroke-dashoffset="163"></circle>
                                </svg>
                                <span id="health-text-${feat}" class="gauge-text">--%</span>
                            </div>
                            <span class="gauge-label">Health</span>
                        </div>
                        <div class="gauge-item">
                            <div class="gauge-circle-container">
                                <svg class="gauge-svg" width="60" height="60">
                                    <circle class="gauge-circle-bg" cx="30" cy="30" r="26"></circle>
                                    <circle id="acc-ring-${feat}" class="gauge-circle-fill" cx="30" cy="30" r="26" stroke="#ff0000" stroke-dasharray="163" stroke-dashoffset="163"></circle>
                                </svg>
                                <span id="acc-text-${feat}" class="gauge-text">--%</span>
                            </div>
                            <span class="gauge-label">Accuracy</span>
                        </div>
                    </div>
                </div>
            `;
            container.insertAdjacentHTML("beforeend", cardHtml);
        });

        // Hide/Show injector sections
        if (currentDataset === "industrial") {
            document.getElementById("industrial-injectors").classList.remove("hidden");
            document.getElementById("single-sensor-injectors").classList.add("hidden");
        } else {
            document.getElementById("industrial-injectors").classList.add("hidden");
            document.getElementById("single-sensor-injectors").classList.remove("hidden");
        }

        // Initialize Live Chart Datasets
        liveChart.data.labels = [];
        liveChart.data.datasets = [];
        
        currentMeta.features.forEach((feat, idx) => {
            const m = sensorMeta[feat] || sensorMeta["value"];
            const colors = ["#ff0000", "#ff3333", "#aa0000", "#ff6666"];
            const col = colors[idx % colors.length];
            
            liveChart.data.datasets.push({
                label: m.name,
                data: [],
                borderColor: col,
                borderWidth: 1.5,
                pointRadius: 2,
                pointBackgroundColor: col,
                pointBorderColor: col,
                fill: false,
                tension: 0.15
            });
        });
        
        liveChart.update();

        // Fetch historical data
        const histRes = await fetch(`/api/historical?dataset=${currentDataset}&limit=40`);
        const history = await histRes.json();
        
        history.forEach(point => {
            const timeStr = new Date(point.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            liveChart.data.labels.push(timeStr);
            
            currentMeta.features.forEach((feat, idx) => {
                liveChart.data.datasets[idx].data.push(point.values[feat]);
            });
        });
        liveChart.update();

    } catch (e) {
        console.error("Failed to load dataset details", e);
    }
}

// Change dataset selection
function changeDataset(datasetId) {
    currentDataset = datasetId;
    resetSimulator();
    loadDatasetMeta();
}

// Toggle Streaming
function toggleStream() {
    const btn = document.getElementById("stream-toggle");
    if (isStreaming) {
        clearInterval(streamInterval);
        isStreaming = false;
        btn.innerHTML = `<i class="fa-solid fa-play"></i> Start Stream`;
        btn.classList.remove("danger");
        btn.classList.add("primary");
    } else {
        isStreaming = true;
        btn.innerHTML = `<i class="fa-solid fa-pause"></i> Pause Stream`;
        btn.classList.remove("primary");
        btn.classList.add("danger");
        
        streamInterval = setInterval(fetchTelemetry, 1500);
    }
}

// Fetch telemetry reading
async function fetchTelemetry() {
    try {
        const pulse = document.getElementById("update-pulse");
        pulse.classList.add("active");
        setTimeout(() => pulse.classList.remove("active"), 300);

        const res = await fetch(`/api/simulator/read?dataset=${currentDataset}`);
        const data = await res.json();
        
        const timeStr = new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        // Update charts
        const features = Object.keys(data.values);
        
        // Push label
        liveChart.data.labels.push(timeStr);
        if (liveChart.data.labels.length > 50) {
            liveChart.data.labels.shift();
        }

        // Push dataset values
        features.forEach((feat, idx) => {
            const val = data.values[feat];
            liveChart.data.datasets[idx].data.push(val);
            if (liveChart.data.datasets[idx].data.length > 50) {
                liveChart.data.datasets[idx].data.shift();
            }
            
            // Render numerical display card
            document.getElementById(`value-${feat}`).innerText = val.toFixed(1);
            
            // Compute circles
            const health = data.ai_health[feat];
            const accuracy = data.ai_accuracy[feat];
            
            updateGaugeRing(`health-ring-${feat}`, `health-text-${feat}`, health);
            updateGaugeRing(`acc-ring-${feat}`, `acc-text-${feat}`, accuracy);
            
            // Card visual level
            const cardEl = document.getElementById(`card-${feat}`);
            const statusEl = document.getElementById(`status-${feat}`);
            
            cardEl.className = "sensor-card glass-card";
            if (health < 40 || accuracy < 40) {
                cardEl.classList.add("danger");
                statusEl.innerText = "Critical Fault";
                statusEl.className = "sensor-status-label status-critical";
            } else if (health < 75 || accuracy < 75) {
                cardEl.classList.add("warning");
                statusEl.innerText = "Degraded / Drift";
                statusEl.className = "sensor-status-label status-warning";
            } else {
                statusEl.innerText = "Normal";
                statusEl.className = "sensor-status-label status-normal";
            }
        });

        // Set point highlight for anomalies
        const isAnom = data.ai_anomaly === 1;
        const lastIdx = liveChart.data.labels.length - 1;
        
        features.forEach((feat, idx) => {
            const dataset = liveChart.data.datasets[idx];
            if (isAnom) {
                dataset.pointRadius = dataset.pointRadius || [];
                dataset.pointBackgroundColor = dataset.pointBackgroundColor || [];
                dataset.pointBorderColor = dataset.pointBorderColor || [];
                
                // Set last point properties
                while (dataset.pointRadius.length < dataset.data.length - 1) {
                    dataset.pointRadius.push(2);
                    dataset.pointBackgroundColor.push(dataset.borderColor);
                    dataset.pointBorderColor.push(dataset.borderColor);
                }
                dataset.pointRadius.push(7);
                dataset.pointBackgroundColor.push("#000000");
                dataset.pointBorderColor.push("#ff0000");
            } else {
                if (Array.isArray(dataset.pointRadius)) {
                    dataset.pointRadius.push(2);
                    dataset.pointBackgroundColor.push(dataset.borderColor);
                    dataset.pointBorderColor.push(dataset.borderColor);
                    if (dataset.pointRadius.length > 50) {
                        dataset.pointRadius.shift();
                        dataset.pointBackgroundColor.shift();
                        dataset.pointBorderColor.shift();
                    }
                }
            }
        });
        
        liveChart.update();

        // Handle Event Log
        handleAnomalyTransitions(data);
        
        // Update general AI active badge in case model trained status changed
        updateAIBadge(data.models_trained);

    } catch (e) {
        console.error("Error fetching telemetry:", e);
    }
}

// Update Gauge Progress Arc
function updateGaugeRing(ringId, textId, value) {
    const ring = document.getElementById(ringId);
    const text = document.getElementById(textId);
    if (!ring || !text) return;

    // Circumference is 163 (2 * pi * r=26)
    const circ = 163;
    const valPercent = Math.max(0, Math.min(100, value));
    const offset = circ - (valPercent / 100) * circ;
    
    ring.style.strokeDashoffset = offset;
    text.innerText = `${Math.round(valPercent)}%`;

    // Colors based on severity (All red theme)
    if (valPercent >= 80) {
        ring.setAttribute("stroke", "#ff0000"); // Solid Red
    } else if (valPercent >= 50) {
        ring.setAttribute("stroke", "#aa0000"); // Medium alert red
    } else {
        ring.setAttribute("stroke", "#440000"); // Dim red (failed)
    }
}

// Log Transitions and format AI suggestions
function handleAnomalyTransitions(data) {
    const isAnom = data.ai_anomaly === 1;
    
    if (isAnom && !wasAnomalous) {
        // Normal -> Anomaly transition
        addLogEvent(data);
    } else if (isAnom && wasAnomalous) {
        // Check if anomaly profiles changed (e.g. new sensors suspect)
        const currentSecs = new Date().getSeconds();
        if (currentSecs % 10 === 0) { // Throttle updates slightly
            addLogEvent(data);
        }
    }
    wasAnomalous = isAnom;
}

function addLogEvent(data) {
    const body = document.getElementById("anomaly-log-body");
    
    // Clear placeholder message if it exists
    if (body.querySelector(".empty-log-msg")) {
        body.innerHTML = "";
    }
    
    // Identify suspected sensors (accuracy < 80% or health < 80%)
    const suspect = [];
    Object.keys(data.ai_health).forEach(f => {
        if (data.ai_health[f] < 80 || data.ai_accuracy[f] < 85) {
            suspect.push(f);
        }
    });

    if (suspect.length === 0) {
        // Fallback to highest prediction error
        if (data.model_details && data.model_details.autoencoder_error) {
            const maxFeat = Object.keys(data.model_details.autoencoder_error).reduce((a, b) => 
                data.model_details.autoencoder_error[a] > data.model_details.autoencoder_error[b] ? a : b
            );
            suspect.push(maxFeat);
        } else {
            suspect.push(Object.keys(data.values)[0]);
        }
    }

    const timestamp = new Date().toLocaleTimeString();
    const suspectText = suspect.map(s => `<span class="suspect-badge">${s}</span>`).join(" ");
    
    let description = "Unusual pattern flagged in sensor array.";
    let level = "warning";
    let rec = "Monitor system parameters.";

    // Custom recommendations based on dataset and injected faults
    if (currentDataset === "industrial") {
        if (suspect.includes("vibration") && data.ai_health.vibration < 80) {
            description = `Vibration level elevated: ${data.values.vibration.toFixed(2)}g`;
            level = data.ai_health.vibration < 50 ? "critical" : "warning";
            rec = "Bearing friction/wear detected. Schedule mechanical lubrication and check alignment.";
        } else if (suspect.includes("pressure") && data.values.pressure < 35.0) {
            description = `Sudden pressure drop detected: ${data.values.pressure.toFixed(1)} PSI`;
            level = "critical";
            rec = "Potential fluid/gas leak. Verify pipe integrity and pressure seals immediately.";
        } else if (suspect.includes("temperature") && data.values.temperature > 85.0) {
            description = `Temperature critical: ${data.values.temperature.toFixed(1)}°C`;
            level = "critical";
            rec = "Cooling system failure/airflow block. Inspect cooling fan operation and clear dust.";
        } else if (suspect.includes("temperature") && data.ai_accuracy.temperature < 80) {
            description = "Temperature sensor signal drifting (uncalibrated)";
            level = "warning";
            rec = "Sensor accuracy is compromised. Plan recalibration of RTD temperature transmitter.";
        } else if (suspect.includes("vibration") && data.ai_accuracy.vibration < 40) {
            description = "Vibration sensor signal flatlined / dead";
            level = "critical";
            rec = "Physical failure of accelerometer. Inspect sensor connection wiring or replace sensor.";
        } else if (suspect.includes("power") && data.values.power > 25.0) {
            description = `High current/power surge: ${data.values.power.toFixed(1)} kW`;
            level = "warning";
            rec = "Transient motor overload. Verify electrical load profile and check windings.";
        }
    } else {
        const val = data.values.value;
        description = `Temperature anomaly detected: ${val.toFixed(1)}°F`;
        level = "warning";
        rec = "Process variance exceeded confidence limits. Review heating loop control configurations.";
    }

    const rowHtml = `
        <tr>
            <td>${timestamp}</td>
            <td><span class="log-level level-${level}">${level}</span></td>
            <td>${description}</td>
            <td>${suspectText}</td>
            <td><strong>${rec}</strong></td>
        </tr>
    `;

    body.insertAdjacentHTML("afterbegin", rowHtml);
    
    // Cap log events
    while (body.rows.length > 30) {
        body.deleteRow(-1);
    }
}

// Update UI badges
function updateAIBadge(isTrained) {
    const badge = document.getElementById("ai-model-status");
    if (isTrained) {
        badge.className = "ai-badge badge-success";
        badge.querySelector(".status-text").innerText = "AI Trained (GPU)";
    } else {
        badge.className = "ai-badge badge-warning";
        badge.querySelector(".status-text").innerText = "AI Heuristics Active";
    }
}

// Inject faults
async function injectFault(faultType, element) {
    try {
        const res = await fetch("/api/simulator/inject", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset: currentDataset, type: faultType })
        });
        const msg = await res.json();
        
        // Visual feedback
        if (element && !element.id.includes("reset")) {
            element.classList.add("active");
            currentFaults.add(faultType);
        }
    } catch (e) {
        console.error("Failed to inject fault:", e);
    }
}

// Reset Simulator
async function resetSimulator() {
    try {
        await fetch("/api/simulator/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset: currentDataset, type: "reset" })
        });
        
        // Reset visual active injector buttons
        document.querySelectorAll(".inject-btn").forEach(btn => btn.classList.remove("active"));
        currentFaults.clear();
        
        // Clear main chart points
        liveChart.data.datasets.forEach(dataset => {
            dataset.pointRadius = 2;
            dataset.pointBackgroundColor = dataset.borderColor;
            dataset.pointBorderColor = dataset.borderColor;
        });
        liveChart.update();
        wasAnomalous = false;
        
    } catch (e) {
        console.error("Failed to reset simulator:", e);
    }
}

// Train ML Model Pipeline
async function startTraining() {
    const btn = document.getElementById("train-btn");
    const epochs = parseInt(document.getElementById("train-epochs").value) || 15;
    const batch = parseInt(document.getElementById("train-batch").value) || 64;
    
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Initializing...`;

    try {
        const res = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset: currentDataset, epochs: epochs, batch_size: batch })
        });
        
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail);
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-play-circle"></i> Train ML Models`;
            return;
        }

        // Show progress box
        document.getElementById("training-progress-container").classList.remove("hidden");
        
        // Reset loss curves
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.data.datasets[1].data = [];
        lossChart.update();
        
        // Start polling
        trainingPollInterval = setInterval(pollTrainingStatus, 1000);

    } catch (e) {
        console.error("Failed to start training:", e);
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-play-circle"></i> Train ML Models`;
    }
}

// Poll Training Progress
async function pollTrainingStatus() {
    try {
        const res = await fetch("/api/train/status");
        const status = await res.json();
        
        const prgBar = document.getElementById("training-progress-bar");
        const epochLbl = document.getElementById("training-epoch-label");
        const speedLbl = document.getElementById("training-speed-label");
        const msgEl = document.getElementById("training-status-message");
        const etaEl = document.getElementById("training-eta");
        const devEl = document.getElementById("training-device");
        
        // Update stats
        epochLbl.innerText = `Epoch ${status.current_epoch}/${status.total_epochs}`;
        speedLbl.innerText = status.training_speed || "Pending...";
        msgEl.innerText = status.message;
        etaEl.innerText = status.eta_seconds > 0 ? `${status.eta_seconds}s` : "--";
        devEl.innerText = status.device_used.toUpperCase();

        const pct = status.total_epochs > 0 ? (status.current_epoch / status.total_epochs) * 100 : 0;
        prgBar.style.width = `${pct}%`;

        // Update Loss Chart
        if (status.train_loss.length > lossChart.data.labels.length) {
            lossChart.data.labels = Array.from({ length: status.train_loss.length }, (_, i) => i + 1);
            lossChart.data.datasets[0].data = status.train_loss;
            lossChart.data.datasets[1].data = status.val_loss;
            lossChart.update();
        }

        if (!status.is_training) {
            // Training finished
            clearInterval(trainingPollInterval);
            
            const btn = document.getElementById("train-btn");
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-play-circle"></i> Train ML Models`;
            
            updateAIBadge(true);
            
            // Clean up status message slowly
            setTimeout(() => {
                msgEl.innerText = "Ready for new training run.";
            }, 5000);
        }

    } catch (e) {
        console.error("Failed to poll training status:", e);
    }
}

/**
 * Sentinel//Risk - Fintech Real-Time Fraud Telemetry Client
 * Vanilla JavaScript (Zero build step required)
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const apiStatusPill = document.getElementById('api-status-pill');
  const apiStatusText = document.getElementById('api-status-text');
  const kpiModel = document.getElementById('kpi-model');
  const kpiPrAuc = document.getElementById('kpi-prauc');
  const kpiRocAuc = document.getElementById('kpi-rocauc');
  const kpiThreshold = document.getElementById('kpi-threshold');
  const kpiSavings = document.getElementById('kpi-savings');
  const kpiLatency = document.getElementById('kpi-latency');
  const kpiDriftStatus = document.getElementById('kpi-drift-status');
  const kpiDriftSub = document.getElementById('kpi-drift-sub');

  const presetsContainer = document.getElementById('presets-container');
  const transactionForm = document.getElementById('transaction-form');
  const submitBtn = document.getElementById('submit-btn');

  const gaugeFill = document.getElementById('gauge-fill');
  const gaugeNeedle = document.getElementById('gauge-needle');
  const gaugeScoreVal = document.getElementById('gauge-score-value');
  const decisionTierBadge = document.getElementById('decision-tier-badge');
  const recommendedActionText = document.getElementById('recommended-action-text');
  const latencyVal = document.getElementById('latency-value');
  const thresholdUsedVal = document.getElementById('threshold-used-value');
  const txIdBadge = document.getElementById('tx-id-badge');
  const shapBarsList = document.getElementById('shap-bars-list');

  const benchmarkTbody = document.getElementById('benchmark-tbody');
  const driftTbody = document.getElementById('drift-tbody');
  const auditTbody = document.getElementById('audit-tbody');

  const thresholdOverrideSlider = document.getElementById('input-threshold-override');
  const thresholdOverrideVal = document.getElementById('threshold-override-val');
  const refreshAllBtn = document.getElementById('refresh-all-btn');
  const refreshAuditBtn = document.getElementById('refresh-audit-btn');

  // Application State
  let activeBlockThreshold = 0.02;
  let activeReviewThreshold = 0.009;
  let customThresholdOverride = null;

  // --------------------------------------------------------------------------
  // API Fetch Utilities
  // --------------------------------------------------------------------------
  async function fetchHealth() {
    try {
      const res = await fetch('/health');
      if (!res.ok) throw new Error('Health check failed');
      const data = await res.json();

      apiStatusText.textContent = 'SYSTEM OPERATIONAL';
      apiStatusPill.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      apiStatusPill.style.background = 'rgba(16, 185, 129, 0.12)';
      apiStatusPill.style.color = '#10b981';

      kpiModel.textContent = data.active_model.replace(' (Calibrated)', '');
      activeBlockThreshold = data.active_threshold;
      activeReviewThreshold = data.review_threshold;
      kpiThreshold.textContent = `${(activeBlockThreshold * 100).toFixed(1)}% (${activeBlockThreshold})`;

      thresholdOverrideSlider.value = activeBlockThreshold;
      thresholdOverrideVal.textContent = `Model Default (${activeBlockThreshold})`;
    } catch (err) {
      console.warn('API Health Warning:', err);
      apiStatusText.textContent = 'OFFLINE / CONNECTING';
      apiStatusPill.style.borderColor = 'rgba(244, 63, 94, 0.4)';
      apiStatusPill.style.background = 'rgba(244, 63, 94, 0.12)';
      apiStatusPill.style.color = '#f43f5e';
    }
  }

  async function fetchPresets() {
    try {
      const res = await fetch('/sample-transactions');
      const presets = await res.json();
      presetsContainer.innerHTML = '';

      presets.forEach((preset, index) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'preset-btn';
        btn.innerHTML = `
          <div class="preset-badge badge-${preset.badge_color}">${preset.badge}</div>
          <div class="preset-name">${preset.name}</div>
        `;
        btn.addEventListener('click', () => {
          populateForm(preset.payload);
          scoreTransaction();
        });
        presetsContainer.appendChild(btn);
      });
    } catch (err) {
      console.error('Failed to load presets:', err);
    }
  }

  async function fetchMetrics() {
    try {
      const res = await fetch('/metrics');
      const data = await res.json();

      kpiPrAuc.textContent = data.best_pr_auc.toFixed(4);
      kpiRocAuc.textContent = `ROC-AUC: ${data.best_roc_auc.toFixed(4)}`;
      kpiSavings.textContent = `Net Savings: $${data.net_cost_savings.toLocaleString()} (${data.savings_pct}%)`;

      renderBenchmarkTable(data.models);
    } catch (err) {
      console.error('Failed to load metrics:', err);
    }
  }

  async function fetchDrift() {
    try {
      const res = await fetch('/drift');
      const data = await res.json();

      kpiDriftStatus.textContent = data.overall_status;
      if (data.overall_status === 'HEALTHY') {
        kpiDriftStatus.className = 'kpi-value highlight-emerald';
      } else if (data.overall_status === 'WARNING') {
        kpiDriftStatus.className = 'kpi-value highlight-amber';
      } else {
        kpiDriftStatus.className = 'kpi-value highlight-rose';
      }
      kpiDriftSub.textContent = `${data.stable_count} Stable, ${data.moderate_drift_count} Mod, ${data.high_drift_count} High`;

      renderDriftTable(data.features || []);
    } catch (err) {
      console.error('Failed to load drift data:', err);
    }
  }

  async function fetchAuditLog() {
    try {
      const res = await fetch('/predictions/recent');
      const records = await res.json();
      renderAuditTable(records);
    } catch (err) {
      console.error('Failed to load audit log:', err);
    }
  }

  // --------------------------------------------------------------------------
  // Form & Scoring Handlers
  // --------------------------------------------------------------------------
  function populateForm(payload) {
    document.getElementById('input-amt').value = payload.TransactionAmt;
    document.getElementById('input-product').value = payload.ProductCD || 'W';
    document.getElementById('input-card1').value = payload.card1 || 10000;
    document.getElementById('input-card4').value = payload.card4 || 'visa';
    document.getElementById('input-card6').value = payload.card6 || 'debit';
    document.getElementById('input-p-email').value = payload.P_emaildomain || '';
    document.getElementById('input-r-email').value = payload.R_emaildomain || '';
    document.getElementById('input-device').value = payload.DeviceType || 'desktop';
    document.getElementById('input-hour').value = payload.Transaction_Hour ?? 14;
  }

  function getFormPayload() {
    const amt = parseFloat(document.getElementById('input-amt').value);
    const product = document.getElementById('input-product').value;
    const card1 = parseInt(document.getElementById('input-card1').value, 10);
    const card4 = document.getElementById('input-card4').value;
    const card6 = document.getElementById('input-card6').value;
    const p_email = document.getElementById('input-p-email').value.trim() || null;
    const r_email = document.getElementById('input-r-email').value.trim() || null;
    const device = document.getElementById('input-device').value;
    const hour = parseInt(document.getElementById('input-hour').value, 10);

    const payload = {
      TransactionAmt: amt,
      ProductCD: product,
      card1: card1,
      card4: card4,
      card6: card6,
      P_emaildomain: p_email,
      R_emaildomain: r_email,
      DeviceType: device,
      Transaction_Hour: isNaN(hour) ? 14 : hour,
    };

    if (customThresholdOverride !== null) {
      payload.custom_threshold = customThresholdOverride;
    }

    return payload;
  }

  async function scoreTransaction() {
    const payload = getFormPayload();
    submitBtn.classList.add('loading');

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorDetail = await res.json();
        throw new Error(errorDetail.detail || 'Prediction failed');
      }

      const decision = await res.json();
      renderDecision(decision);
      fetchAuditLog(); // auto-refresh audit table
    } catch (err) {
      alert(`Scoring Error: ${err.message}`);
    } finally {
      submitBtn.classList.remove('loading');
    }
  }

  transactionForm.addEventListener('submit', (e) => {
    e.preventDefault();
    scoreTransaction();
  });

  // Threshold Override Slider Listener
  thresholdOverrideSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    customThresholdOverride = val;
    thresholdOverrideVal.textContent = `Custom: ${(val * 100).toFixed(1)}% (${val.toFixed(3)})`;
  });

  // --------------------------------------------------------------------------
  // Rendering UI Components
  // --------------------------------------------------------------------------
  function renderDecision(data) {
    const prob = data.fraud_probability;
    const pct = (prob * 100).toFixed(2);
    gaugeScoreVal.textContent = `${pct}%`;

    // Visual angle mapping for semi-circular gauge:
    // Scale calibrated probability [0.0 .. 0.15] to gauge needle arc [-90deg .. +90deg]
    // A score of 0.05 is already 2.5x higher than the optimal block threshold
    const normalizedScore = Math.min(1.0, prob / 0.10);
    const needleDeg = -90 + normalizedScore * 180;
    gaugeNeedle.style.transform = `rotate(${needleDeg}deg)`;

    // Arc stroke dashoffset: circumference ~ 251.2
    const totalArc = 251.2;
    const offset = totalArc - (normalizedScore * totalArc);
    gaugeFill.style.strokeDashoffset = offset;

    // Decision Tier & Colors
    decisionTierBadge.className = 'decision-tier-badge';
    let strokeColor = '#10b981';

    if (data.risk_tier === 'HIGH') {
      decisionTierBadge.textContent = 'BLOCK TRANSACTION';
      decisionTierBadge.classList.add('tier-block');
      strokeColor = '#f43f5e';
      recommendedActionText.textContent = 'Exceeds financial loss threshold. Payment authorization declined.';
    } else if (data.risk_tier === 'MEDIUM') {
      decisionTierBadge.textContent = 'MANUAL REVIEW / 3DS';
      decisionTierBadge.classList.add('tier-review');
      strokeColor = '#f59e0b';
      recommendedActionText.textContent = 'Suspicious velocity or device anomalies. Trigger step-up authentication.';
    } else {
      decisionTierBadge.textContent = 'APPROVE PAYMENT';
      decisionTierBadge.classList.add('tier-approve');
      strokeColor = '#10b981';
      recommendedActionText.textContent = 'Low fraud risk profile. Cleared without friction.';
    }
    gaugeFill.style.stroke = strokeColor;

    // Chips & Metadata
    latencyVal.textContent = `${data.latency_ms.toFixed(1)} ms`;
    thresholdUsedVal.textContent = `${data.threshold_used.toFixed(3)}`;
    txIdBadge.textContent = `ID: ${data.transaction_id}`;
    kpiLatency.textContent = `${data.latency_ms.toFixed(1)} ms`;

    // Render SHAP Factors
    renderShapFactors(data.top_risk_factors);
  }

  function renderShapFactors(factors) {
    if (!factors || factors.length === 0) {
      shapBarsList.innerHTML = `<div class="empty-state"><p>No explanation drivers available.</p></div>`;
      return;
    }

    const maxAbsImpact = Math.max(...factors.map(f => Math.abs(f.shap_impact)), 0.01);
    shapBarsList.innerHTML = '';

    factors.forEach((f) => {
      const isPos = f.direction === 'increases_risk';
      const widthPct = Math.min(100, Math.max(8, (Math.abs(f.shap_impact) / maxAbsImpact) * 100));
      const badgeClass = isPos ? 'impact-pos' : 'impact-neg';
      const barClass = isPos ? 'pos' : 'neg';
      const sign = isPos ? '+' : '';

      const card = document.createElement('div');
      card.className = 'shap-card';
      card.innerHTML = `
        <div class="shap-card-header">
          <span class="shap-feature-name">${f.feature} <small style="color:var(--text-muted)">(${f.raw_value})</small></span>
          <span class="shap-impact-badge ${badgeClass}">${sign}${f.shap_impact.toFixed(4)}</span>
        </div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${barClass}" style="width: ${widthPct}%;"></div>
        </div>
        <div class="shap-explanation">${f.explanation}</div>
      `;
      shapBarsList.appendChild(card);
    });
  }

  function renderBenchmarkTable(models) {
    benchmarkTbody.innerHTML = '';
    for (const [name, m] of Object.entries(models)) {
      const isProduction = name.includes('Calibrated');
      const tr = document.createElement('tr');
      if (isProduction) tr.style.background = 'rgba(16, 185, 129, 0.05)';

      const savingsDisplay = m.net_cost_savings !== undefined ? `$${m.net_cost_savings.toLocaleString()} (${m.savings_pct}%)` : '--';

      tr.innerHTML = `
        <td class="strong">${isProduction ? '★ ' : ''}${name}</td>
        <td class="mono ${isProduction ? 'highlight-emerald' : ''}">${m.pr_auc?.toFixed(4) || '--'}</td>
        <td class="mono">${m.roc_auc?.toFixed(4) || '--'}</td>
        <td class="mono">${m.brier_score?.toFixed(4) || '--'}</td>
        <td class="mono">${m.f1?.toFixed(4) || '--'}</td>
        <td class="mono ${isProduction ? 'highlight-cyan' : ''}">${savingsDisplay}</td>
      `;
      benchmarkTbody.appendChild(tr);
    }
  }

  function renderDriftTable(features) {
    driftTbody.innerHTML = '';
    features.slice(0, 10).forEach((f) => {
      const tr = document.createElement('tr');
      let badgeClass = 'badge-stable';
      if (f.status === 'MODERATE_DRIFT') badgeClass = 'badge-moderate';
      if (f.status === 'HIGH_DRIFT') badgeClass = 'badge-high';

      tr.innerHTML = `
        <td class="strong mono">${f.feature}</td>
        <td class="mono">${f.psi?.toFixed(4) || '0.000'}</td>
        <td class="mono">${f.p_value !== undefined ? f.p_value.toFixed(4) : '--'}</td>
        <td class="mono">${f.ref_mean?.toFixed(2) || '--'}</td>
        <td class="mono">${f.target_mean?.toFixed(2) || '--'}</td>
        <td><span class="status-badge ${badgeClass}">${f.status}</span></td>
      `;
      driftTbody.appendChild(tr);
    });
  }

  function renderAuditTable(records) {
    auditTbody.innerHTML = '';
    if (!records || records.length === 0) {
      auditTbody.innerHTML = `<tr><td colspan="10" class="text-center" style="padding:20px; color:var(--text-muted)">No transactions recorded in audit history.</td></tr>`;
      return;
    }

    records.forEach((r) => {
      const tr = document.createElement('tr');
      let tierBadge = 'badge-low-risk';
      if (r.risk_tier === 'HIGH') tierBadge = 'badge-high-risk';
      if (r.risk_tier === 'MEDIUM') tierBadge = 'badge-medium-risk';

      const shortTime = r.timestamp ? r.timestamp.substring(11, 19) + ' UTC' : '--';

      tr.innerHTML = `
        <td class="mono strong">${r.transaction_id}</td>
        <td class="mono">${shortTime}</td>
        <td class="mono strong">$${r.amount.toFixed(2)}</td>
        <td class="mono">${r.card1}</td>
        <td class="mono">${r.product_cd}</td>
        <td class="mono strong">${(r.fraud_probability * 100).toFixed(2)}%</td>
        <td><span class="status-badge ${tierBadge}">${r.risk_tier}</span></td>
        <td class="strong">${r.action}</td>
        <td class="mono">${r.latency_ms.toFixed(1)} ms</td>
        <td class="mono" style="color:var(--text-muted)">${r.top_factor || 'None'}</td>
      `;
      auditTbody.appendChild(tr);
    });
  }

  // Refresh Buttons
  refreshAllBtn.addEventListener('click', () => {
    fetchHealth();
    fetchMetrics();
    fetchDrift();
    fetchAuditLog();
  });

  refreshAuditBtn.addEventListener('click', () => {
    fetchAuditLog();
  });

  // Initial Load Pipeline
  fetchHealth();
  fetchPresets();
  fetchMetrics();
  fetchDrift();
  fetchAuditLog();
});

import { useEffect, useMemo, useState } from 'react'
import './App.css'

const DEFAULT_API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')
const API_URL_STORAGE_KEY = 'fraudguard.apiBaseUrl'
const REQUIRED_COLUMNS = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024

const navigation = [
  { id: 'dashboard', label: 'Overview', icon: 'grid', group: 'Workspace' },
  { id: 'analysis', label: 'Transaction analysis', icon: 'scan', group: 'Workspace' },
  { id: 'batch', label: 'Batch prediction', icon: 'upload', group: 'Workspace' },
  { id: 'history', label: 'Transaction history', icon: 'clock', group: 'Workspace' },
  { id: 'monitoring', label: 'Fraud monitoring', icon: 'pulse', group: 'Intelligence' },
  { id: 'performance', label: 'Model performance', icon: 'chart', group: 'Intelligence' },
  { id: 'settings', label: 'Settings', icon: 'settings', group: 'System' },
]

function getApiBaseUrl() {
  try {
    return (localStorage.getItem(API_URL_STORAGE_KEY) || DEFAULT_API_BASE_URL).replace(/\/+$/, '')
  } catch {
    return DEFAULT_API_BASE_URL
  }
}

async function apiRequest(path, options = {}) {
  const apiBaseUrl = getApiBaseUrl()
  let response
  try {
    response = await fetch(`${apiBaseUrl}${path}`, options)
  } catch {
    throw new Error(`Could not connect to the API at ${apiBaseUrl}. Check that the backend is running and VITE_API_BASE_URL is correct.`)
  }

  const contentType = response.headers.get('content-type') || ''
  const body = contentType.includes('application/json')
    ? await response.json()
    : await response.text()
  if (!response.ok) {
    const detail = typeof body === 'object' && body !== null ? body.detail : body
    const message = response.status === 503
      ? 'The requested model service or evaluation report is unavailable.'
      : typeof detail === 'string'
        ? detail
        : JSON.stringify(detail || `Request failed (${response.status}).`)
    throw new Error(`HTTP ${response.status}: ${message}`)
  }
  return body
}

function formatNumber(value) {
  return Number(value).toLocaleString('en-US')
}

function formatPercent(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`
}

function formatTimestamp(value) {
  const timestamp = new Date(value)
  return Number.isNaN(timestamp.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(timestamp)
}

function formatCurrentDate() {
  return new Intl.DateTimeFormat(undefined, {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date())
}

function validProbability(value) {
  return Number.isFinite(value) && value >= 0 && value <= 1
}

function riskLevel(probability) {
  if (probability >= 0.75) return 'high'
  if (probability >= 0.25) return 'medium'
  return 'low'
}

function Icon({ name, size = 18 }) {
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
    scan: <><path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m13-5v3a2 2 0 0 1-2 2h-3" /><path d="M7 12h10M12 7v10" /></>,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5" /><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
    chart: <><path d="M4 19V5m0 14h17" /><path d="m7 15 4-4 3 2 6-7" /><path d="M17 6h3v3" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    search: <><circle cx="10.8" cy="10.8" r="6.8" /><path d="m16 16 4.5 4.5" /></>,
    pulse: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6" /></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M6 9a7 7 0 0 1 12-2l2 2M4 15l2 2a7 7 0 0 0 12-2" /></>,
    download: <><path d="M12 4v11m-5-5 5 5 5-5" /><path d="M4 17v3h16v-3" /></>,
    settings: <><path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z" /><path d="m19.4 15 .1.1 1.1.9-1.2 2.1-1.4-.5a7.7 7.7 0 0 1-1.6.9l-.3 1.5h-2.4l-.3-1.5a7.7 7.7 0 0 1-1.6-.9l-1.4.5-1.2-2.1 1.1-.9a7.2 7.2 0 0 1 0-1.9l-1.1-.9 1.2-2.1 1.4.5a7.7 7.7 0 0 1 1.6-.9l.3-1.5h2.4l.3 1.5a7.7 7.7 0 0 1 1.6.9l1.4-.5 1.2 2.1-1.1.9a7.2 7.2 0 0 1 0 1.9Z" /></>,
    collapse: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M9 4v16m6-11-2.5 3L15 15" /></>,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 15 .9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15Z" /></>,
    file: <><path d="M13 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10z" /><path d="M13 3v7h7M8 14h8m-8 4h8" /></>,
  }
  return <svg aria-hidden="true" className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name] || paths.grid}</svg>
}

function Sidebar({ activePage, onNavigate, collapsed, onToggleCollapse }) {
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="brand">
        <div className="brand-logo" aria-hidden="true"><Icon name="spark" size={21} /></div>
        <div className="brand-copy"><strong>Fraud Intelligence</strong><small>FRAUDGUARD · ML PLATFORM</small></div>
      </div>
      <nav aria-label="Main navigation" className="side-nav">
        {['Workspace', 'Intelligence', 'System'].map((group) => (
          <div className="nav-group" key={group}>
            <p className="nav-group-title">{group}</p>
            {navigation.filter((item) => item.group === group).map((item) => (
              <button
                aria-current={activePage === item.id ? 'page' : undefined}
                className={`nav-link ${activePage === item.id ? 'active' : ''}`}
                key={item.id}
                onClick={() => onNavigate(item.id)}
                title={item.label}
                type="button"
              >
                <Icon name={item.icon} />
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        ))}
      </nav>
      <button
        aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
        className="collapse-toggle"
        onClick={onToggleCollapse}
        title={collapsed ? 'Expand navigation' : 'Collapse navigation'}
        type="button"
      >
        <Icon name="collapse" size={16} /><span>{collapsed ? 'Expand sidebar' : 'Collapse sidebar'}</span>
      </button>
      <div className="sidebar-footer">
        <div className="sidebar-footer-icon"><Icon name="pulse" size={17} /></div>
        <div className="sidebar-footer-copy"><strong>Candidate model</strong><small>Threshold is provisional</small></div>
      </div>
    </aside>
  )
}

function PageHeading({ eyebrow, title, description, action }) {
  return (
    <div className="page-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {action}
    </div>
  )
}

function Panel({ title, subtitle, action, className = '', children }) {
  return (
    <section className={`panel ${className}`}>
      {(title || action) && (
        <div className="panel-header">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <p>{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  )
}

function StatCard({ label, value, note, icon, tone = 'blue' }) {
  return (
    <article className="stat-card">
      <div className="stat-top"><span>{label}</span><span className={`stat-icon ${tone}`}><Icon name={icon} size={17} /></span></div>
      <strong className="stat-value">{value}</strong>
      <span className="stat-note">{note}</span>
    </article>
  )
}

function LoadingState({ label = 'Loading verified project data…' }) {
  return <div className="state-message"><span className="spinner" />{label}</div>
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="state-message error-state" role="alert">
      <span>{message}</span>
      {onRetry && <button className="text-button" onClick={onRetry} type="button">Try again</button>}
    </div>
  )
}

function EmptyState({ icon = 'file', title, children }) {
  return (
    <div className="empty-state">
      <span className="empty-icon"><Icon name={icon} size={22} /></span>
      <strong>{title}</strong>
      <p>{children}</p>
    </div>
  )
}

function Dashboard({ onNavigate }) {
  const [performance, setPerformance] = useState(null)
  const [recentHistory, setRecentHistory] = useState(null)
  const [fraudHistory, setFraudHistory] = useState(null)
  const [modelStatus, setModelStatus] = useState(null)
  const [error, setError] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [loading, setLoading] = useState(true)
  const [refreshCount, setRefreshCount] = useState(0)

  useEffect(() => {
    let alive = true
    Promise.allSettled([
      apiRequest('/model/performance'),
      apiRequest('/history?limit=5'),
      apiRequest('/history?fraud_status=fraud&limit=5'),
      apiRequest('/model/status'),
    ]).then(([performanceResult, historyResult, fraudResult, statusResult]) => {
      if (!alive) return
      if (performanceResult.status === 'fulfilled') {
        setPerformance(performanceResult.value)
      } else {
        setError(performanceResult.reason.message)
      }
      if (historyResult.status === 'fulfilled') {
        setRecentHistory(historyResult.value)
      } else {
        setHistoryError(historyResult.reason.message)
      }
      if (fraudResult.status === 'fulfilled') setFraudHistory(fraudResult.value)
      if (statusResult.status === 'fulfilled') setModelStatus(statusResult.value)
      setLoading(false)
    })
    return () => { alive = false }
  }, [refreshCount])

  function retry() {
    setError('')
    setHistoryError('')
    setLoading(true)
    setRefreshCount((count) => count + 1)
  }

  const report = performance?.test_evaluation
  const predictionCount = recentHistory?.total ?? 0
  const flaggedCount = fraudHistory?.total ?? 0
  const legitimateCount = Math.max(0, predictionCount - flaggedCount)
  const recordedFraudRate = predictionCount > 0 ? flaggedCount / predictionCount : null

  return (
    <>
      <PageHeading
        eyebrow="FRAUD INTELLIGENCE · OVERVIEW"
        title="Fraud intelligence overview"
        description="Your candidate model, saved prediction activity, and recorded offline evaluation."
        action={<span className="date-stamp"><Icon name="clock" size={15} />{formatCurrentDate()}</span>}
      />
      {loading && <LoadingState />}
      {error && <ErrorState message={error} onRetry={retry} />}
      {modelStatus && (
        <div className="overview-status-strip">
          <span className={`service-indicator ${modelStatus.model_artifacts_available ? 'online' : 'offline'}`} />
          <span>Candidate model</span>
          <strong className={modelStatus.model_artifacts_available ? 'text-green' : 'text-red'}>
            {modelStatus.model_artifacts_available ? 'Available' : 'Artifacts unavailable'}
          </strong>
          <span className="strip-divider" />
          <span>Prediction threshold</span>
          <strong className="text-amber">95% provisional</strong>
          <span className="strip-spacer" />
          <span className="snapshot-tag"><span /> Recorded evaluation</span>
        </div>
      )}
      {(report || recentHistory) && (
        <>
          <div className="stats-grid">
            <StatCard label="Predictions recorded" value={recentHistory ? formatNumber(predictionCount) : '—'} note="Persisted local history" icon="file" tone="blue" />
            <StatCard label="Model flagged" value={fraudHistory ? formatNumber(flaggedCount) : '—'} note="From saved predictions" icon="scan" tone="red" />
            <StatCard label="Recorded fraud rate" value={recordedFraudRate === null ? '—' : formatPercent(recordedFraudRate, 2)} note={predictionCount ? 'Saved predictions only' : 'Available after predictions'} icon="chart" tone="violet" />
            <StatCard label="Test-set F1" value={report ? report.f1.toFixed(3) : '—'} note="Offline evaluation · provisional threshold" icon="pulse" tone="green" />
          </div>
          <div className="content-grid dashboard-grid">
            <Panel title="Saved prediction distribution" subtitle="Actual predictions from the local history database">
              {predictionCount > 0 ? (
                <>
                  <div className="distribution-summary">
                    <div><span className="legend-dot safe-dot" />Not flagged <strong>{formatNumber(legitimateCount)}</strong></div>
                    <div><span className="legend-dot high-dot" />Model flagged <strong>{formatNumber(flaggedCount)}</strong></div>
                  </div>
                  <div className="stacked-bar" role="img" aria-label={`${formatNumber(flaggedCount)} flagged and ${formatNumber(legitimateCount)} not-flagged saved predictions`}>
                    <span className="bar-not-fraud" style={{ width: `${100 - (recordedFraudRate * 100)}%` }} />
                    <span className="bar-fraud" style={{ width: `${recordedFraudRate * 100}%` }} />
                  </div>
                  <div className="chart-caption"><span>0</span><span>Flagged: {formatPercent(recordedFraudRate, 2)}</span><span>{formatNumber(predictionCount)} predictions</span></div>
                </>
              ) : (
                <EmptyState icon="chart" title="No prediction distribution yet">
                  Once transactions are analyzed, saved prediction outcomes will appear here.
                </EmptyState>
              )}
              <p className="chart-footnote">Local recorded history only. This is not a live traffic feed.</p>
            </Panel>
            <Panel title="Offline model evaluation" subtitle="Genuine held-out test results · not production telemetry">
              {report ? (
                <>
                  <div className="metric-list">
                    <MetricBar label="Average precision" value={report.average_precision} color="blue" />
                    <MetricBar label="Precision" value={report.precision} color="violet" />
                    <MetricBar label="Recall" value={report.recall} color="green" />
                  </div>
                  <div className="threshold-note"><span className="status-dot amber" /><div><strong>0.95 decision threshold is provisional</strong><p>Threshold selection used validation data; alert-volume and error-cost criteria are not defined.</p></div></div>
                </>
              ) : <EmptyState title="Evaluation report unavailable">Check the model performance page or backend connection.</EmptyState>}
            </Panel>
          </div>
          <Panel className="activity-panel" title="Recent transaction activity" subtitle="Latest persisted predictions · not a real-time alert stream">
            {historyError && <ErrorState message={historyError} onRetry={retry} />}
            {!historyError && recentHistory?.items.length === 0 && (
              <EmptyState icon="clock" title="No predictions recorded yet">
                Successful single and batch predictions will appear here.
              </EmptyState>
            )}
            {!historyError && recentHistory?.items.length > 0 && (
              <div className="recent-activity-list">
                {recentHistory.items.map((record) => (
                  <div className="recent-activity-row" key={record.id}>
                    <span className={`activity-class-dot ${record.is_fraud ? 'fraud' : 'safe'}`} />
                    <div className="recent-activity-main">
                      <strong>{record.type.replaceAll('_', ' ')} <span>· {formatNumber(record.amount)} units</span></strong>
                      <small>{formatTimestamp(record.timestamp)} · {record.id.slice(0, 8)}</small>
                    </div>
                    <span className={`risk-pill ${record.fraud_probability == null ? 'neutral' : riskLevel(record.fraud_probability)}`}>
                      {record.is_fraud ? 'Model flagged' : 'Not flagged'}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <button className="text-button activity-link" onClick={() => onNavigate('history')} type="button">
              View transaction history <Icon name="arrow" size={14} />
            </button>
          </Panel>
          <div className="quick-actions">
            <button className="primary-button" onClick={() => onNavigate('analysis')} type="button"><Icon name="scan" size={16} /> Analyze a transaction <Icon name="arrow" size={16} /></button>
            <button className="secondary-button" onClick={() => onNavigate('performance')} type="button">Review model performance <Icon name="arrow" size={16} /></button>
          </div>
        </>
      )}
    </>
  )
}

function MetricBar({ label, value, color = 'blue' }) {
  return (
    <div className="metric-bar-row">
      <div className="metric-bar-label"><span>{label}</span><strong>{formatPercent(value, 1)}</strong></div>
      <div className="metric-track"><span className={color} style={{ width: `${Math.min(100, value * 100)}%` }} /></div>
    </div>
  )
}

function TransactionAnalysis() {
  const [form, setForm] = useState({
    step: '1',
    type: 'TRANSFER',
    amount: '1000',
    oldbalanceOrg: '5000',
    oldbalanceDest: '2000',
  })
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function handleChange(event) {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }))
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setResult(null)
    setError('')
    const numeric = ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest']
    for (const field of numeric) {
      if (!Number.isFinite(Number(form[field]))) {
        setError(`${field} must be a valid number.`)
        return
      }
    }
    if (Number(form.step) < 1 || !Number.isInteger(Number(form.step))) {
      setError('Transaction step must be a whole number greater than or equal to 1.')
      return
    }
    if (Number(form.amount) <= 0) {
      setError('Transaction amount must be greater than 0.')
      return
    }
    if (Number(form.oldbalanceOrg) < 0 || Number(form.oldbalanceDest) < 0) {
      setError('Balances cannot be negative.')
      return
    }

    setLoading(true)
    try {
      const prediction = await apiRequest('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          step: Number(form.step),
          amount: Number(form.amount),
          oldbalanceOrg: Number(form.oldbalanceOrg),
          oldbalanceDest: Number(form.oldbalanceDest),
        }),
      })
      setResult(prediction)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  const risk = result
    ? validProbability(result.fraud_probability)
      ? riskLevel(result.fraud_probability)
      : result.is_fraud
        ? 'high'
        : 'low'
    : 'low'
  return (
    <>
      <PageHeading
        eyebrow="AI TRANSACTION INTELLIGENCE"
        title="Transaction analysis"
        description="Evaluate a transaction with the trained decision-time model. Your features are processed by the existing backend pipeline."
        action={<span className="ai-tag"><Icon name="spark" size={14} /> Model-assisted risk estimate</span>}
      />
      <div className="content-grid analysis-grid">
        <Panel title="Transaction details" subtitle="Enter the five features expected by the trained model.">
          <form className="analysis-form" onSubmit={handleSubmit}>
            <label>Transaction type
              <select name="type" value={form.type} onChange={handleChange}>
                <option value="TRANSFER">Transfer</option><option value="CASH_OUT">Cash out</option><option value="PAYMENT">Payment</option><option value="CASH_IN">Cash in</option><option value="DEBIT">Debit</option>
              </select>
            </label>
            <label>Transaction step
              <input name="step" type="number" min="1" step="1" required value={form.step} onChange={handleChange} />
              <span className="field-help">Sequential time step recorded for this transaction.</span>
            </label>
            <label>Transaction amount
              <input name="amount" type="number" min="0.01" step="any" required value={form.amount} onChange={handleChange} />
              <span className="field-help">Amount in the units supplied by your source data.</span>
            </label>
            <div className="field-row">
              <label>Origin balance before
                <input name="oldbalanceOrg" type="number" min="0" step="any" required value={form.oldbalanceOrg} onChange={handleChange} />
                <span className="field-help">Origin account balance before the transaction.</span>
              </label>
              <label>Destination balance before
                <input name="oldbalanceDest" type="number" min="0" step="any" required value={form.oldbalanceDest} onChange={handleChange} />
                <span className="field-help">Destination account balance before the transaction.</span>
              </label>
            </div>
            {error && <div className="inline-error" role="alert">{error}</div>}
            <button className="primary-button full-button" disabled={loading} type="submit">
              {loading ? <><span className="button-spinner" />Analyzing transaction…</> : <>Analyze transaction <Icon name="arrow" size={17} /></>}
            </button>
            <p className="form-footnote">A model prediction is a risk estimate, not proof of fraud. The current decision threshold is provisional.</p>
          </form>
        </Panel>
        <Panel className="result-card" title="Prediction result" subtitle="Results are returned by the live prediction API.">
          {loading && <LoadingState label="The candidate model is scoring this transaction…" />}
          {!loading && !result && !error && <EmptyState icon="scan" title="Ready to analyze">Submit transaction details to see the model classification and estimated fraud probability.</EmptyState>}
          {!loading && error && !result && <EmptyState icon="pulse" title="Prediction unavailable">Correct the input or check the backend status in MLOps monitoring.</EmptyState>}
          {result && (
            <div className="prediction-result">
              <div className="prediction-result-top">
                <span className={`risk-pill ${risk}`}>{risk} risk</span>
                <span className="result-class">{result.is_fraud ? 'Flagged for review' : 'Not flagged'}</span>
              </div>
              {validProbability(result.fraud_probability) ? (
                <>
                  <div className={`probability-value ${risk}`}>{formatPercent(result.fraud_probability, 2)}<small>model-estimated fraud probability</small></div>
                  <div aria-label={`Fraud probability ${formatPercent(result.fraud_probability, 2)}`} className="probability-track" role="img"><span className={risk} style={{ width: `${result.fraud_probability * 100}%` }} /></div>
                </>
              ) : (
                <div className="probability-unavailable">The API did not return a valid fraud probability.</div>
              )}
              <div className="result-detail"><span>Model prediction</span><strong>{result.is_fraud ? 'Flagged for review' : 'Not flagged'}</strong></div>
              <div className="result-detail"><span>Returned decision threshold</span><strong>{formatPercent(result.threshold, 0)} · provisional</strong></div>
              <div className="threshold-note"><span className="status-dot blue" /><div><strong>Risk bands are visual guidance</strong><p>Low &lt;25%, medium 25–75%, high ≥75%. Classification follows the API threshold.</p></div></div>
            </div>
          )}
        </Panel>
      </div>
    </>
  )
}

function csvCell(value) {
  let text = String(value ?? '')
  if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`
  return `"${text.replaceAll('"', '""')}"`
}

function BatchAnalysis() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const visibleResults = useMemo(() => result?.predictions?.slice(0, 25) || [], [result])
  const flaggedResults = result?.predictions?.filter((row) => row.is_fraud === 1).length ?? 0

  function handleFile(event) {
    const selected = event.target.files?.[0] || null
    setResult(null)
    setError('')
    if (selected && !selected.name.toLowerCase().endsWith('.csv')) {
      setFile(null)
      setError('Choose a CSV file.')
    } else if (selected && selected.size > MAX_UPLOAD_BYTES) {
      setFile(null)
      setError('CSV uploads are limited to 5 MB.')
    } else {
      setFile(selected)
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (!file) {
      setError('Choose a CSV file to analyze.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await apiRequest('/predict/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'text/csv' },
        body: file,
      })
      setResult(response)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  function exportResults() {
    if (!result?.predictions?.length) return
    const columns = [...REQUIRED_COLUMNS, 'row_number', 'fraud_probability', 'is_fraud', 'threshold']
    const csv = [
      columns.map(csvCell).join(','),
      ...result.predictions.map((row) => columns.map((column) => csvCell(row[column])).join(',')),
    ].join('\r\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'fraud-analysis-results.csv'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <>
      <PageHeading eyebrow="BATCH SCORING" title="Batch prediction" description="Score transaction files through the existing server-side preprocessing and model pipeline." />
      <div className="batch-layout">
        <Panel title="Upload transaction data" subtitle="The API validates rows and reports invalid inputs without loading the PaySim dataset into the browser.">
          <form className="upload-form" onSubmit={handleSubmit}>
            <label className="upload-zone" htmlFor="batch-file">
              <span className="upload-icon"><Icon name="upload" size={22} /></span>
              <strong>{file ? file.name : 'Choose a CSV file'}</strong>
              <span>{file ? `${(file.size / 1024).toFixed(1)} KB · ready to score` : 'CSV only · up to 5 MB and 10,000 rows'}</span>
              <input accept=".csv,text/csv" id="batch-file" onChange={handleFile} type="file" />
            </label>
            <div className="requirements">
              <strong>Required feature columns</strong>
              <div className="column-chips">{REQUIRED_COLUMNS.map((column) => <code key={column}>{column}</code>)}</div>
              <p>Optional extra columns are ignored. Invalid rows are listed separately; valid rows can still be scored.</p>
            </div>
            {error && <div className="inline-error" role="alert">{error}</div>}
            <button className="primary-button" disabled={loading || !file} type="submit">
              {loading ? <><span className="button-spinner" />Validating and scoring…</> : <>Run batch prediction <Icon name="arrow" size={17} /></>}
            </button>
          </form>
        </Panel>
        <Panel title="Batch limits" subtitle="Limits protect the API while processing uploads.">
          <div className="limit-list">
            <div><span>Maximum file size</span><strong>5 MB</strong></div>
            <div><span>Maximum data rows</span><strong>10,000</strong></div>
            <div><span>Prediction method</span><strong>Existing model pipeline</strong></div>
          </div>
          <div className="threshold-note"><span className="status-dot amber" /><div><strong>Threshold remains provisional</strong><p>Each result includes the threshold returned by the current API.</p></div></div>
        </Panel>
      </div>
      {loading && <Panel className="results-panel"><LoadingState label="The backend is validating rows and applying the trained preprocessing pipeline…" /></Panel>}
      {result && (
        <Panel
          className="results-panel"
          title="Batch prediction results"
          subtitle={`${formatNumber(result.valid_rows)} valid of ${formatNumber(result.total_rows)} data rows processed`}
          action={<button className="secondary-button compact" disabled={!result.predictions.length} onClick={exportResults} type="button"><Icon name="download" size={16} /> Export CSV</button>}
        >
          <div className="batch-summary">
            <span className="summary-chip safe">{formatNumber(result.valid_rows)} scored</span>
            <span className="summary-chip high">{formatNumber(flaggedResults)} model flagged</span>
            <span className={`summary-chip ${result.invalid_rows.length ? 'high' : 'neutral'}`}>{formatNumber(result.invalid_rows.length)} invalid</span>
          </div>
          {result.predictions.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead><tr><th>CSV row</th><th>Type</th><th>Amount</th><th>Fraud probability</th><th>Class</th><th>Risk</th></tr></thead>
                <tbody>{visibleResults.map((row) => (
                  <tr key={row.row_number}>
                    <td>#{row.row_number}</td><td>{row.type}</td><td>{formatNumber(row.amount)} units</td><td>{validProbability(row.fraud_probability) ? formatPercent(row.fraud_probability, 2) : 'Not provided'}</td><td>{row.is_fraud ? 'Model flagged' : 'Not flagged'}</td><td><span className={`risk-pill ${validProbability(row.fraud_probability) ? riskLevel(row.fraud_probability) : 'neutral'}`}>{validProbability(row.fraud_probability) ? `${riskLevel(row.fraud_probability)} score` : 'Not scored'}</span></td>
                  </tr>
                ))}</tbody>
              </table>
              {result.predictions.length > visibleResults.length && <p className="table-footnote">Showing the first 25 of {formatNumber(result.predictions.length)} results. Export the complete result set as CSV.</p>}
            </div>
          ) : <EmptyState title="No valid rows to score">Correct the reported row errors and upload the CSV again.</EmptyState>}
          {result.invalid_rows.length > 0 && (
            <div className="invalid-rows">
              <h3>Invalid rows <span>{formatNumber(result.invalid_rows.length)}</span></h3>
              <div className="invalid-list">{result.invalid_rows.slice(0, 20).map((row) => <p key={row.row_number}><strong>Row {row.row_number}:</strong> {row.errors.join(' ')}</p>)}</div>
              {result.invalid_rows.length > 20 && <p className="table-footnote">Showing 20 of {formatNumber(result.invalid_rows.length)} invalid rows.</p>}
            </div>
          )}
        </Panel>
      )}
    </>
  )
}

function ValidationChart({ rows }) {
  if (!rows?.length) return <EmptyState title="Validation threshold report unavailable">No threshold comparison is recorded.</EmptyState>
  const width = 680
  const height = 250
  const left = 48
  const right = 18
  const top = 20
  const bottom = 36
  const x = (index) => left + (index / Math.max(rows.length - 1, 1)) * (width - left - right)
  const y = (value) => top + (1 - value) * (height - top - bottom)
  const series = [
    { key: 'precision', label: 'Precision', className: 'chart-line-blue' },
    { key: 'recall', label: 'Recall', className: 'chart-line-violet' },
    { key: 'f1', label: 'F1', className: 'chart-line-green' },
  ]
  return (
    <div className="validation-chart">
      <div className="chart-legend">{series.map((item) => <span key={item.key}><i className={item.className} />{item.label}</span>)}</div>
      <div className="svg-chart-scroll">
        <svg aria-label="Validation precision, recall and F1 by threshold" role="img" viewBox={`0 0 ${width} ${height}`}>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={tick}><line className="chart-grid-line" x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} /><text className="chart-tick" textAnchor="end" x={left - 8} y={y(tick) + 4}>{Math.round(tick * 100)}%</text></g>
          ))}
          {series.map((item) => (
            <g key={item.key}>
              <polyline className={item.className} points={rows.map((row, index) => `${x(index)},${y(row[item.key])}`).join(' ')} />
              {rows.map((row, index) => <circle className={item.className} cx={x(index)} cy={y(row[item.key])} key={`${item.key}-${row.threshold}`} r="3.5" />)}
            </g>
          ))}
          {rows.map((row, index) => <text className="chart-x-tick" key={row.threshold} textAnchor="middle" x={x(index)} y={height - 10}>{row.threshold.toFixed(2)}</text>)}
        </svg>
      </div>
      <p className="chart-footnote">Validation-set threshold comparison. Held-out test results are not used to select the threshold.</p>
    </div>
  )
}

function ModelPerformance() {
  const [performance, setPerformance] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [refreshCount, setRefreshCount] = useState(0)

  useEffect(() => {
    let alive = true
    apiRequest('/model/performance')
      .then((data) => { if (alive) setPerformance(data) })
      .catch((requestError) => { if (alive) setError(requestError.message) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [refreshCount])

  function retry() {
    setError('')
    setLoading(true)
    setRefreshCount((count) => count + 1)
  }

  const report = performance?.test_evaluation
  const matrix = report?.confusion_matrix
  const thresholds = performance?.validation_thresholds
  return (
    <>
      <PageHeading eyebrow="MODEL EVALUATION" title="Model performance" description="Recorded test metrics and validation-only threshold comparisons from the project reports." />
      {loading && <LoadingState />}
      {error && <ErrorState message={error} onRetry={retry} />}
      {report && (
        <>
          <div className="threshold-banner"><span className="status-dot amber" /><div><strong>Decision threshold: {formatPercent(report.threshold, 0)} · provisional</strong><p>Threshold selection used validation data only. Test metrics below are a separate held-out evaluation and did not select the threshold.</p></div></div>
          <div className="stats-grid performance-stats">
            <StatCard label="Average precision" value={report.average_precision.toFixed(4)} note="Held-out test set" icon="chart" tone="blue" />
            <StatCard label="Precision" value={formatPercent(report.precision, 2)} note="Held-out test set" icon="scan" tone="violet" />
            <StatCard label="Recall" value={formatPercent(report.recall, 2)} note="Held-out test set" icon="pulse" tone="green" />
            <StatCard label="F1 score" value={report.f1.toFixed(4)} note="Held-out test set" icon="chart" tone="red" />
          </div>
          <div className="content-grid performance-grid">
            <Panel title="Confusion matrix" subtitle={`${formatNumber(report.test_rows)} held-out rows · threshold ${formatPercent(report.threshold, 0)}`}>
              <div className="matrix-label">PREDICTED CLASS</div>
              <div className="confusion-matrix">
                <span className="matrix-corner">ACTUAL</span><span className="matrix-col-head">Not fraud</span><span className="matrix-col-head">Fraud</span>
                <span className="matrix-row-head">Not fraud</span><div className="matrix-cell true-negative"><strong>{formatNumber(matrix.true_negatives)}</strong><small>True negative</small></div><div className="matrix-cell false-positive"><strong>{formatNumber(matrix.false_positives)}</strong><small>False positive</small></div>
                <span className="matrix-row-head">Fraud</span><div className="matrix-cell false-negative"><strong>{formatNumber(matrix.false_negatives)}</strong><small>False negative</small></div><div className="matrix-cell true-positive"><strong>{formatNumber(matrix.true_positives)}</strong><small>True positive</small></div>
              </div>
            </Panel>
            <Panel title="Validation threshold comparison" subtitle={`${formatNumber(thresholds.validation_rows)} validation rows · validation set only`}>
              <ValidationChart rows={thresholds.threshold_results} />
            </Panel>
          </div>
          <p className="data-provenance"><Icon name="file" size={15} /> Source: repository evaluation reports. These are recorded offline metrics, not live service telemetry.</p>
        </>
      )}
    </>
  )
}

function TransactionHistory() {
  const [records, setRecords] = useState(null)
  const [searchInput, setSearchInput] = useState('')
  const [search, setSearch] = useState('')
  const [fraudStatus, setFraudStatus] = useState('all')
  const [sort, setSort] = useState('newest')
  const [page, setPage] = useState(0)
  const [refreshCount, setRefreshCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const pageSize = 25

  useEffect(() => {
    let alive = true
    const query = new URLSearchParams({
      fraud_status: fraudStatus,
      sort,
      limit: String(pageSize),
      offset: String(page * pageSize),
    })
    if (search) query.set('search', search)
    apiRequest(`/history?${query.toString()}`)
      .then((data) => { if (alive) setRecords(data) })
      .catch((requestError) => { if (alive) setError(requestError.message) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [fraudStatus, page, refreshCount, search, sort])

  function applySearch(event) {
    event.preventDefault()
    setError('')
    setPage(0)
    setSearch(searchInput.trim())
  }

  function refresh() {
    setError('')
    setLoading(true)
    setRefreshCount((count) => count + 1)
  }

  const pageCount = records ? Math.max(1, Math.ceil(records.total / pageSize)) : 1

  return (
    <>
      <PageHeading
        eyebrow="AUDIT TRAIL"
        title="Transaction history"
        description="Search and review predictions persisted locally by the API."
        action={<button className="secondary-button compact" disabled={loading} onClick={refresh} type="button"><Icon name="refresh" size={16} /> Refresh</button>}
      />
      <Panel className="history-panel" title="Recorded predictions" subtitle={records ? `${formatNumber(records.total)} matching prediction${records.total === 1 ? '' : 's'}` : 'Prediction records are stored in the local SQLite database.'}>
        <div className="history-toolbar">
          <form className="history-search" onSubmit={applySearch}>
            <Icon name="search" size={16} />
            <input aria-label="Search transaction history" maxLength="100" onChange={(event) => setSearchInput(event.target.value)} placeholder="Search ID, type, step, or amount…" value={searchInput} />
            <button className="secondary-button compact" type="submit">Search</button>
          </form>
          <label className="history-filter">Fraud status
            <select onChange={(event) => { setFraudStatus(event.target.value); setPage(0) }} value={fraudStatus}>
              <option value="all">All predictions</option>
              <option value="fraud">Model flagged</option>
              <option value="not_fraud">Not flagged</option>
            </select>
          </label>
          <label className="history-filter">Timestamp
            <select onChange={(event) => setSort(event.target.value)} value={sort}>
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
            </select>
          </label>
        </div>
        {loading && <LoadingState label="Loading saved predictions…" />}
        {error && <ErrorState message={error} onRetry={refresh} />}
        {!loading && !error && records?.total === 0 && (
          <EmptyState icon="clock" title={search || fraudStatus !== 'all' ? 'No matching predictions' : 'No predictions recorded yet'}>
            {search || fraudStatus !== 'all'
              ? 'Try a different search term or change the fraud-status filter.'
              : 'Successful single-transaction and batch predictions will appear here. Records are kept in the local SQLite database across backend restarts.'}
          </EmptyState>
        )}
        {!loading && !error && records?.items.length > 0 && (
          <>
            <div className="history-table-wrap">
              <table className="history-table">
                <thead><tr><th>Timestamp</th><th>Transaction</th><th>Step</th><th>Amount</th><th>Origin balance</th><th>Destination balance</th><th>Prediction</th><th>Fraud probability</th><th>Threshold</th></tr></thead>
                <tbody>{records.items.map((record) => (
                  <tr key={record.id}>
                    <td className="timestamp-cell">{formatTimestamp(record.timestamp)}</td>
                    <td><strong>{record.type.replaceAll('_', ' ')}</strong><small className="record-id">{record.id}</small></td>
                    <td>{formatNumber(record.step)}</td>
                    <td>{formatNumber(record.amount)} units</td>
                    <td>{formatNumber(record.oldbalanceOrg)} units</td>
                    <td>{formatNumber(record.oldbalanceDest)} units</td>
                    <td><span className={`risk-pill ${record.is_fraud ? 'high' : 'low'}`}>{record.is_fraud ? 'Model flagged' : 'Not flagged'}</span></td>
                    <td>{validProbability(record.fraud_probability) ? formatPercent(record.fraud_probability, 2) : 'Not provided'}</td>
                    <td>{formatPercent(record.threshold, 0)} <span className="provisional-label">provisional</span></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div className="history-pagination">
              <span>Showing {formatNumber(records.offset + 1)}–{formatNumber(records.offset + records.items.length)} of {formatNumber(records.total)}</span>
              <div>
                <button className="secondary-button compact" disabled={page === 0} onClick={() => setPage((current) => Math.max(0, current - 1))} type="button">Previous</button>
                <span>Page {page + 1} of {pageCount}</span>
                <button className="secondary-button compact" disabled={page + 1 >= pageCount} onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))} type="button">Next</button>
              </div>
            </div>
          </>
        )}
      </Panel>
    </>
  )
}

function Monitoring() {
  const [snapshot, setSnapshot] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [refreshCount, setRefreshCount] = useState(0)

  useEffect(() => {
    let alive = true
    Promise.allSettled([
      apiRequest('/health'),
      apiRequest('/model/status'),
      apiRequest('/history?limit=1'),
      apiRequest('/history?fraud_status=fraud&limit=5'),
    ]).then(([healthResult, modelResult, historyResult, flaggedResult]) => {
      if (!alive) return
      setSnapshot({
        health: healthResult.status === 'fulfilled' ? healthResult.value : null,
        healthError: healthResult.status === 'rejected' ? healthResult.reason.message : '',
        model: modelResult.status === 'fulfilled' ? modelResult.value : null,
        modelError: modelResult.status === 'rejected' ? modelResult.reason.message : '',
        history: historyResult.status === 'fulfilled' ? historyResult.value : null,
        historyError: historyResult.status === 'rejected' ? historyResult.reason.message : '',
        flagged: flaggedResult.status === 'fulfilled' ? flaggedResult.value : null,
        flaggedError: flaggedResult.status === 'rejected' ? flaggedResult.reason.message : '',
      })
      if ([healthResult, modelResult, historyResult, flaggedResult].every((result) => result.status === 'rejected')) {
        setError('The API health and model status endpoints could not be reached.')
      }
      setLoading(false)
    })
    return () => { alive = false }
  }, [refreshCount])

  const capabilityLabels = {
    prediction: 'Single-transaction prediction',
    batch_prediction: 'CSV batch prediction',
    persistent_transaction_history: 'Persistent transaction history',
    drift_monitoring: 'Automated drift monitoring',
    automated_retraining: 'Automated model retraining',
    production_deployment: 'Production deployment',
  }

  return (
    <>
      <PageHeading
        eyebrow="RISK OPERATIONS"
        title="Fraud monitoring"
        description="Review saved model flags and current service health. This workspace does not represent a live alert feed."
        action={<button className="secondary-button compact" disabled={loading} onClick={() => { setLoading(true); setError(''); setRefreshCount((count) => count + 1) }} type="button"><Icon name="refresh" size={16} /> Refresh</button>}
      />
      {loading && <LoadingState label="Checking API health and model artifact availability…" />}
      {error && <ErrorState message={error} />}
      {snapshot && (
        <>
          {snapshot.healthError && <ErrorState message={snapshot.healthError} />}
          {snapshot.modelError && <ErrorState message={snapshot.modelError} />}
          {snapshot.historyError && <ErrorState message={snapshot.historyError} />}
          {snapshot.flaggedError && <ErrorState message={snapshot.flaggedError} />}
          <div className="stats-grid monitoring-stats">
            <StatCard label="Saved predictions" value={snapshot.history ? formatNumber(snapshot.history.total) : '—'} note="Persistent local history" icon="file" tone="blue" />
            <StatCard label="Model-flagged" value={snapshot.flagged ? formatNumber(snapshot.flagged.total) : '—'} note="Predicted class only · not confirmed fraud" icon="scan" tone="red" />
            <StatCard label="Flag rate" value={snapshot.history && snapshot.flagged && snapshot.history.total ? formatPercent(snapshot.flagged.total / snapshot.history.total, 2) : '—'} note="Among saved predictions" icon="chart" tone="violet" />
            <StatCard label="API health" value={snapshot.health?.status === 'healthy' ? 'Healthy' : 'Unavailable'} note="On-demand health check" icon="pulse" tone={snapshot.health?.status === 'healthy' ? 'green' : 'red'} />
          </div>
          <div className="content-grid monitoring-grid">
            <Panel title="API health" subtitle="Live response from GET /health">
              <div className="service-status">
                <span className={`service-indicator ${snapshot.health ? 'online' : 'offline'}`} />
                <div><strong>{snapshot.health?.status === 'healthy' ? 'Healthy' : 'Unavailable'}</strong><p>{snapshot.health?.status === 'healthy' ? 'The API health endpoint responded successfully.' : 'The health endpoint did not return a healthy response.'}</p></div>
              </div>
              <div className="capability-row"><span>Endpoint</span><code>/health</code></div>
              <div className="capability-row"><span>Check performed</span><strong>On page load / refresh</strong></div>
            </Panel>
            <Panel title="Model artifact availability" subtitle="File presence check; does not force-load the lazy model.">
              {snapshot.model ? (
                <>
                  <div className="service-status">
                    <span className={`service-indicator ${snapshot.model.model_artifacts_available ? 'online' : 'offline'}`} />
                    <div><strong>{snapshot.model.model_artifacts_available ? 'Artifacts available' : 'Artifacts missing'}</strong><p>Prediction loads these files on demand when the first request arrives.</p></div>
                  </div>
                  {Object.values(snapshot.model.artifacts).map((artifact) => (
                    <div className="artifact-row" key={artifact.name}><Icon name="file" size={16} /><span>{artifact.name}</span><strong className={artifact.available ? 'text-green' : 'text-red'}>{artifact.available ? 'Present' : 'Missing'}</strong></div>
                  ))}
                  <div className="capability-row"><span>Model version / training metadata</span><strong>{snapshot.model.model_version || snapshot.model.training_information ? 'Available' : 'Not recorded'}</strong></div>
                </>
              ) : <p className="unavailable-note">Model status is unavailable until the backend responds.</p>}
            </Panel>
          </div>
          <div className="content-grid monitoring-grid">
            <Panel title="Recent model-flagged transactions" subtitle="Saved prediction records returned by the history API">
              {snapshot.flagged?.items.length ? (
                <div className="recent-activity-list">
                  {snapshot.flagged.items.map((record) => (
                    <div className="recent-activity-row" key={record.id}>
                      <span className="activity-class-dot fraud" />
                      <div className="recent-activity-main">
                        <strong>{record.type.replaceAll('_', ' ')} <span>· {formatNumber(record.amount)} units</span></strong>
                        <small>{formatTimestamp(record.timestamp)} · {record.id.slice(0, 8)}</small>
                      </div>
                      <span className={`risk-pill ${validProbability(record.fraud_probability) ? riskLevel(record.fraud_probability) : 'neutral'}`}>
                        {validProbability(record.fraud_probability) ? formatPercent(record.fraud_probability, 1) : 'Probability unavailable'}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState icon="scan" title="No saved model flags">
                  {snapshot.flagged?.total === 0
                    ? 'No recorded predictions have been classified as fraud.'
                    : 'Flagged transaction history is unavailable.'}
                </EmptyState>
              )}
            </Panel>
            <Panel title="Prediction class distribution" subtitle="Derived from persistent prediction history · not live traffic">
              {snapshot.history && snapshot.flagged && snapshot.history.total > 0 ? (
                <>
                  <div className="distribution-summary">
                    <div><span className="legend-dot safe-dot" />Not flagged <strong>{formatNumber(snapshot.history.total - snapshot.flagged.total)}</strong></div>
                    <div><span className="legend-dot high-dot" />Model flagged <strong>{formatNumber(snapshot.flagged.total)}</strong></div>
                  </div>
                  <div aria-label={`${snapshot.flagged.total} flagged out of ${snapshot.history.total} saved predictions`} className="stacked-bar" role="img">
                    <span className="bar-not-fraud" style={{ width: `${100 - (snapshot.flagged.total / snapshot.history.total * 100)}%` }} />
                    <span className="bar-fraud" style={{ width: `${snapshot.flagged.total / snapshot.history.total * 100}%` }} />
                  </div>
                  <div className="chart-caption"><span>0</span><span>Flagged share: {formatPercent(snapshot.flagged.total / snapshot.history.total, 2)}</span><span>{formatNumber(snapshot.history.total)} records</span></div>
                </>
              ) : <EmptyState icon="chart" title="No recorded risk distribution">Prediction distribution appears after successful analyses are saved.</EmptyState>}
            </Panel>
          </div>
          <Panel className="capability-panel" title="Implemented and future capabilities" subtitle="Status reflects the backend; this page does not imply production telemetry.">
            <div className="capability-list">
              {snapshot.model && Object.entries(snapshot.model.capabilities).map(([key, status]) => (
                <div className="capability-item" key={key}>
                  <span className={`status-dot ${status === 'implemented' ? 'green' : 'gray'}`} />
                  <span>{capabilityLabels[key] || key}</span>
                  <strong className={status === 'implemented' ? 'capability-live' : 'capability-status'}>{status === 'implemented' ? 'Implemented' : 'Not implemented'}</strong>
                </div>
              ))}
              {!snapshot.model && <p className="unavailable-note">Capability status is unavailable until the backend responds.</p>}
            </div>
          </Panel>
          <div className="monitoring-disclaimer"><strong>Saved prediction history is not real-time monitoring.</strong><span>Health checks are on demand; no continuous polling, automated alerts, drift detection, retraining, or deployment telemetry is configured.</span></div>
        </>
      )}
    </>
  )
}

function Settings({ apiBaseUrl, onSave }) {
  const [value, setValue] = useState(apiBaseUrl)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  function saveSettings(event) {
    event.preventDefault()
    setMessage('')
    setError('')

    let parsedUrl
    try {
      parsedUrl = new URL(value.trim())
    } catch {
      setError('Enter a valid absolute API URL, such as http://127.0.0.1:8000.')
      return
    }
    if (!['http:', 'https:'].includes(parsedUrl.protocol) || parsedUrl.username || parsedUrl.password || parsedUrl.search || parsedUrl.hash) {
      setError('Use an HTTP or HTTPS API base URL without credentials, query parameters, or a fragment.')
      return
    }

    const normalizedUrl = value.trim().replace(/\/+$/, '')
    try {
      localStorage.setItem(API_URL_STORAGE_KEY, normalizedUrl)
    } catch {
      setError('Browser storage is unavailable. The API URL could not be saved.')
      return
    }
    setValue(normalizedUrl)
    onSave(normalizedUrl)
    setMessage('API endpoint saved. Connection status is being checked.')
  }

  function resetSettings() {
    setError('')
    setMessage('')
    try {
      localStorage.removeItem(API_URL_STORAGE_KEY)
    } catch {
      setError('Browser storage is unavailable. The saved API URL could not be removed.')
      return
    }
    setValue(DEFAULT_API_BASE_URL)
    onSave(DEFAULT_API_BASE_URL)
    setMessage('API endpoint reset to the configured Vite default.')
  }

  return (
    <>
      <PageHeading eyebrow="WORKSPACE PREFERENCES" title="Settings" description="Configure the backend endpoint used by this browser. This setting is stored locally on this device." />
      <div className="settings-layout">
        <Panel title="API connection" subtitle="The frontend calls the configured FastAPI service directly from your browser.">
          <form className="settings-form" onSubmit={saveSettings}>
            <label htmlFor="api-base-url">API base URL</label>
            <input
              autoComplete="url"
              id="api-base-url"
              maxLength="250"
              onChange={(event) => { setValue(event.target.value); setMessage(''); setError('') }}
              placeholder="http://127.0.0.1:8000"
              spellCheck="false"
              type="url"
              value={value}
            />
            <p className="field-help">Use the FastAPI origin, optionally followed by a base path. Do not include credentials or tokens.</p>
            {error && <div className="inline-error" role="alert">{error}</div>}
            {message && <div className="inline-success" role="status">{message}</div>}
            <div className="settings-actions">
              <button className="primary-button" type="submit">Save API endpoint</button>
              <button className="secondary-button" onClick={resetSettings} type="button">Reset to Vite default</button>
            </div>
          </form>
        </Panel>
        <Panel title="Data and model" subtitle="Current application behavior">
          <div className="settings-fact"><span>Prediction history</span><strong>Stored locally in the backend SQLite database</strong></div>
          <div className="settings-fact"><span>Model threshold</span><strong>0.95 · provisional</strong></div>
          <div className="settings-fact"><span>Browser storage</span><strong>API URL only · no credentials</strong></div>
          <p className="settings-note">The selected model and preprocessing pipeline are configured by the backend; changing this URL does not modify or retrain model artifacts.</p>
        </Panel>
      </div>
    </>
  )
}

function App() {
  const [activePage, setActivePage] = useState('dashboard')
  const [apiHealthy, setApiHealthy] = useState(null)
  const [modelAvailable, setModelAvailable] = useState(null)
  const [apiBaseUrl, setApiBaseUrl] = useState(getApiBaseUrl)
  const [connectionCheck, setConnectionCheck] = useState(0)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const page = navigation.find((item) => item.id === activePage)

  useEffect(() => {
    let alive = true
    Promise.allSettled([apiRequest('/health'), apiRequest('/model/status')]).then(([healthResult, statusResult]) => {
      if (!alive) return
      setApiHealthy(healthResult.status === 'fulfilled' && healthResult.value.status === 'healthy')
      setModelAvailable(statusResult.status === 'fulfilled' ? statusResult.value.model_artifacts_available : false)
    })
    return () => { alive = false }
  }, [activePage, apiBaseUrl, connectionCheck])

  function saveApiBaseUrl(url) {
    setApiHealthy(null)
    setModelAvailable(null)
    setApiBaseUrl(url)
    setConnectionCheck((check) => check + 1)
  }

  const pageContent = {
    dashboard: <Dashboard onNavigate={setActivePage} />,
    analysis: <TransactionAnalysis />,
    batch: <BatchAnalysis />,
    performance: <ModelPerformance />,
    history: <TransactionHistory />,
    monitoring: <Monitoring />,
    settings: <Settings apiBaseUrl={apiBaseUrl} onSave={saveApiBaseUrl} />,
  }[activePage]

  return (
    <div className="app-shell">
      <Sidebar
        activePage={activePage}
        collapsed={sidebarCollapsed}
        onNavigate={setActivePage}
        onToggleCollapse={() => setSidebarCollapsed((collapsed) => !collapsed)}
      />
      <main className="main-workspace">
        <header className="topbar">
          <div className="breadcrumb"><span>Fraud Intelligence</span><span className="breadcrumb-separator">/</span><strong>{page?.label}</strong></div>
          <div className="topbar-statuses">
            <div aria-label={apiHealthy === true ? 'API connected' : apiHealthy === false ? 'API unavailable' : 'Checking API connection'} className={`api-indicator ${apiHealthy === true ? 'is-online' : apiHealthy === false ? 'is-offline' : ''}`}>
              <span />{apiHealthy === true ? 'API connected' : apiHealthy === false ? 'API unavailable' : 'Checking API'}
            </div>
            <div aria-label={modelAvailable === true ? 'Model artifacts available' : modelAvailable === false ? 'Model artifacts unavailable' : 'Checking model status'} className={`model-indicator ${modelAvailable === true ? 'is-online' : modelAvailable === false ? 'is-offline' : ''}`}>
              <span />{modelAvailable === true ? 'Artifacts available' : modelAvailable === false ? 'Artifacts unavailable' : 'Checking model'}
            </div>
          </div>
        </header>
        <div className="page-content" key={activePage}>{pageContent}</div>
        <footer className="app-footer"><span>Fraud Intelligence · Candidate-model evaluation platform</span><span>Predictions are risk estimates · 0.95 threshold is provisional</span></footer>
      </main>
    </div>
  )
}

export default App

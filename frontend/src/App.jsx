
import { useState } from 'react'
import './App.css'

function App() {
  const [form, setForm] = useState({
    step: '1',
    amount: '1000',
    oldbalanceOrg: '5000',
    oldbalanceDest: '2000',
    type: 'TRANSFER',
  })

  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function handleChange(event) {
    setForm({ ...form, [event.target.name]: event.target.value })
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setResult(null)
    setError('')

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          step: Number(form.step),
          amount: Number(form.amount),
          oldbalanceOrg: Number(form.oldbalanceOrg),
          oldbalanceDest: Number(form.oldbalanceDest),
          type: form.type,
        }),
      })

      if (!response.ok) {
        const details = await response.text()
        throw new Error(`API error (${response.status}): ${details}`)
      }

      setResult(await response.json())
    } catch (err) {
      setError(
        'Could not reach the prediction API. Make sure your FastAPI backend is running on port 8000. ' +
        (err.message || ''),
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="dashboard">
      <header className="topbar">
        <div className="brand-mark">F</div>
        <div>
          <h1>FraudGuard</h1>
          <p>Transaction Intelligence Platform</p>
        </div>
        <span className="status"><span /> Model API</span>
      </header>

      <section className="welcome">
        <p className="eyebrow">FRAUD DETECTION · MACHINE LEARNING</p>
        <h2>Transaction Risk Dashboard</h2>
        <p>Evaluate a transaction using your trained machine learning model.</p>
      </section>

      <section className="content-grid">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="panel-heading">
            <div>
              <h3>Transaction details</h3>
              <p>Enter the transaction information below.</p>
            </div>
            <span className="step-badge">01</span>
          </div>

          <label>
            Transaction type
            <select name="type" value={form.type} onChange={handleChange}>
              <option value="TRANSFER">Transfer</option>
              <option value="CASH_OUT">Cash out</option>
              <option value="PAYMENT">Payment</option>
              <option value="CASH_IN">Cash in</option>
              <option value="DEBIT">Debit</option>
            </select>
          </label>

          <label>
            Transaction step
            <input name="step" type="number" min="1" required value={form.step} onChange={handleChange} />
          </label>

          <label>
            Transaction amount
            <input name="amount" type="number" min="0.01" step="any" required value={form.amount} onChange={handleChange} />
          </label>

          <div className="field-row">
            <label>
              Origin balance before
              <input name="oldbalanceOrg" type="number" min="0" step="any" required value={form.oldbalanceOrg} onChange={handleChange} />
            </label>
            <label>
              Destination balance before
              <input name="oldbalanceDest" type="number" min="0" step="any" required value={form.oldbalanceDest} onChange={handleChange} />
            </label>
          </div>

          <button className="submit-button" type="submit" disabled={loading}>
            {loading ? 'Analyzing transaction...' : 'Analyze transaction →'}
          </button>
          <p className="form-note">Predictions are model estimates, not definitive proof of fraud.</p>
        </form>

        <section className="panel result-panel">
          <p className="eyebrow">MODEL ASSESSMENT</p>
          <h3>Prediction result</h3>

          {!result && !error && (
            <div className="empty-result">
              <div className="shield">✓</div>
              <h4>Ready to analyze</h4>
              <p>Submit the transaction details to see its estimated fraud probability.</p>
            </div>
          )}

          {loading && <p className="message">Your transaction is being analyzed…</p>}

          {error && <div className="error-message">{error}</div>}

          {result && (
            <div className="prediction">
              <div className={`result-symbol ${result.is_fraud ? 'danger' : 'safe'}`}>
                {result.is_fraud ? '!' : '✓'}
              </div>
              <p className="result-label">MODEL CLASSIFICATION</p>
              <h2 className={result.is_fraud ? 'danger-text' : 'safe-text'}>
                {result.is_fraud ? 'Potential Fraud' : 'Low Risk Classification'}
              </h2>
              <div className="probability">
                <span>Fraud probability</span>
                <strong>{(result.fraud_probability * 100).toFixed(2)}%</strong>
              </div>
              <div className="probability-track">
                <div style={{ width: `${Math.max(0, Math.min(100, result.fraud_probability * 100))}%` }} />
              </div>
              <div className="result-meta">
                <span>Decision threshold</span>
                <strong>{(result.threshold * 100).toFixed(0)}%</strong>
              </div>
              <p className="form-note">
                This is a candidate model with a provisional threshold. Validate its performance before real-world use.
              </p>
            </div>
          )}
        </section>
      </section>

      <footer>FraudGuard · MLOps Portfolio Project · Candidate model</footer>
    </main>
  )
}

export default App

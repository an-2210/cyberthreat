import { useState } from 'react';
import { api, PredictionResult } from '../api';
import { useLoader } from './useLoader';

const TASK_LABEL: Record<string, string> = { binary: 'Attack or benign', multiclass: 'Attack type' };
const MODEL_LABEL: Record<string, string> = { random_forest: 'Random Forest', xgboost: 'XGBoost' };

export function TryItTab() {
  const models = useLoader(api.models);
  const [modelKey, setModelKey] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chosenKey = modelKey || models.data?.find((m) => m.task === 'binary' && m.model === 'xgboost')?.key || '';

  async function run() {
    if (!file || !chosenKey) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.predict(file, chosenKey));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="card">
        <h2>Classify your own flows</h2>
        <p>
          Upload a CSV with one flow per row. It must contain the CIC-IDS2017 feature columns (the same names as in the
          dataset, e.g. <code>Flow Duration</code>, <code>Total Fwd Packets</code>). A <code>Label</code> column is not needed and
          is ignored if present. Each row is classified independently.
        </p>
        <div className="controls">
          <label>
            Model
            <select value={chosenKey} onChange={(e) => setModelKey(e.target.value)} disabled={!models.data}>
              {models.data?.map((m) => (
                <option key={m.key} value={m.key}>
                  {MODEL_LABEL[m.model]} · {TASK_LABEL[m.task]}
                </option>
              ))}
            </select>
          </label>
          <label>
            CSV file
            <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <button className="primary" disabled={!file || busy || !chosenKey} onClick={run}>
            {busy ? 'Classifying…' : 'Classify'}
          </button>
        </div>
        <p className="muted">
          Binary models answer "attack or benign". Multiclass models name the attack type. Try a file of attack flows from
          the dataset to see the models react.
        </p>
      </div>

      {error && <div className="notice error">{error}</div>}

      {result && (
        <div className="card">
          <h3>
            {result.rowsClassified.toLocaleString()} flows classified in {result.milliseconds.toFixed(0)} ms
          </h3>
          <div className="grid" style={{ marginBottom: '1rem' }}>
            {Object.entries(result.counts)
              .sort((a, b) => b[1] - a[1])
              .map(([label, count]) => (
                <div key={label} className="card stat" style={{ margin: 0 }}>
                  <div className="value">{count.toLocaleString()}</div>
                  <div className="label">
                    <span className={`pill ${label === 'BENIGN' ? 'benign' : 'attack'}`}>{label}</span>
                  </div>
                </div>
              ))}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th className="num">Row</th><th>Prediction</th><th className="num">Confidence</th></tr>
              </thead>
              <tbody>
                {result.rows.slice(0, 200).map((r) => (
                  <tr key={r.row}>
                    <td className="num">{r.row}</td>
                    <td><span className={`pill ${r.prediction === 'BENIGN' ? 'benign' : 'attack'}`}>{r.prediction}</span></td>
                    <td className="num">{(r.confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">Showing the first {Math.min(result.rows.length, 200)} rows. Confidence is the model's probability for its chosen label.</p>
        </div>
      )}
    </>
  );
}

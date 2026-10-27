import { useState } from 'react';
import { api, AnalysisResult, AnalyzedRow, FlowExplanation, GeneratedReport } from '../api';
import { useLoader } from './useLoader';

const RISK_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNRATED', 'NONE'];

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

function downloadBlob(name: string, type: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function rowsToCsv(rows: AnalyzedRow[]): string {
  const header = 'row,prediction,attack_type,attack_confidence,risk,risk_score';
  const lines = rows.map((r) => [r.row, r.prediction, r.attackType ?? '', r.confidence, r.risk, r.riskScore ?? ''].join(','));
  return [header, ...lines].join('\n');
}

export function TryItTab() {
  const ollama = useLoader(api.ollama);
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [explanation, setExplanation] = useState<FlowExplanation | null>(null);
  const [explainBusy, setExplainBusy] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [report, setReport] = useState<GeneratedReport | null>(null);
  const [reportBusy, setReportBusy] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);

  async function analyze() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setAnalysis(null);
    setSelected(null);
    setExplanation(null);
    setReport(null);
    try {
      setAnalysis(await api.analyze(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function explain(row: number) {
    if (!file) return;
    setSelected(row);
    setExplanation(null);
    setExplainError(null);
    setExplainBusy(true);
    try {
      setExplanation(await api.explain(file, row));
    } catch (e) {
      setExplainError((e as Error).message);
    } finally {
      setExplainBusy(false);
    }
  }

  async function generateReport() {
    if (!file) return;
    setReportBusy(true);
    setReportError(null);
    setReport(null);
    try {
      setReport(await api.summarize(file));
    } catch (e) {
      setReportError((e as Error).message);
    } finally {
      setReportBusy(false);
    }
  }

  const incidents = analysis
    ? analysis.rows
        .filter((r) => r.prediction === 'ATTACK')
        .sort((a, b) => (b.riskScore ?? -1) - (a.riskScore ?? -1))
        .slice(0, 200)
    : [];

  return (
    <>
      <div className="card">
        <div className="report-head">
          <div>
            <h2>Analyse a flow file</h2>
            <p className="muted" style={{ margin: 0 }}>
              Upload a CSV with one network flow per row, using the CIC-IDS2017 feature columns (a <code>Label</code> column is
              ignored if present). Every flow is classified, flagged flows get an attack type, and each gets a risk level.
            </p>
          </div>
          <div>
            {ollama.data && (
              <span className={`pill ${ollama.data.reachable ? 'ok' : 'attack'}`}>
                {ollama.data.reachable ? `Ollama · ${ollama.data.model}` : 'Ollama not reachable'}
              </span>
            )}
          </div>
        </div>
        <div className="controls" style={{ marginTop: '0.9rem' }}>
          <label>
            CSV file
            <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <button className="primary" disabled={!file || busy} onClick={analyze}>
            {busy ? 'Analysing…' : 'Analyse'}
          </button>
        </div>
        <p className="muted">
          Detection uses the XGBoost binary model; attack types come from the XGBoost multiclass model. The risk score is the
          transparent formula in src/threat_intelligence/risk_engine.py.
        </p>
      </div>

      {error && <div className="notice error">{error}</div>}

      {analysis && (
        <>
          <div className="grid" style={{ marginBottom: '1rem' }}>
            <Fact label="flows analysed" value={analysis.rowsAnalyzed.toLocaleString()} />
            <Fact label="flagged as attack" value={analysis.attackRows.toLocaleString()} />
            <Fact label="benign" value={analysis.benignRows.toLocaleString()} />
            <Fact
              label="critical or high risk"
              value={((analysis.riskCounts.CRITICAL ?? 0) + (analysis.riskCounts.HIGH ?? 0)).toLocaleString()}
            />
          </div>

          <div className="card">
            <div className="report-head">
              <div>
                <h3>Risk distribution</h3>
                <p className="muted" style={{ margin: 0 }}>
                  UNRATED: flagged by the binary model, but no attack type was named, so no impact value applies.
                </p>
              </div>
              <div className="techniques">
                {RISK_ORDER.filter((lvl) => analysis.riskCounts[lvl]).map((lvl) => (
                  <span key={lvl} className={`pill severity-${lvl.toLowerCase()}`}>
                    {lvl} · {analysis.riskCounts[lvl].toLocaleString()}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {Object.keys(analysis.attackTypeCounts).length > 0 && (
            <div className="card">
              <h3>Attack types and ATT&amp;CK mapping</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Attack type</th>
                      <th className="num">Flows</th>
                      <th>ATT&amp;CK techniques</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analysis.attackTypeCounts)
                      .sort((a, b) => b[1] - a[1])
                      .map(([name, count]) => (
                        <tr key={name}>
                          <td>{name}</td>
                          <td className="num">{count.toLocaleString()}</td>
                          <td>
                            <div className="techniques">
                              {(analysis.mitre[name] ?? []).length === 0 && <span className="muted">none assigned</span>}
                              {(analysis.mitre[name] ?? []).map((t) => (
                                <span key={t.technique} className="pill technique">
                                  {t.technique} · {t.confidence}
                                </span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="card">
            <div className="report-head">
              <div>
                <h3>Flagged flows, highest risk first</h3>
                <p className="muted" style={{ margin: 0 }}>
                  Showing {incidents.length.toLocaleString()} of {analysis.attackRows.toLocaleString()} flagged flows. Select
                  one to see which features drove its decision.
                </p>
              </div>
              <div className="controls" style={{ margin: 0 }}>
                <button onClick={() => downloadBlob('analysis.csv', 'text/csv', rowsToCsv(analysis.rows))}>
                  Download all rows (CSV)
                </button>
              </div>
            </div>
            {incidents.length === 0 ? (
              <p className="muted">No flows were flagged in this file.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th className="num">Row</th>
                      <th>Attack type</th>
                      <th className="num">Attack confidence</th>
                      <th>Risk</th>
                      <th className="num">Score</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {incidents.map((r) => (
                      <tr key={r.row}>
                        <td className="num">{r.row}</td>
                        <td>{r.attackType ?? <span className="muted">not named</span>}</td>
                        <td className="num">{pct(r.confidence)}</td>
                        <td>
                          <span className={`pill severity-${r.risk.toLowerCase()}`}>{r.risk}</span>
                        </td>
                        <td className="num">{r.riskScore === null ? '—' : r.riskScore.toFixed(3)}</td>
                        <td>
                          <button onClick={() => explain(r.row)} disabled={explainBusy}>
                            Explain
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {selected !== null && (
            <div className="card">
              <h3>Why flow {selected} was flagged</h3>
              {explainBusy && <p className="muted">Computing SHAP values…</p>}
              {explainError && <div className="notice error">{explainError}</div>}
              {explanation && (
                <>
                  <p>
                    Attack probability {pct(explanation.attackProbability)}. SHAP contributions (log-odds) of the most influential
                    features, from the binary XGBoost model. Positive values push the flow towards ATTACK.
                  </p>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Feature</th>
                          <th className="num">Value</th>
                          <th className="num">Contribution</th>
                          <th>Direction</th>
                        </tr>
                      </thead>
                      <tbody>
                        {explanation.top_features.map((f) => (
                          <tr key={f.feature}>
                            <td>{f.feature}</td>
                            <td className="num">{f.value.toLocaleString(undefined, { maximumFractionDigits: 3 })}</td>
                            <td className="num">
                              {f.shap > 0 ? '+' : ''}
                              {f.shap.toFixed(3)}
                            </td>
                            <td>
                              <span className={`pill ${f.shap > 0 ? 'attack' : 'benign'}`}>{f.direction}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}

          <div className="card">
            <div className="report-head">
              <div>
                <h3>Detection report</h3>
                <p className="muted" style={{ margin: 0 }}>
                  Written by {ollama.data?.model ?? 'the local model'} through Ollama. The numbers come from this file; the text
                  is drawn from the saved ATT&amp;CK reference text for the attack types found.
                </p>
              </div>
              <div className="controls" style={{ margin: 0 }}>
                <button className="primary" disabled={reportBusy || ollama.data?.reachable === false} onClick={generateReport}>
                  {reportBusy ? 'Generating…' : report ? 'Generate again' : 'Generate report'}
                </button>
                {report && (
                  <button onClick={() => downloadBlob('report.json', 'application/json', JSON.stringify(report, null, 2))}>
                    Download report (JSON)
                  </button>
                )}
              </div>
            </div>
            {ollama.data && !ollama.data.reachable && (
              <div className="notice error">
                Ollama is not reachable at {ollama.data.host}. Start it with <code>ollama serve</code>.
              </div>
            )}
            {reportError && <div className="notice error">{reportError}</div>}
            {report && (
              <>
                <p className="headline">{report.headline}</p>
                {report.summary && <p>{report.summary}</p>}
                <div className="grid" style={{ marginBottom: '1rem' }}>
                  <Fact label="flows flagged as attack" value={report.facts.flaggedAttack.toLocaleString()} />
                  <Fact label="flagged with 90%+ confidence" value={report.facts.highConfidenceAttack.toLocaleString()} />
                  <Fact label="flagged below 70% confidence" value={report.facts.borderlineAttack.toLocaleString()} />
                  <Fact
                    label="mean confidence on flagged flows"
                    value={report.facts.meanAttackConfidence === null ? '—' : pct(report.facts.meanAttackConfidence)}
                  />
                </div>
                {report.facts.attackTypes.length > 0 && (
                  <div className="table-wrap" style={{ marginBottom: '1rem' }}>
                    <table>
                      <thead>
                        <tr>
                          <th>Attack type</th>
                          <th className="num">Flows</th>
                          <th className="num">Mean confidence</th>
                          <th>Risk</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.facts.attackTypes.map((a) => (
                          <tr key={a.label}>
                            <td>
                              {a.label} {report.referencesUsed.includes(a.label) && <span className="pill">reference text</span>}
                            </td>
                            <td className="num">{a.count.toLocaleString()}</td>
                            <td className="num">{pct(a.meanConfidence)}</td>
                            <td>
                              <span className={`pill severity-${a.risk.toLowerCase()}`}>{a.risk}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {report.facts.flaggedWithoutType > 0 && (
                  <p className="muted">
                    {report.facts.flaggedWithoutType.toLocaleString()} flagged flows were not assigned an attack type by the
                    multiclass model.
                  </p>
                )}
                <div className="note-cols">
                  <div>
                    <h4>Findings</h4>
                    <ul>
                      {report.findings.map((f) => (
                        <li key={f}>{f}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4>Next steps</h4>
                    <ul>
                      {report.nextSteps.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="card stat" style={{ margin: 0 }}>
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}

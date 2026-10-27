import { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api, ModelResult, RareClassRow } from '../api';
import { ErrorBox, useLoader } from './useLoader';

const MODEL_NAME: Record<string, string> = { random_forest: 'Random Forest', xgboost: 'XGBoost' };
const TASK_NAME: Record<string, string> = { binary: 'Binary (attack or benign)', multiclass: 'Multiclass (attack type)' };

export function ModelsTab() {
  const models = useLoader(api.models);
  const rare = useLoader(api.rareClass);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const selected: ModelResult | undefined = useMemo(() => {
    if (!models.data || models.data.length === 0) return undefined;
    return models.data.find((m) => m.key === selectedKey) ?? models.data[0];
  }, [models.data, selectedKey]);

  if (models.error) return <ErrorBox message={models.error} />;
  if (!models.data || !selected) return <p className="muted">Loading…</p>;

  return (
    <>
      <div className="card">
        <h2>Trained models</h2>
        <p>
          Four models were trained: two algorithms (Random Forest and XGBoost), each once for the binary question and
          once for the attack-type question. Every number below is on the test set, which the models never saw during
          training.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Question</th>
                <th className="num">Accuracy</th>
                <th className="num">Precision</th>
                <th className="num">Recall</th>
                <th className="num">F1</th>
                <th className="num">MCC</th>
                <th className="num">Train time</th>
              </tr>
            </thead>
            <tbody>
              {models.data.map((m) => (
                <tr
                  key={m.key}
                  onClick={() => setSelectedKey(m.key)}
                  style={{ cursor: 'pointer', background: m.key === selected.key ? '#eef4ff' : undefined }}
                >
                  <td>{MODEL_NAME[m.model]}</td>
                  <td>{m.task}</td>
                  <td className="num">{m.accuracy.toFixed(4)}</td>
                  <td className="num">{m.precision.toFixed(3)}</td>
                  <td className="num">{m.recall.toFixed(3)}</td>
                  <td className="num">{m.f1.toFixed(3)}</td>
                  <td className="num">{m.mcc.toFixed(3)}</td>
                  <td className="num">{m.trainSeconds.toFixed(0)} s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Click a row to inspect that model. For the binary question, precision/recall/F1 are for the ATTACK class. For the
          multiclass question they are averaged over all 15 classes equally, so rare attacks count as much as common ones.
        </p>
      </div>

      <div className="card">
        <h2>{MODEL_NAME[selected.model]} · {TASK_NAME[selected.task]}</h2>
        <p className="muted">
          Trained on {selected.trainRows.toLocaleString()} rows, tested on {selected.testRows.toLocaleString()} rows.
          {selected.rocAuc !== null && <> ROC-AUC {selected.rocAuc.toFixed(4)}.</>}
          {selected.fpr !== null && <> False-positive rate {selected.fpr.toFixed(4)}: the share of benign flows wrongly flagged.</>}
        </p>

        <h3>Confusion matrix</h3>
        <p className="muted">
          Rows are the true label, columns are what the model predicted. Green is correct, red is a mistake.
        </p>
        <ConfusionTable cm={selected.confusionMatrix.matrix} labels={selected.confusionMatrix.labels} />
      </div>

      {selected.perClass && (
        <div className="card">
          <h3>Per attack type</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Class</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Test rows</th></tr>
              </thead>
              <tbody>
                {Object.entries(selected.perClass).map(([label, s]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td className="num">{s.precision.toFixed(3)}</td>
                    <td className="num">{s.recall.toFixed(3)}</td>
                    <td className="num">{s.f1.toFixed(3)}</td>
                    <td className="num">{s.support.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">
            Recall on a class is the share of that attack type the model flags. Small classes (Infiltration, SQL injection,
            Heartbleed) have very few test rows, so their scores are less precise.
          </p>
        </div>
      )}

      <div className="card">
        <h3>What the model relies on most</h3>
        <p className="muted">Feature importance from the trained model: higher means the model uses that feature more when splitting flows.</p>
        <div className="chart" style={{ height: 380 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={selected.topFeatures.map(([name, value]) => ({ name, value }))} layout="vertical" margin={{ left: 200, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" fontSize={12} />
              <YAxis type="category" dataKey="name" width={200} fontSize={11} />
              <Tooltip formatter={(v: number) => v.toFixed(4)} />
              <Bar dataKey="value" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {rare.data && rare.data.length > 0 && <RareClassCard rows={rare.data} />}
    </>
  );
}

function ConfusionTable({ cm, labels }: { cm: number[][]; labels: string[] }) {
  return (
    <div className="table-wrap">
      <table className="matrix">
        <thead>
          <tr>
            <th>true \ predicted</th>
            {labels.map((l) => <th key={l} className="num">{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={labels[i]}>
              <th>{labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} className={i === j ? 'diag' : v > 0 ? 'off' : ''}>{v.toLocaleString()}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RareClassCard({ rows }: { rows: RareClassRow[] }) {
  return (
    <div className="card">
      <h3>Experiment: balancing the rare attack classes</h3>
      <p>
        The multiclass models struggle with rare attacks. The rare-class experiment caps each class in the training data
        so no attack type dominates, and optionally weights rare classes more heavily. The test set is unchanged, so the
        numbers are directly comparable.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Training data</th><th>Model</th><th className="num">Macro F1</th><th className="num">Weighted F1</th><th className="num">Accuracy</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.variant}-${r.model}`}>
                <td>{r.variant}</td>
                <td>{MODEL_NAME[r.model] ?? r.model}</td>
                <td className="num">{r.macro_f1.toFixed(3)}</td>
                <td className="num">{r.weighted_f1.toFixed(3)}</td>
                <td className="num">{r.accuracy.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted">
        "original" is the main pipeline above. The capped variants are research results only; they are not the models
        used by the "Try it" tab.
      </p>
    </div>
  );
}

import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';
import { ErrorBox, useLoader } from './useLoader';
import { FamilyHoldoutCard } from './FamilyHoldoutCard';

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

export function UnseenTab() {
  const detectors = useLoader(api.detectors);
  const unseen = useLoader(api.unseenAttacks);

  if (detectors.error) return <ErrorBox message={detectors.error} />;
  if (unseen.error) return <ErrorBox message={unseen.error} />;
  if (!detectors.data || !unseen.data) return <p className="muted">Loading…</p>;

  const d = detectors.data;
  const u = unseen.data;
  const chart = u.folds
    .slice()
    .sort((a, b) => a.heldOut.localeCompare(b.heldOut))
    .map((f) => ({ name: f.heldOut, supervised: f.supervised, autoencoder: f.autoencoder }));

  return (
    <>
      <div className="card">
        <h2>Can it catch attacks it has not seen?</h2>
        <p>
          The supervised model learns from labelled attacks, so it can only name attack types it was trained on. An
          anomaly detector learns only what normal traffic looks like, so it can flag unfamiliar behaviour without knowing
          its name. Two tests check this, and both use the same false-positive budget: <strong>1% of normal flows flagged</strong>.
        </p>
        <p className="muted">
          Test 1 compares the detectors on the normal test split. Test 2 removes one attack type from training entirely,
          then checks whether the model still catches it.
        </p>
      </div>

      <div className="card">
        <h3>Test 1 · Detectors compared on the test split</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Detector</th>
                <th className="num">Recall</th>
                <th className="num">Precision</th>
                <th className="num">F1</th>
                <th className="num">False-positive rate</th>
                <th className="num">ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {d.detectors.map((row) => (
                <tr key={row.key}>
                  <td>{row.label}</td>
                  <td className="num">{row.recall.toFixed(3)}</td>
                  <td className="num">{row.precision.toFixed(3)}</td>
                  <td className="num">{row.f1.toFixed(3)}</td>
                  <td className="num">{pct(row.fpr)}</td>
                  <td className="num">{row.rocAuc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Anomaly detectors were picked on validation data. The autoencoder had the best validation ROC-AUC (
          {d.anomalyAucOnValidation.autoencoder?.toFixed(3)}, against{' '}
          {d.anomalyAucOnValidation.isolation_forest_log?.toFixed(3)} for the best Isolation Forest). Fusion weights the
          supervised score at {d.fusion.alpha.toFixed(2)} and the autoencoder at {(1 - d.fusion.alpha).toFixed(2)}. On this
          test split, fusion matches the supervised model: the attacks are the ones it was trained on, so the supervised
          score already catches them.
        </p>
      </div>

      <div className="card">
        <h3>Test 2 · Each attack type held out of training</h3>
        <p>
          For every attack type, the supervised model is retrained with that attack removed from training and validation.
          The bars show how many of that attack's test flows are still flagged. The autoencoder never saw attacks, so its
          score is the same in every test.
        </p>
        <div className="grid" style={{ marginBottom: '1rem' }}>
          <div className="card stat" style={{ margin: 0 }}>
            <div className="value">{pct(u.meanDetection.supervised)}</div>
            <div className="label">mean detection, supervised (held-out attacks)</div>
          </div>
          <div className="card stat" style={{ margin: 0 }}>
            <div className="value">{pct(u.meanDetection.autoencoder)}</div>
            <div className="label">mean detection, autoencoder (held-out attacks)</div>
          </div>
          <div className="card stat" style={{ margin: 0 }}>
            <div className="value">
              {u.detectedAtLeastHalf.supervised} / {u.nHeldOut}
            </div>
            <div className="label">attack types detected at least half the time (supervised)</div>
          </div>
        </div>

        <div className="chart" style={{ height: 380 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chart} margin={{ left: 10, right: 20, bottom: 70 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" angle={-40} textAnchor="end" interval={0} fontSize={11} height={90} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={12} />
              <Tooltip formatter={(v: number) => pct(v)} />
              <Legend verticalAlign="top" />
              <Bar dataKey="supervised" name="Supervised (retrained without this attack)" fill="#2563eb" />
              <Bar dataKey="autoencoder" name="Autoencoder (never saw attacks)" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Held-out attack</th>
                <th className="num">Test flows</th>
                <th className="num">Supervised</th>
                <th className="num">Autoencoder</th>
                <th className="num">Fusion</th>
              </tr>
            </thead>
            <tbody>
              {u.folds
                .slice()
                .sort((a, b) => b.supervised - a.supervised)
                .map((f) => (
                  <tr key={f.heldOut}>
                    <td>
                      {f.heldOut}{' '}
                      {f.testRows < 50 && <span className="pill">low sample</span>}
                    </td>
                    <td className="num">{f.testRows.toLocaleString()}</td>
                    <td className="num">{pct(f.supervised)}</td>
                    <td className="num">{pct(f.autoencoder)}</td>
                    <td className="num">{pct(f.fusion)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Baseline with nothing held out: supervised {pct(u.baseline.supervised)}, autoencoder {pct(u.baseline.autoencoder)},
          at {pct(u.baseline.falsePositiveRate)} false positives. Fusion chose the supervised score alone in every fold
          (weight 1.00), so it is identical to the supervised column. {u.nLowSample} attack types have fewer than 50 test
          flows, so their rates are noisy.
        </p>
        <p className="muted">
          Held-out attacks still have close relatives in training. For example, DoS Hulk is in training when DDoS is held
          out. Test 3 below removes whole families at once.
        </p>
      </div>
      <FamilyHoldoutCard />
    </>
  );
}

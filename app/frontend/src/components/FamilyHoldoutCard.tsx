import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';
import { ErrorBox, useLoader } from './useLoader';

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

// Test 3: whole attack families held out together, so no member of the family is left in training.
export function FamilyHoldoutCard() {
  const { data, error } = useLoader(api.familyHoldout);

  if (error) return <ErrorBox message={error} />;
  if (!data) return <p className="muted">Loading…</p>;

  const chart = data.folds.map((f) => ({ name: f.heldOut, supervised: f.supervised, autoencoder: f.autoencoder }));

  return (
    <div className="card">
      <h3>Test 3 · Whole attack families held out together</h3>
      <p>
        Here the supervised model is retrained with a whole family removed, so none of its members are in training. Thresholds
        are still set on benign validation flows ({pct(data.targetFpr)} false positives). The DoS family is still
        mostly caught, probably because its flows look like other attacks in training (high volume, short flows). The
        Patator family is mostly missed.
      </p>

      <div className="grid" style={{ marginBottom: '1rem' }}>
        <div className="card stat" style={{ margin: 0 }}>
          <div className="value">{pct(data.meanDetection.supervised)}</div>
          <div className="label">mean detection, supervised</div>
        </div>
        <div className="card stat" style={{ margin: 0 }}>
          <div className="value">{pct(data.meanDetection.autoencoder)}</div>
          <div className="label">mean detection, autoencoder</div>
        </div>
        <div className="card stat" style={{ margin: 0 }}>
          <div className="value">
            {data.detectedAtLeastHalf.supervised} / {data.nHeldOut}
          </div>
          <div className="label">families detected at least half the time (supervised)</div>
        </div>
      </div>

      <div className="chart" style={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chart} margin={{ left: 10, right: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" interval={0} fontSize={11} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={12} />
            <Tooltip formatter={(v: number) => pct(v)} />
            <Legend verticalAlign="top" />
            <Bar dataKey="supervised" name="Supervised (family removed from training)" fill="#2563eb" />
            <Bar dataKey="autoencoder" name="Autoencoder (never saw attacks)" fill="#f59e0b" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Held-out family</th>
              <th className="num">Test flows</th>
              <th className="num">Supervised</th>
              <th className="num">Autoencoder</th>
              <th className="num">Fusion</th>
            </tr>
          </thead>
          <tbody>
            {data.folds.map((f) => (
              <tr key={f.heldOut}>
                <td>
                  {f.heldOut}
                  {f.heldOutLabels && <div className="muted" style={{ fontSize: '0.8rem' }}>{f.heldOutLabels.join(', ')}</div>}
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
        Baseline with nothing held out: supervised {pct(data.baseline.supervised)}, autoencoder {pct(data.baseline.autoencoder)}.
        The autoencoder's numbers are the same as in Test 2, because it never trains on attacks.
      </p>
    </div>
  );
}

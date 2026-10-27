import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api, BinaryMetrics } from '../api';
import { ErrorBox, useLoader } from './useLoader';

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const num = (v: number) => v.toFixed(3);

export function UnswTab() {
  const { data: u, error } = useLoader(api.unsw);

  if (error) return <ErrorBox message={error} />;
  if (!u) return <p className="muted">Loading…</p>;

  const t = u.transfer;
  const n = u.native;

  const comparison: { label: string; m: BinaryMetrics; note: string }[] = [
    { label: 'CIC-trained, transferred to UNSW (calibrated threshold)', m: t.calibrated, note: 'Fails: it flags almost no UNSW attacks.' },
    { label: 'UNSW-trained, 9 shared features', m: t.unswInDomain, note: 'Same nine features as the transfer test.' },
    { label: `UNSW-trained, all ${n.featureCount} features`, m: n.test, note: 'Headline result for UNSW-NB15.' },
  ];

  const shiftChart = t.featureShift.map((r) => ({ name: r.feature, cic: r.aucCic, unsw: r.aucUnsw }));

  const fuzzChart = [
    ...u.fuzzers.variants.map((v) => ({ name: v.label, recall: v.fuzzersRecall })),
    ...u.fuzzers.seedRuns.map((s) => ({ name: s.label, recall: s.fuzzersRecall.mean })),
  ];

  const ttlLabel = Object.fromEntries(u.ttl.variants.map((v) => [v.key, v.label]));

  return (
    <>
      <div className="card">
        <h2>UNSW-NB15: a second dataset</h2>
        <p>
          UNSW-NB15 is a separate attack dataset with different feature definitions. It answers two questions. First, does a
          detector trained on CIC-IDS2017 work on UNSW-NB15? Second, how well can UNSW-NB15 be detected when a model is trained
          on its own features? Both use a 1% benign false-positive budget, with thresholds chosen on training data only.
        </p>
        <p>
          <strong>Short answer:</strong> the transfer fails. A detector trained on UNSW-NB15 itself reaches an F1 of{' '}
          {num(n.test.f1)}, but that result is in-domain and should not be read as proof it generalizes.
        </p>
      </div>

      <div className="grid" style={{ marginBottom: '1rem' }}>
        <div className="card stat" style={{ margin: 0 }}>
          <div className="value">{num(n.test.f1)}</div>
          <div className="label">F1, UNSW-native, all features</div>
        </div>
        <div className="card stat" style={{ margin: 0 }}>
          <div className="value">{num(n.test.roc_auc)}</div>
          <div className="label">ROC-AUC, UNSW-native</div>
        </div>
        <div className="card stat" style={{ margin: 0 }}>
          <div className="value">{pct(n.test.accuracy)}</div>
          <div className="label">accuracy on {n.rows.test_rows.toLocaleString()} test flows</div>
        </div>
        <div className="card stat" style={{ margin: 0 }}>
          <div className="value">{pct(t.calibrated.recall)}</div>
          <div className="label">UNSW attacks caught by the CIC-trained model</div>
        </div>
      </div>

      <div className="card">
        <h3>Three ways to test on UNSW-NB15</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">Precision</th>
                <th className="num">Recall</th>
                <th className="num">F1</th>
                <th className="num">False-positive rate</th>
                <th className="num">ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((c) => (
                <tr key={c.label}>
                  <td>
                    {c.label}
                    <div className="muted">{c.note}</div>
                  </td>
                  <td className="num">{num(c.m.precision)}</td>
                  <td className="num">{num(c.m.recall)}</td>
                  <td className="num">{num(c.m.f1)}</td>
                  <td className="num">{pct(c.m.fpr)}</td>
                  <td className="num">{num(c.m.roc_auc)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          The CIC-trained model reached F1 {num(t.cicInDomain.f1)} on its own CIC test split. Its precision on UNSW is{' '}
          {num(t.calibrated.precision)}, but recall is {pct(t.calibrated.recall)}: the threshold moves the decision point, not
          the ranking, and the ranking itself is weak (ROC-AUC {num(t.calibrated.roc_auc)}).
        </p>
      </div>

      <div className="card">
        <h3>Why the transfer fails: the features point in different directions</h3>
        <p>
          For each shared feature, the chart shows how well it separates attacks from benign flows on each dataset (ROC-AUC).
          0.5 means no separation. Below 0.5 means the direction is reversed: a feature that marks attacks as large on CIC-IDS2017
          marks them as small on UNSW-NB15. A model that learns the CIC direction is then wrong on UNSW.
        </p>
        <div className="chart" style={{ height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={shiftChart} margin={{ left: 10, right: 20, bottom: 90 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" angle={-40} textAnchor="end" interval={0} fontSize={11} height={100} />
              <YAxis domain={[0, 1]} fontSize={12} />
              <Tooltip formatter={(v: number) => v.toFixed(3)} />
              <Legend verticalAlign="top" />
              <ReferenceLine y={0.5} stroke="#6b7280" strokeDasharray="4 4" />
              <Bar dataKey="cic" name="Separation on CIC-IDS2017" fill="#2563eb" />
              <Bar dataKey="unsw" name="Separation on UNSW-NB15" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="muted">
          Seven of the nine shared features point one way on CIC-IDS2017 and the reverse way on UNSW-NB15. For example, Bwd
          Packet Length Mean separates at 0.77 on CIC and 0.24 on UNSW. The other two sit near 0.5 on both datasets, so they
          carry little signal either way. The nine features are approximations of the same ideas, not the same measurements.
        </p>
      </div>

      <div className="card">
        <h3>What the UNSW-native model relies on</h3>
        <p>
          The top features are TTL fields and connection-state counters. To check whether the result depends on them, the model
          was retrained three ways, each with the same threshold rule.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Feature set</th>
                <th className="num">Features</th>
                <th className="num">Accuracy</th>
                <th className="num">F1</th>
                <th className="num">ROC-AUC</th>
                <th className="num">Fuzzers recall</th>
              </tr>
            </thead>
            <tbody>
              {u.ttl.variants.map((v) => (
                <tr key={v.key}>
                  <td>{v.label}</td>
                  <td className="num">{v.featureCount}</td>
                  <td className="num">{pct(v.test.accuracy)}</td>
                  <td className="num">{num(v.test.f1)}</td>
                  <td className="num">{num(v.test.roc_auc)}</td>
                  <td className="num">{pct(v.fuzzersRecall)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Removing the TTL fields costs {num(u.ttl.variants[0].test.f1 - u.ttl.variants[1].test.f1)} F1, so the model does not
          depend on them. TTL alone reaches F1 {num(u.ttl.variants[2].test.f1)}. This does not rule out that the attacks were
          generated in a way that shapes these fields. Testing that needs a second, independently generated attack dataset.
        </p>
        <p className="muted">
          Top features (all features): {n.topFeatures.slice(0, 5).map((f) => f.feature).join(', ')}. The leakage scan flagged{' '}
          {n.leakageFlagged.length === 0 ? 'no feature' : n.leakageFlagged.join(', ')} at AUC 0.99 or above (highest single feature:{' '}
          {n.leakageTopFeature.feature}, AUC {num(n.leakageTopFeature.auc_separability)}).
        </p>
      </div>

      <div className="card">
        <h3>Detection by attack category</h3>
        <p>
          Recall per UNSW-NB15 attack category on the test file, for each feature set. These rates are not a single accuracy
          number: categories with few test flows (Worms has {u.ttl.perCategory.find((c) => c.category === 'Worms')?.support ?? '—'})
          are noisy.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th className="num">Test flows</th>
                <th className="num">All features</th>
                <th className="num">Without TTL</th>
                <th className="num">Only TTL</th>
              </tr>
            </thead>
            <tbody>
              {u.ttl.perCategory.map((c) => (
                <tr key={c.category}>
                  <td>
                    {c.category}{' '}
                    {c.category === 'Fuzzers' && <span className="pill">weakest</span>}
                    {c.support < 100 && <span className="pill">low sample</span>}
                  </td>
                  <td className="num">{c.support.toLocaleString()}</td>
                  <td className="num">{pct(c.all_features)}</td>
                  <td className="num">{pct(c.without_ttl)}</td>
                  <td className="num">{pct(c.only_ttl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Columns: {ttlLabel.all_features}, {ttlLabel.without_ttl}, {ttlLabel.only_ttl}.
        </p>

        <h3 style={{ marginTop: '1.5rem' }}>Confusion matrix (all features, test file)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Actual \ predicted</th>
                {n.confusionMatrix.labels.map((l) => (
                  <th key={l} className="num">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {n.confusionMatrix.labels.map((l, i) => (
                <tr key={l}>
                  <td>{l}</td>
                  {n.confusionMatrix.matrix[i].map((v, j) => (
                    <td key={j} className="num">
                      {v.toLocaleString()}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          {n.confusionMatrix.matrix[0][1].toLocaleString()} benign flows were flagged ({pct(n.test.fpr)} of benign). Missed attacks:{' '}
          {n.confusionMatrix.matrix[1][0].toLocaleString()} of {(n.confusionMatrix.matrix[1][0] + n.confusionMatrix.matrix[1][1]).toLocaleString()}.
          Most misses are Fuzzers flows.
        </p>
      </div>

      <div className="card">
        <h3>Fuzzers: the weakest category</h3>
        <p>
          Fuzzers are detected about half the time. Two approaches were tried. A multiclass model did not improve on the binary
          headline model. Per-flow ratio features did not change the result across three seeds. The chart shows recall on the
          test file for each variant; the selection rule for each was fixed on validation data.
        </p>
        <div className="chart" style={{ height: 260 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={fuzzChart} layout="vertical" margin={{ left: 10, right: 30, top: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={12} />
              <YAxis type="category" dataKey="name" width={230} fontSize={11} />
              <Tooltip formatter={(v: number) => pct(v)} />
              <Bar dataKey="recall" name="Fuzzers recall (test)" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="muted">
          Chosen on validation: {u.fuzzers.chosenOnValidation.replace('_', ' ')}. Its test Fuzzers recall is lower than the
          headline model's, so the headline model stays. A diagnostic run (not saved in the results) found that Fuzzers flows sit
          in the same score range as the top few percent of benign traffic, which is why thresholds and features alone have not
          closed the gap.
        </p>
      </div>

      <div className="card">
        <h3>What this does and does not show</h3>
        <ul>
          <li>
            The UNSW-native result is in-domain: a model trained and tested on the same dataset family. It does not show the model would work on
            a different network.
          </li>
          <li>
            The UNSW-NB15 training file has no timestamps and no IP addresses, so per-destination counts over a time window cannot be
            built from it. The dataset's own connection counters (<code>ct_*</code>) are used instead.
          </li>
          <li>
            The CIC-to-UNSW transfer used nine shared features mapped by hand. Since most features reverse direction, a different
            mapping is unlikely to be the whole explanation, but this was not tested directly.
          </li>
        </ul>
      </div>
    </>
  );
}

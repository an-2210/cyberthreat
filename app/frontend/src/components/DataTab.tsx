import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';
import { ErrorBox, useLoader } from './useLoader';

export function DataTab() {
  const { data, error } = useLoader(api.data);
  const figures = useLoader(api.figures);

  if (error) return <ErrorBox message={error} />;
  if (!data) return <p className="muted">Loading…</p>;

  const missingFeatures = Object.entries(data.missingByFeature);
  const infiniteFeatures = Object.entries(data.infiniteByFeature);

  return (
    <>
      <div className="card">
        <h2>The data</h2>
        <p>
          {data.rows.toLocaleString()} flows, {data.featureColumns} numeric features. Each row is one network flow, and
          the label says whether it is benign or which attack it belongs to. The classes are very unbalanced: the table
          below shows that benign traffic dominates, and some attacks have only a few dozen rows.
        </p>
        <p className="muted">
          Duplicates: {data.duplicateRows.toLocaleString()} exact copies (removed before training).
          Rows whose features match but whose labels disagree: {data.conflictingFeatureRows.toLocaleString()}.
          Constant features (always the same value, so useless to a model): {data.constantFeatures.length}.
        </p>
      </div>

      <div className="card">
        <h3>Flows per class</h3>
        <div className="chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.classes} layout="vertical" margin={{ left: 130, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" tickFormatter={(v) => v.toLocaleString()} fontSize={12} />
              <YAxis type="category" dataKey="label" width={130} fontSize={12} />
              <Tooltip formatter={(v: number) => v.toLocaleString()} />
              <Bar dataKey="count" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="muted">
          BENIGN makes up {data.classes[0]?.percent.toFixed(1)}% of rows, so a model that always said "benign" would score
          high accuracy while catching no attacks. That is why the Models tab also reports recall and F1.
        </p>
      </div>

      {(missingFeatures.length > 0 || infiniteFeatures.length > 0) && (
        <div className="card">
          <h3>Bad values by feature</h3>
          <p className="muted">
            Missing values are imputed during preprocessing. Infinite values come from division by a zero duration and
            are replaced the same way.
          </p>
          <table>
            <thead>
              <tr><th>Feature</th><th className="num">Missing</th><th className="num">Infinite</th></tr>
            </thead>
            <tbody>
              {[...new Set([...missingFeatures, ...infiniteFeatures].map(([k]) => k))].map((f) => (
                <tr key={f}>
                  <td>{f}</td>
                  <td className="num">{(data.missingByFeature[f] ?? 0).toLocaleString()}</td>
                  <td className="num">{(data.infiniteByFeature[f] ?? 0).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {figures.data && figures.data.length > 0 && (
        <div className="card">
          <h3>Saved figures from the pipeline</h3>
          <div className="figures">
            {figures.data
              .filter((name) => name.includes('class_distribution') || name.includes('column_quality') || name.includes('by_source'))
              .map((name) => (
                <figure key={name} style={{ margin: 0 }}>
                  <img src={api.figureUrl(name)} alt={name} />
                  <figcaption className="muted">{name}</figcaption>
                </figure>
              ))}
          </div>
        </div>
      )}
    </>
  );
}

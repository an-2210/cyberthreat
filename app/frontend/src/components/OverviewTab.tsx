import { api } from '../api';
import { ErrorBox, useLoader } from './useLoader';

const pct = (n: number, d: number) => `${((100 * n) / d).toFixed(1)}%`;
const fmt = (n: number) => n.toLocaleString();

export function OverviewTab() {
  const overview = useLoader(api.overview);
  const data = useLoader(api.data);
  const models = useLoader(api.models);

  if (overview.error) return <ErrorBox message={overview.error} />;
  if (!overview.data || !data.data || !models.data) return <p className="muted">Loading…</p>;

  const o = overview.data;
  const d = data.data;
  const best = models.data.filter((m) => m.task === 'binary').sort((a, b) => b.f1 - a.f1)[0];
  const bestMulti = models.data.filter((m) => m.task === 'multiclass').sort((a, b) => b.f1 - a.f1)[0];

  return (
    <>
      <div className="card">
        <h2>What this project does</h2>
        <p>
          It learns to tell normal network traffic from attacks. The input is a flow record: one row per network
          connection, with 77 numeric features such as packet counts, byte counts and timing. The models were trained on
          the CIC-IDS2017 dataset, which has labelled traffic for 15 classes (normal plus 14 attack types).
        </p>
        <p>
          Two questions are answered. <strong>Binary:</strong> is this flow an attack or benign? <strong>Multiclass:</strong>{' '}
          if it is an attack, which kind?
        </p>
      </div>

      <div className="grid">
        <div className="card stat">
          <div className="value">{fmt(d.rows)}</div>
          <div className="label">flows in the dataset</div>
        </div>
        <div className="card stat">
          <div className="value">{best ? best.f1.toFixed(3) : '–'}</div>
          <div className="label">best attack-detection F1 (binary)</div>
        </div>
        <div className="card stat">
          <div className="value">{bestMulti ? bestMulti.f1.toFixed(3) : '–'}</div>
          <div className="label">best macro F1 across attack types</div>
        </div>
        <div className="card stat">
          <div className="value">{o.featureCount}</div>
          <div className="label">features fed to the models</div>
        </div>
      </div>

      <div className="card">
        <h2>How the pipeline works</h2>
        <ol className="steps">
          <li>
            <div>
              <strong>Load and audit.</strong> Read the 8 CSV files and check quality: {fmt(d.duplicateRows)} duplicate
              rows ({pct(d.duplicateRows, d.rows)}), {fmt(d.missingCells)} missing and {fmt(d.infiniteCells)} infinite values.
            </div>
          </li>
          <li>
            <div>
              <strong>Remove duplicates and split.</strong> Exact duplicates are dropped before splitting, so the same flow
              cannot appear in both training and test data. The rest is split stratified into train, validation and test:{' '}
              {fmt(o.splitSizes.train_before_sampling)} / {fmt(o.splitSizes.validation)} / {fmt(o.splitSizes.test)}.
            </div>
          </li>
          <li>
            <div>
              <strong>Clean and scale, fitted on training data only.</strong> Constant columns are removed, values are
              imputed and scaled. Fitting only on training rows stops test information leaking into the models.
            </div>
          </li>
          <li>
            <div>
              <strong>Train.</strong> Random Forest and XGBoost are trained on a stratified sample of{' '}
              {fmt(o.splitSizes.train_used)} training rows, with the same rows used for every model.
            </div>
          </li>
          <li>
            <div>
              <strong>Evaluate on the untouched test set.</strong> Precision, recall, F1, false-positive rate and confusion
              matrices are on the Models tab.
            </div>
          </li>
        </ol>
      </div>

      {o.leakageFlaggedFeatures.length === 0 && (
        <p className="muted">
          Leakage check: no single feature alone predicts the label almost perfectly (AUC ≥ 0.99), so none was flagged.
        </p>
      )}
      <p className="muted">
        Last run took {o.elapsedSeconds.toFixed(0)} s · seed {o.seed} · data source: {o.dataSource}
      </p>
    </>
  );
}

import React, { useState } from 'react';
import { ConfusionMatrixData } from '../../types/evaluation';
import { Table, Percent } from 'lucide-react';

interface ConfusionMatrixProps {
  data: ConfusionMatrixData | null;
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ data }) => {
  const [showNormalized, setShowNormalized] = useState(false);

  if (!data || !data.matrix || data.matrix.length === 0) {
    return (
      <div className="soc-card p-6 text-center text-xs font-mono text-slate-400 border border-slate-800">
        No confusion matrix evaluated for this model yet.
      </div>
    );
  }

  const { labels, matrix, modelName } = data;

  // Calculate totals and max for color intensity
  const totalSamples = matrix.flatMap(row => row).reduce((a, b) => a + b, 0);
  const maxVal = Math.max(...matrix.flatMap(row => row));

  const getCellBg = (value: number, isDiagonal: boolean) => {
    const ratio = value / (maxVal || 1);
    if (isDiagonal) {
      if (ratio > 0.7) return 'bg-emerald-950/80 text-emerald-300 border-emerald-700/60 font-bold';
      if (ratio > 0.3) return 'bg-emerald-950/50 text-emerald-300 border-emerald-800/40';
      return 'bg-emerald-950/20 text-emerald-400 border-emerald-900/30';
    } else {
      if (value === 0) return 'bg-slate-900/40 text-slate-500 border-slate-800/40';
      if (ratio > 0.2) return 'bg-rose-950/80 text-rose-300 border-rose-700/60 font-bold';
      return 'bg-rose-950/30 text-rose-400 border-rose-900/40';
    }
  };

  return (
    <div className="soc-card border border-soc-border p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-soc-border/60 pb-2">
        <div>
          <h4 className="text-xs font-mono font-bold uppercase text-slate-200">
            Confusion Matrix: <span className="text-cyan-400">{modelName}</span>
          </h4>
          <p className="text-[10px] font-mono text-slate-400">
            Rows: Actual Class | Columns: Predicted Class
          </p>
        </div>

        <button
          onClick={() => setShowNormalized(!showNormalized)}
          className="flex items-center space-x-1 px-2.5 py-1 rounded text-[10px] font-mono border border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800 transition-colors"
        >
          {showNormalized ? <Table className="w-3 h-3" /> : <Percent className="w-3 h-3" />}
          <span>{showNormalized ? 'Raw Counts' : 'Normalized %'}</span>
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-center text-xs font-mono border-collapse">
          <thead>
            <tr>
              <th className="p-2 text-[10px] text-slate-400 border border-slate-800 bg-slate-950">
                Actual \ Pred
              </th>
              {labels.map((label) => (
                <th key={label} className="p-2 text-[10px] font-semibold text-cyan-400 border border-slate-800 bg-slate-950">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rIdx) => (
              <tr key={labels[rIdx] || rIdx}>
                <th className="p-2 text-[10px] font-semibold text-slate-300 border border-slate-800 bg-slate-950 text-left">
                  {labels[rIdx]}
                </th>
                {row.map((val, cIdx) => {
                  const isDiagonal = rIdx === cIdx;
                  const displayValue = showNormalized
                    ? `${((val / (row.reduce((a, b) => a + b, 0) || 1)) * 100).toFixed(1)}%`
                    : val.toLocaleString();

                  return (
                    <td
                      key={cIdx}
                      className={`p-2 border text-xs transition-colors ${getCellBg(val, isDiagonal)}`}
                    >
                      {displayValue}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

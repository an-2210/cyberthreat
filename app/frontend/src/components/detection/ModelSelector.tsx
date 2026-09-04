import React from 'react';
import { MLModelInfo } from '../../types/detection';
import { ShieldCheck, Lock, CheckCircle2 } from 'lucide-react';

interface ModelSelectorProps {
  models: MLModelInfo[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModelId,
  onSelectModel
}) => {
  return (
    <div className="soc-card border border-soc-border p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-soc-border/60 pb-2">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
          <ShieldCheck className="w-4 h-4 text-cyan-400" />
          <span>Select Model for Inference</span>
        </h3>
        <span className="text-[10px] font-mono text-emerald-400">
          Phase 3 Supervised Baselines
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
        {models.map((model) => {
          const isSelected = model.id === selectedModelId;
          const isSupported = model.isSupportedByBackend;

          return (
            <div
              key={model.id}
              onClick={() => isSupported && onSelectModel(model.id)}
              className={`p-3 rounded-md border text-left transition-all ${
                !isSupported
                  ? 'bg-slate-950/40 border-slate-800/60 opacity-50 cursor-not-allowed'
                  : isSelected
                  ? 'bg-cyan-950/60 border-cyan-500/70 shadow-soc-glow-cyan cursor-pointer'
                  : 'bg-slate-900/80 border-soc-border hover:border-slate-700 cursor-pointer'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-slate-200 truncate">
                  {model.name}
                </span>
                {isSupported ? (
                  isSelected ? (
                    <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                  ) : (
                    <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0"></span>
                  )
                ) : (
                  <Lock className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                )}
              </div>

              <div className="mt-2 flex items-center justify-between text-[10px] font-mono text-slate-400">
                <span>Phase {model.phase}</span>
                {isSupported ? (
                  <span className="text-emerald-400 font-semibold">
                    {model.accuracy ? `Acc: ${(model.accuracy * 100).toFixed(1)}%` : 'Ready'}
                  </span>
                ) : (
                  <span className="text-slate-500">Not Implemented</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

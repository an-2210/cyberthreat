import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';

interface ResearchNoticeProps {
  phaseText?: string;
}

export const ResearchNotice: React.FC<ResearchNoticeProps> = ({
  phaseText = "Backend system active up to Phase 3 (Baseline Models). Future phases (SHAP, MITRE, RAG, LLM) remain disabled."
}) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3 flex items-start space-x-3 mb-6">
      <div className="p-1.5 bg-cyan-950/80 text-cyan-400 border border-cyan-800/60 rounded shrink-0 mt-0.5">
        <Info className="w-4 h-4" />
      </div>
      <div className="text-xs font-mono">
        <span className="font-bold text-slate-200 uppercase tracking-wide mr-2">Research Integrity Safeguard:</span>
        <span className="text-slate-400">{phaseText}</span>
        <span className="text-slate-500 block text-[10px] mt-0.5">
          Metrics are generated exclusively from executed backend experiments. Un-evaluated values display "—" or "Not evaluated".
        </span>
      </div>
    </div>
  );
};

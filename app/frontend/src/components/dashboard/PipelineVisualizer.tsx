import React from 'react';
import {
  Database,
  CheckCircle2,
  Sliders,
  Cpu,
  ShieldAlert,
  Brain,
  Layers,
  BrainCircuit,
  Crosshair,
  BookOpen,
  FileSpreadsheet,
  ArrowRight,
  Lock
} from 'lucide-react';

export const PipelineVisualizer: React.FC = () => {
  const activeStages = [
    { name: 'Data Ingestion', phase: 'Phase 1', icon: Database, desc: 'PCAP / CSV Adapters' },
    { name: 'Validation & Audit', phase: 'Phase 1', icon: CheckCircle2, desc: 'Quality & Null Audit' },
    { name: 'Leakage-Safe Preprocess', phase: 'Phase 2', icon: Sliders, desc: 'Train Split Scalers' },
    { name: 'Feature Engineering', phase: 'Phase 2', icon: Cpu, desc: 'Variance & Scaling' },
    { name: 'Baseline ML Detection', phase: 'Phase 3', icon: ShieldAlert, desc: 'RF / XGBoost / SVM' },
  ];

  const disabledStages = [
    { name: 'Anomaly Detection', phase: 'Phase 4', icon: Brain, desc: 'Autoencoders' },
    { name: 'Decision Fusion', phase: 'Phase 5', icon: Layers, desc: 'Hybrid α-Score' },
    { name: 'SHAP Explainability', phase: 'Phase 8', icon: BrainCircuit, desc: 'TreeSHAP Attribution' },
    { name: 'MITRE ATT&CK', phase: 'Phase 9', icon: Crosshair, desc: 'Technique Mapping' },
    { name: 'RAG Retrieval', phase: 'Phase 11', icon: BookOpen, desc: 'FAISS Vector Index' },
    { name: 'LLM CTI Report', phase: 'Phase 12', icon: FileSpreadsheet, desc: 'Grounding Report' },
  ];

  return (
    <div className="soc-card border border-soc-border p-5 space-y-4 bg-[#0d1322]/80">
      <div className="flex items-center justify-between border-b border-soc-border/60 pb-3">
        <div>
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span>AI Architecture Pipeline Flow</span>
          </h3>
          <p className="text-[11px] font-mono text-slate-400 mt-0.5">
            Active Phases 1–3 vs Future Modular Integration Pipeline (Phases 4–12)
          </p>
        </div>
        <div className="flex items-center space-x-3 text-[10px] font-mono">
          <span className="flex items-center text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
            ACTIVE (PHASE 1-3)
          </span>
          <span className="flex items-center text-slate-500">
            <Lock className="w-3 h-3 mr-1 text-slate-500" />
            DISABLED (PHASE 4+)
          </span>
        </div>
      </div>

      {/* Pipeline Track */}
      <div className="space-y-4">
        {/* Active Track (Phases 1-3) */}
        <div>
          <div className="text-[10px] font-mono text-emerald-400 font-semibold uppercase tracking-wider mb-2">
            Implemented Execution Pipeline (Phases 1–3)
          </div>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
            {activeStages.map((stage, idx) => (
              <div key={stage.name} className="relative group">
                <div className="p-3 rounded-md bg-slate-900/90 border border-emerald-900/50 hover:border-emerald-500/60 transition-all flex flex-col justify-between h-24 shadow-soc-glow-emerald">
                  <div className="flex items-center justify-between">
                    <stage.icon className="w-4 h-4 text-emerald-400" />
                    <span className="text-[9px] font-mono bg-emerald-950 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-800/60">
                      {stage.phase}
                    </span>
                  </div>
                  <div>
                    <div className="text-xs font-mono font-bold text-slate-200 truncate">
                      {stage.name}
                    </div>
                    <div className="text-[10px] font-mono text-slate-400 truncate">
                      {stage.desc}
                    </div>
                  </div>
                </div>
                {idx < activeStages.length - 1 && (
                  <div className="hidden md:block absolute -right-2.5 top-1/2 -translate-y-1/2 z-10 text-emerald-500/80">
                    <ArrowRight className="w-4 h-4" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Pipeline Divider */}
        <div className="flex items-center justify-center my-2">
          <div className="border-t border-dashed border-soc-border flex-1"></div>
          <span className="mx-4 text-[10px] font-mono text-slate-500 bg-slate-900 px-3 py-1 rounded-full border border-slate-800">
            Phase 3 Supervised Output ↓ Phase 4+ Advanced Pipeline Integration
          </span>
          <div className="border-t border-dashed border-soc-border flex-1"></div>
        </div>

        {/* Future Track (Phases 4-12) */}
        <div>
          <div className="text-[10px] font-mono text-slate-400 font-semibold uppercase tracking-wider mb-2 flex items-center justify-between">
            <span>Future Modular Architecture (Phase 4+)</span>
            <span className="text-[9px] font-mono text-amber-400/80">Pending Implementation</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
            {disabledStages.map((stage) => (
              <div
                key={stage.name}
                className="p-3 rounded-md bg-slate-950/60 border border-slate-800/80 opacity-60 flex flex-col justify-between h-24 hover:opacity-80 transition-opacity"
              >
                <div className="flex items-center justify-between">
                  <stage.icon className="w-4 h-4 text-slate-500" />
                  <span className="text-[9px] font-mono bg-slate-900 text-slate-400 px-1.5 py-0.5 rounded border border-slate-800">
                    {stage.phase}
                  </span>
                </div>
                <div>
                  <div className="text-[11px] font-mono font-semibold text-slate-400 truncate flex items-center">
                    <Lock className="w-3 h-3 mr-1 shrink-0 text-slate-500" />
                    <span className="truncate">{stage.name}</span>
                  </div>
                  <div className="text-[9px] font-mono text-slate-500 truncate">
                    {stage.desc}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

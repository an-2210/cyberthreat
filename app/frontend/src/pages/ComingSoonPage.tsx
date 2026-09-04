import React from 'react';
import { Lock, BrainCircuit, Crosshair, BookOpen, FileSpreadsheet, ArrowRight, Shield, Layers } from 'lucide-react';

interface ComingSoonPageProps {
  moduleKey: 'explainability' | 'mitre-attack' | 'threat-intelligence' | 'cti-reports';
}

export const ComingSoonPage: React.FC<ComingSoonPageProps> = ({ moduleKey }) => {
  const moduleConfigs = {
    'explainability': {
      title: 'SHAP Explainability Engine',
      phase: 'Phase 8',
      icon: BrainCircuit,
      statusMessage: 'SHAP-based prediction explanations will be available after Phase 8.',
      description: 'TreeSHAP and KernelSHAP attribution engines providing exact feature-importance values for individual alert classifications.',
      expectedInputs: 'Phase 3 Supervised / Phase 5 Fusion Model Weights + Ingestion Features',
      outputArtifacts: 'Waterfall Plots, Summary Plots, Feature Contribution Matrices',
      techStack: ['SHAP', 'TreeExplainer', 'KernelExplainer', 'Matplotlib']
    },
    'mitre-attack': {
      title: 'MITRE ATT&CK Mapping Engine',
      phase: 'Phase 9',
      icon: Crosshair,
      statusMessage: 'Behavior-to-technique mapping will be available after Phase 9.',
      description: 'Automated correlation of classified attack vectors to official MITRE ATT&CK Enterprise TTP IDs (e.g. T1498 DDoS, T1110 Brute Force).',
      expectedInputs: 'Classifier Label Output + Anomaly Feature Distances',
      outputArtifacts: 'STIX 2.1 Threat Objects, Matrix Heatmap Overlays',
      techStack: ['MITRE STIX API', 'TAXII 2.1 Server', 'Python stix2']
    },
    'threat-intelligence': {
      title: 'RAG Threat Intelligence Knowledge Base',
      phase: 'Phase 11',
      icon: BookOpen,
      statusMessage: 'RAG-powered intelligence retrieval will be available after Phase 11.',
      description: 'Vector-indexed retrieval of CVE advisories, NVD databases, and open-source CTI threat actor profiles using FAISS similarity search.',
      expectedInputs: 'MITRE TTP IDs + Extracted Attack Subnet Headers',
      outputArtifacts: 'Ranked Grounding Passages, CVE Matches, Threat Actor Context',
      techStack: ['FAISS', 'SentenceTransformers (all-MiniLM-L6-v2)', 'LangChain']
    },
    'cti-reports': {
      title: 'Evidence-Grounded LLM CTI Incident Reports',
      phase: 'Phase 12',
      icon: FileSpreadsheet,
      statusMessage: 'Evidence-grounded LLM reports will be available after Phase 12.',
      description: 'Constrained, hallucination-safe synthesis of executive SOC incident reports grounded strictly in retrieved RAG evidence.',
      expectedInputs: 'RAG Context Passages + Incident Metrics + SHAP Attributions',
      outputArtifacts: 'Structured Markdown Reports, Actionable Mitigations',
      techStack: ['LLaMA-3 8B / Mistral-7B', 'Outlines / Guidance', 'FastAPI Async']
    }
  };

  const config = moduleConfigs[moduleKey];
  const Icon = config.icon;

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-4">
      {/* Module Title Header */}
      <div className="flex items-center space-x-3">
        <div className="p-3 bg-slate-900 text-slate-400 border border-slate-800 rounded-lg">
          <Icon className="w-6 h-6 text-cyan-400" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-mono font-bold text-slate-100 uppercase tracking-wide">
              {config.title}
            </h1>
            <span className="px-2 py-0.5 text-[10px] font-mono bg-amber-950/80 text-amber-300 border border-amber-800 rounded">
              {config.phase}
            </span>
          </div>
          <p className="text-xs font-mono text-slate-400 mt-0.5">
            Module Architecture & Integration Roadmap Preview
          </p>
        </div>
      </div>

      {/* Primary Status Banner */}
      <div className="soc-card border-amber-900/60 bg-amber-950/20 p-6 flex items-start space-x-4">
        <div className="p-2 bg-amber-900/40 text-amber-400 border border-amber-800 rounded-full shrink-0">
          <Lock className="w-5 h-5" />
        </div>
        <div className="space-y-1">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-amber-300">
            Backend Component Not Implemented Yet
          </h3>
          <p className="text-xs font-mono text-slate-300 leading-relaxed font-semibold">
            "{config.statusMessage}"
          </p>
          <p className="text-[11px] font-mono text-slate-400">
            Per research integrity guidelines, live results, attribution charts, or AI report text are not fabricated prior to executing the backend pipeline.
          </p>
        </div>
      </div>

      {/* Target Module Specifications Preview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="soc-card border border-soc-border p-5 space-y-3">
          <h4 className="text-xs font-mono font-bold uppercase text-slate-200 flex items-center space-x-2">
            <Shield className="w-4 h-4 text-cyan-400" />
            <span>Planned Functional Architecture</span>
          </h4>
          <p className="text-xs font-mono text-slate-300 leading-relaxed">
            {config.description}
          </p>
          <div className="pt-3 border-t border-soc-border/60">
            <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold block mb-1">
              Required Upstream Inputs:
            </span>
            <span className="text-xs font-mono text-cyan-300 bg-slate-900 px-2.5 py-1 rounded border border-slate-800 block">
              {config.expectedInputs}
            </span>
          </div>
        </div>

        <div className="soc-card border border-soc-border p-5 space-y-3">
          <h4 className="text-xs font-mono font-bold uppercase text-slate-200 flex items-center space-x-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            <span>Target Output & Technology Stack</span>
          </h4>
          <div>
            <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold block mb-1">
              Generated Research Artifacts:
            </span>
            <span className="text-xs font-mono text-slate-200 bg-slate-900 px-2.5 py-1 rounded border border-slate-800 block">
              {config.outputArtifacts}
            </span>
          </div>
          <div className="pt-2">
            <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold block mb-1">
              Target Technical Libraries:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {config.techStack.map(tech => (
                <span key={tech} className="text-[10px] font-mono bg-cyan-950/60 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800/60">
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

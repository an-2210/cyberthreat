import React, { useState, useEffect } from 'react';
import { FlaskConical, Search, Filter, Lock, FileCode, CheckCircle2, ChevronRight, X } from 'lucide-react';
import { DataTable, Column } from '../components/common/DataTable';
import { StatusBadge } from '../components/common/StatusBadge';
import { EmptyState } from '../components/common/EmptyState';
import { experimentApi } from '../api/experimentApi';
import { ExperimentRecord, RoadmapPhase } from '../types/experiment';

export const ExperimentsPage: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [roadmap, setRoadmap] = useState<RoadmapPhase[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentRecord | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const [expList, roadList] = await Promise.all([
        experimentApi.getExperiments(),
        experimentApi.getRoadmapPhases()
      ]);
      setExperiments(expList);
      setRoadmap(roadList);
    } catch (err) {
      console.warn("Failed fetching experiment records:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExperiments();
    const handleDemoChange = () => fetchExperiments();
    window.addEventListener('demo-mode-change', handleDemoChange);
    return () => window.removeEventListener('demo-mode-change', handleDemoChange);
  }, []);

  const columns: Column<ExperimentRecord>[] = [
    { key: 'id', header: 'Experiment ID', render: (row) => <span className="font-bold text-cyan-400">{row.id}</span> },
    { key: 'phaseName', header: 'Phase', render: (row) => `Phase ${row.phase} — ${row.phaseName}` },
    { key: 'datasetName', header: 'Dataset' },
    { key: 'modelName', header: 'Model', render: (row) => <span className="font-bold text-slate-200">{row.modelName}</span> },
    { key: 'featuresCount', header: 'Features', render: (row) => row.featuresCount ?? '—' },
    { key: 'accuracy', header: 'Accuracy', render: (row) => row.accuracy !== null ? `${(row.accuracy * 100).toFixed(2)}%` : '—' },
    { key: 'macroF1', header: 'Macro F1', render: (row) => row.macroF1 !== null ? row.macroF1.toFixed(4) : '—' },
    { key: 'mcc', header: 'MCC', render: (row) => row.mcc !== null ? row.mcc.toFixed(4) : '—' },
    { key: 'timestamp', header: 'Timestamp' },
    { key: 'status', header: 'Status', render: (row) => <StatusBadge status={row.status} type="status" /> },
    {
      key: 'action',
      header: 'Details',
      render: (row) => (
        <button
          onClick={() => setSelectedExperiment(row)}
          className="px-2 py-1 rounded text-[10px] font-mono bg-slate-900 border border-slate-700 text-slate-300 hover:text-cyan-400 hover:border-cyan-800 transition-colors"
        >
          Inspect
        </button>
      )
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-mono font-bold text-slate-100 uppercase tracking-wide flex items-center space-x-2">
          <FlaskConical className="w-5 h-5 text-cyan-400" />
          <span>Experiment Tracking & Research Roadmap</span>
        </h1>
        <p className="text-xs font-mono text-slate-400 mt-1">
          Trace hyperparameter configurations, metrics logs, and planned research phases
        </p>
      </div>

      {/* Executed Experiments Table */}
      <div className="space-y-3">
        <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
          <FileCode className="w-4 h-4 text-emerald-400" />
          <span>Executed Experiments (Phases 1–3)</span>
        </h2>

        {experiments.length === 0 ? (
          <EmptyState
            title="No experiment runs recorded"
            description="Execute Phase 1-3 baseline scripts to populate the experiment registry."
            icon={FlaskConical}
          />
        ) : (
          <DataTable
            columns={columns}
            data={experiments}
            searchPlaceholder="Search by ID, model, or dataset..."
          />
        )}
      </div>

      {/* Future Research Roadmap */}
      <div className="space-y-3 pt-4 border-t border-soc-border">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
            <Lock className="w-4 h-4 text-amber-400" />
            <span>Planned Future Experiments (Phases 4–12 Roadmap)</span>
          </h2>
          <span className="text-[10px] font-mono text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-800">
            STATUS: NOT IMPLEMENTED
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {roadmap.filter(p => p.status === 'NOT IMPLEMENTED').map((phase) => (
            <div
              key={phase.phaseNumber}
              className="soc-card border border-slate-800/80 bg-slate-950/50 p-4 space-y-2 opacity-75 hover:opacity-100 transition-opacity"
            >
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-2">
                <span className="text-xs font-mono font-bold text-slate-300">
                  Phase {phase.phaseNumber} — {phase.title}
                </span>
                <span className="text-[9px] font-mono bg-slate-900 text-slate-400 px-1.5 py-0.5 rounded border border-slate-800">
                  NOT IMPLEMENTED
                </span>
              </div>
              <p className="text-xs font-mono text-slate-400 leading-relaxed">
                {phase.description}
              </p>
              <div className="pt-2 border-t border-slate-800/60 flex flex-wrap gap-1">
                {phase.keyTechnologies.map(tech => (
                  <span key={tech} className="text-[9px] font-mono bg-slate-900 text-cyan-400/80 px-1.5 py-0.5 rounded border border-slate-800">
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Experiment Detail Modal */}
      {selectedExperiment && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0e1424] border border-soc-border rounded-lg max-w-xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-soc-border pb-3">
              <div>
                <h3 className="text-sm font-mono font-bold text-cyan-400">
                  {selectedExperiment.id}: {selectedExperiment.modelName}
                </h3>
                <p className="text-xs font-mono text-slate-400">
                  Phase {selectedExperiment.phase} — {selectedExperiment.phaseName}
                </p>
              </div>
              <button
                onClick={() => setSelectedExperiment(null)}
                className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 font-mono text-xs">
              <div className="grid grid-cols-2 gap-2 p-3 bg-slate-900 rounded border border-slate-800">
                <div><span className="text-slate-400">Accuracy:</span> <span className="text-emerald-400 font-bold">{selectedExperiment.accuracy ? `${(selectedExperiment.accuracy * 100).toFixed(2)}%` : '—'}</span></div>
                <div><span className="text-slate-400">Macro F1:</span> <span className="text-cyan-400 font-bold">{selectedExperiment.macroF1 ? selectedExperiment.macroF1.toFixed(4) : '—'}</span></div>
                <div><span className="text-slate-400">Training Time:</span> <span className="text-slate-200">{selectedExperiment.trainingTimeSec}s</span></div>
                <div><span className="text-slate-400">Inference Time:</span> <span className="text-slate-200">{selectedExperiment.inferenceTimeMs}ms/sample</span></div>
              </div>

              <div>
                <span className="text-slate-400 block mb-1 uppercase font-semibold text-[10px]">Hyperparameter Configurations:</span>
                <pre className="p-3 bg-slate-950 rounded border border-slate-800 text-[11px] text-cyan-300 overflow-x-auto">
                  {JSON.stringify(selectedExperiment.hyperparameters, null, 2)}
                </pre>
              </div>

              {selectedExperiment.notes && (
                <div>
                  <span className="text-slate-400 block mb-1 uppercase font-semibold text-[10px]">Execution Log Notes:</span>
                  <p className="p-2 bg-slate-900/60 rounded border border-slate-800 text-slate-300">
                    {selectedExperiment.notes}
                  </p>
                </div>
              )}
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedExperiment(null)}
                className="px-4 py-1.5 rounded text-xs font-mono bg-slate-800 text-slate-200 hover:bg-slate-700"
              >
                Close Window
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

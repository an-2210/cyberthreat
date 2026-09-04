import React, { useState, useEffect } from 'react';
import { ShieldAlert, Upload, Play, FileText, RefreshCw, AlertCircle } from 'lucide-react';
import { ModelSelector } from '../components/detection/ModelSelector';
import { DataTable, Column } from '../components/common/DataTable';
import { StatusBadge } from '../components/common/StatusBadge';
import { EmptyState } from '../components/common/EmptyState';
import { detectionApi } from '../api/detectionApi';
import { MLModelInfo, DetectionEvent } from '../types/detection';

export const ThreatDetectionPage: React.FC = () => {
  const [models, setModels] = useState<MLModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('xgboost_baseline');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [predictions, setPredictions] = useState<DetectionEvent[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const list = await detectionApi.getAvailableModels();
        setModels(list);
        // Find first ready supported model
        const firstSupported = list.find(m => m.isSupportedByBackend);
        if (firstSupported) setSelectedModelId(firstSupported.id);
      } catch (err) {
        console.warn("Failed fetching baseline models:", err);
      }
    };
    fetchModels();

    const handleDemoChange = () => fetchModels();
    window.addEventListener('demo-mode-change', handleDemoChange);
    return () => window.removeEventListener('demo-mode-change', handleDemoChange);
  }, []);

  const handleRunPrediction = async () => {
    if (!selectedFile) {
      setErrorMessage("Please select a CSV file containing flow features for prediction.");
      return;
    }

    setIsPredicting(true);
    setErrorMessage(null);

    try {
      const res = await detectionApi.predictBatch(selectedFile, selectedModelId);
      if (res && res.predictions) {
        setPredictions(res.predictions);
      } else {
        setPredictions([]);
      }
    } catch (err: any) {
      setErrorMessage(err?.response?.data?.message || "Inference failed. Verify model compatibility and API connection.");
    } finally {
      setIsPredicting(false);
    }
  };

  const columns: Column<DetectionEvent>[] = [
    { key: 'timestamp', header: 'Timestamp', sortable: true },
    {
      key: 'source',
      header: 'Source (IP:Port)',
      render: (row) => row.sourceIp ? `${row.sourceIp}:${row.sourcePort ?? '—'}` : '—'
    },
    {
      key: 'destination',
      header: 'Destination (IP:Port)',
      render: (row) => row.destinationIp ? `${row.destinationIp}:${row.destinationPort ?? '—'}` : '—'
    },
    { key: 'protocol', header: 'Protocol' },
    {
      key: 'prediction',
      header: 'Prediction Label',
      render: (row) => (
        <span className="font-bold text-slate-100">{row.prediction}</span>
      )
    },
    {
      key: 'confidence',
      header: 'Confidence',
      render: (row) => row.confidence !== null ? `${(row.confidence * 100).toFixed(1)}%` : '—'
    },
    {
      key: 'severity',
      header: 'Severity',
      render: (row) => <StatusBadge status={row.severity} type="severity" />
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => <StatusBadge status={row.status} type="status" />
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-mono font-bold text-slate-100 uppercase tracking-wide flex items-center space-x-2">
          <ShieldAlert className="w-5 h-5 text-cyan-400" />
          <span>ML Threat Detection & Batch Inference (Phase 3)</span>
        </h1>
        <p className="text-xs font-mono text-slate-400 mt-1">
          Execute baseline supervised model inference on network flow features
        </p>
      </div>

      {/* Model Selector Component */}
      <ModelSelector
        models={models}
        selectedModelId={selectedModelId}
        onSelectModel={(id) => setSelectedModelId(id)}
      />

      {/* Inference Controls */}
      <div className="soc-card border border-soc-border p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-soc-border/60 pb-3">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
            <Upload className="w-4 h-4 text-cyan-400" />
            <span>Upload Network Flow CSV for Inference</span>
          </h3>
          <span className="text-[10px] font-mono text-slate-400">
            Model Selected: <span className="text-cyan-400 font-bold">{selectedModelId}</span>
          </span>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-3 sm:space-y-0 sm:space-x-3">
          <label className="flex-1 flex items-center justify-between px-3 py-2.5 rounded-md bg-slate-900 border border-dashed border-soc-border cursor-pointer hover:border-cyan-500/60 transition-colors">
            <div className="flex items-center space-x-2 truncate">
              <FileText className="w-4 h-4 text-cyan-400 shrink-0" />
              <span className="text-xs font-mono text-slate-300 truncate">
                {selectedFile ? selectedFile.name : 'Select CSV flow records file...'}
              </span>
            </div>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setSelectedFile(e.target.files[0]);
                  setErrorMessage(null);
                }
              }}
              className="hidden"
            />
          </label>

          <button
            onClick={handleRunPrediction}
            disabled={isPredicting || !selectedFile}
            className="px-5 py-2.5 rounded-md text-xs font-mono font-bold uppercase bg-cyan-950/90 text-cyan-300 border border-cyan-800/80 hover:bg-cyan-900 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shrink-0"
          >
            {isPredicting ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                <span>Running Inference...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 text-cyan-400" />
                <span>Run Prediction</span>
              </>
            )}
          </button>
        </div>

        {errorMessage && (
          <div className="p-3 bg-rose-950/40 border border-rose-900/60 rounded-md text-xs font-mono text-rose-300 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}
      </div>

      {/* Prediction Results Table */}
      <div className="space-y-3">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-emerald-400" />
          <span>Prediction Results ({predictions.length} Events)</span>
        </h3>

        {predictions.length === 0 ? (
          <EmptyState
            title="No prediction results generated yet"
            description="Select a baseline model, upload a network flow CSV dataset, and click 'Run Prediction' to evaluate threat events."
            icon={ShieldAlert}
          />
        ) : (
          <DataTable
            columns={columns}
            data={predictions}
            emptyTitle="No threat records"
            searchPlaceholder="Search IP, protocol, or attack label..."
          />
        )}
      </div>
    </div>
  );
};

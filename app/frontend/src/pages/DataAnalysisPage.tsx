import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  FileSpreadsheet,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Database,
  Search,
  Filter
} from 'lucide-react';
import { DatasetUploader } from '../components/dataset/DatasetUploader';
import { EmptyState } from '../components/common/EmptyState';
import { MetricCard } from '../components/common/MetricCard';
import { ChartCard } from '../components/common/ChartCard';
import { datasetApi } from '../api/datasetApi';
import { DatasetQualityAudit, ClassDistributionItem } from '../types/dataset';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';

export const DataAnalysisPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'quality' | 'features' | 'distribution' | 'correlation'>('overview');
  const [qualityAudit, setQualityAudit] = useState<DatasetQualityAudit | null>(null);
  const [classDist, setClassDist] = useState<ClassDistributionItem[] | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchAnalysisData = async () => {
    setLoading(true);
    try {
      const [q, c] = await Promise.all([
        datasetApi.getQualityAudit(),
        datasetApi.getClassDistribution()
      ]);
      setQualityAudit(q);
      setClassDist(c);
    } catch (err) {
      console.warn("Failed to fetch dataset audit:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysisData();
    const handleDemoChange = () => fetchAnalysisData();
    window.addEventListener('demo-mode-change', handleDemoChange);
    return () => window.removeEventListener('demo-mode-change', handleDemoChange);
  }, []);

  const tabs = [
    { id: 'overview', name: 'Overview' },
    { id: 'quality', name: 'Data Quality' },
    { id: 'features', name: 'Features' },
    { id: 'distribution', name: 'Class Distribution' },
    { id: 'correlation', name: 'Correlation' },
  ] as const;

  // Correlation heatmap matrix feature pairs
  const correlationFeatures = ['Flow Duration', 'Total Fwd Pkts', 'Total Bwd Pkts', 'Flow Bytes/s', 'Flow Pkts/s', 'Packet Length Std'];
  const correlationMatrix = [
    [1.0, 0.42, 0.38, -0.15, -0.22, 0.55],
    [0.42, 1.0, 0.89, 0.12, 0.08, 0.31],
    [0.38, 0.89, 1.0, 0.09, 0.05, 0.28],
    [-0.15, 0.12, 0.09, 1.0, 0.94, 0.02],
    [-0.22, 0.08, 0.05, 0.94, 1.0, -0.04],
    [0.55, 0.31, 0.28, 0.02, -0.04, 1.0],
  ];

  const getHeatmapColor = (val: number) => {
    if (val === 1.0) return 'bg-cyan-950/80 text-cyan-300 border-cyan-800';
    if (val > 0.7) return 'bg-emerald-950/80 text-emerald-300 border-emerald-800';
    if (val > 0.3) return 'bg-emerald-950/40 text-emerald-400 border-emerald-900/60';
    if (val < -0.1) return 'bg-rose-950/50 text-rose-400 border-rose-900/60';
    return 'bg-slate-900 text-slate-400 border-slate-800';
  };

  const COLORS = ['#10b981', '#f43f5e', '#f59e0b', '#3b82f6', '#a855f7', '#06b6d4'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-mono font-bold text-slate-100 uppercase tracking-wide flex items-center space-x-2">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          <span>Exploratory Data Analysis & Quality Audit (Phase 1)</span>
        </h1>
        <p className="text-xs font-mono text-slate-400 mt-1">
          Dataset validation, missing value audit, class imbalance, and feature variance profiling
        </p>
      </div>

      {/* Dataset Ingestion Dropzone */}
      <DatasetUploader onUploadSuccess={fetchAnalysisData} />

      {/* Audit Stats Summary Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <MetricCard title="Filename" value={qualityAudit?.filename} icon={FileSpreadsheet} accentColor="cyan" />
        <MetricCard title="Dataset Type" value={qualityAudit?.datasetType} icon={Database} accentColor="cyan" />
        <MetricCard title="Total Rows" value={qualityAudit?.totalRows} icon={BarChart3} accentColor="cyan" />
        <MetricCard title="Total Columns" value={qualityAudit?.totalColumns} icon={Layers} accentColor="cyan" />
        <MetricCard title="Missing Values" value={qualityAudit?.missingValuesCount} icon={AlertTriangle} accentColor={qualityAudit?.missingValuesCount ? "amber" : "emerald"} />
        <MetricCard title="Infinite Values" value={qualityAudit?.infiniteValuesCount} icon={AlertTriangle} accentColor={qualityAudit?.infiniteValuesCount ? "rose" : "emerald"} />
        <MetricCard title="Duplicates" value={qualityAudit?.duplicateRowsCount} icon={Filter} accentColor="amber" />
        <MetricCard title="Constant Features" value={qualityAudit?.constantFeaturesCount} icon={Layers} accentColor="slate" />
        <MetricCard title="Categorical" value={qualityAudit?.categoricalFeaturesCount} icon={Layers} accentColor="slate" />
        <MetricCard title="Numerical" value={qualityAudit?.numericalFeaturesCount} icon={Layers} accentColor="cyan" />
      </div>

      {/* Tab Navigation Header */}
      <div className="border-b border-soc-border flex items-center space-x-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-xs font-mono font-semibold transition-all border-b-2 ${
              activeTab === tab.id
                ? 'border-cyan-400 text-cyan-400 bg-cyan-950/30'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
            }`}
          >
            {tab.name}
          </button>
        ))}
      </div>

      {/* Tab Content Areas */}

      {/* 1. OVERVIEW TAB */}
      {activeTab === 'overview' && (
        <div className="space-y-4">
          {!qualityAudit ? (
            <EmptyState
              title="No dataset has been analyzed yet"
              description="Run Phase 1 EDA pipeline or upload a CSV dataset above to populate this data quality breakdown."
            />
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ChartCard title="Class Distribution Overview" subtitle="Benign vs Attack flows count">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={classDist || []} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                    <XAxis dataKey="className" stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '11px', fontFamily: 'monospace' }} />
                    <Bar dataKey="sampleCount" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Feature Type Breakdown" subtitle="Numerical vs Categorical vs Constant attributes">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Numerical', value: qualityAudit.numericalFeaturesCount || 0 },
                        { name: 'Categorical', value: qualityAudit.categoricalFeaturesCount || 0 },
                        { name: 'Constant (Zero Var)', value: qualityAudit.constantFeaturesCount || 0 },
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={75}
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {COLORS.map((color, index) => (
                        <Cell key={`cell-${index}`} fill={color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '11px', fontFamily: 'monospace' }} />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
          )}
        </div>
      )}

      {/* 2. DATA QUALITY TAB */}
      {activeTab === 'quality' && (
        <div className="space-y-4">
          <div className="soc-card border border-soc-border p-4 space-y-3">
            <h3 className="text-xs font-mono font-bold uppercase text-slate-200">
              Data Cleaning & Sanitation Checks
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
              <div className="p-3 bg-slate-900 rounded border border-slate-800">
                <div className="text-slate-400">Missing Values Audit</div>
                <div className="text-lg font-bold text-emerald-400 mt-1">
                  {qualityAudit?.missingValuesCount ?? '—'} <span className="text-xs text-slate-400 font-normal">rows</span>
                </div>
                <div className="text-[10px] text-slate-500 mt-1">
                  Imputed using median for numerical features during Phase 2.
                </div>
              </div>

              <div className="p-3 bg-slate-900 rounded border border-slate-800">
                <div className="text-slate-400">Infinite Values Check</div>
                <div className="text-lg font-bold text-rose-400 mt-1">
                  {qualityAudit?.infiniteValuesCount ?? '—'} <span className="text-xs text-slate-400 font-normal font-mono">Inf values</span>
                </div>
                <div className="text-[10px] text-slate-500 mt-1">
                  Replaced with max finite float boundaries to prevent loss overflow.
                </div>
              </div>

              <div className="p-3 bg-slate-900 rounded border border-slate-800">
                <div className="text-slate-400">Duplicate Flow Records</div>
                <div className="text-lg font-bold text-amber-400 mt-1">
                  {qualityAudit?.duplicateRowsCount ?? '—'} <span className="text-xs text-slate-400 font-normal font-mono">duplicates</span>
                </div>
                <div className="text-[10px] text-slate-500 mt-1">
                  Evaluated for train-test split isolation.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 3. FEATURES TAB */}
      {activeTab === 'features' && (
        <div className="soc-card border border-soc-border p-4 space-y-3">
          <h3 className="text-xs font-mono font-bold uppercase text-slate-200">
            Feature Selection & Variance Profiling
          </h3>
          <p className="text-xs font-mono text-slate-400">
            Features with zero variance (constant features) are automatically dropped during Phase 2 feature engineering to prevent model instability.
          </p>
          <div className="p-3 bg-slate-900 rounded border border-slate-800 text-xs font-mono text-slate-300">
            <span className="text-cyan-400 font-bold">Identified Constant Features (8 Dropped):</span> Bwd PSH Flags, Bwd URG Flags, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate.
          </div>
        </div>
      )}

      {/* 4. CLASS DISTRIBUTION TAB */}
      {activeTab === 'distribution' && (
        <div className="space-y-4">
          <ChartCard title="Detailed Class & Imbalance Breakdown" height="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classDist || []} margin={{ top: 10, right: 10, left: -10, bottom: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                <XAxis dataKey="className" stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '11px', fontFamily: 'monospace' }} />
                <Bar dataKey="sampleCount" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      {/* 5. CORRELATION TAB */}
      {activeTab === 'correlation' && (
        <div className="soc-card border border-soc-border p-4 space-y-3">
          <h3 className="text-xs font-mono font-bold uppercase text-slate-200">
            Pearson Correlation Matrix (Network Flow Attributes)
          </h3>
          <p className="text-xs font-mono text-slate-400">
            Multicollinearity inspection for feature scaling and dimensionality reduction.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-center text-xs font-mono border-collapse">
              <thead>
                <tr>
                  <th className="p-2 border border-slate-800 bg-slate-950"></th>
                  {correlationFeatures.map((f) => (
                    <th key={f} className="p-2 border border-slate-800 bg-slate-950 text-cyan-400 text-[10px]">
                      {f}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlationMatrix.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <th className="p-2 border border-slate-800 bg-slate-950 text-left text-slate-300 text-[10px]">
                      {correlationFeatures[rIdx]}
                    </th>
                    {row.map((val, cIdx) => (
                      <td key={cIdx} className={`p-2 border text-xs ${getHeatmapColor(val)}`}>
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

import React, { useState, useEffect } from 'react';
import {
  Database,
  ShieldAlert,
  Cpu,
  BarChart2,
  TrendingUp,
  Activity,
  Layers,
  FileCode,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';
import { MetricCard } from '../components/common/MetricCard';
import { ChartCard } from '../components/common/ChartCard';
import { PipelineVisualizer } from '../components/dashboard/PipelineVisualizer';
import { ResearchNotice } from '../components/common/ResearchNotice';
import { datasetApi } from '../api/datasetApi';
import { detectionApi } from '../api/detectionApi';
import { evaluationApi } from '../api/evaluationApi';
import { DatasetOverview } from '../types/dataset';
import { DetectionOverview } from '../types/detection';
import { ModelOverview } from '../types/evaluation';
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
  Legend,
  AreaChart,
  Area
} from 'recharts';

export const DashboardPage: React.FC = () => {
  const [datasetStats, setDatasetStats] = useState<DatasetOverview | null>(null);
  const [detectionStats, setDetectionStats] = useState<DetectionOverview | null>(null);
  const [modelStats, setModelStats] = useState<ModelOverview | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [ds, det, mod] = await Promise.all([
          datasetApi.getOverview(),
          detectionApi.getOverview(),
          evaluationApi.getModelOverview()
        ]);
        setDatasetStats(ds);
        setDetectionStats(det);
        setModelStats(mod);
      } catch (err) {
        console.warn("Failed loading dashboard overview:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();

    const handleDemoChange = () => fetchData();
    window.addEventListener('demo-mode-change', handleDemoChange);
    return () => window.removeEventListener('demo-mode-change', handleDemoChange);
  }, []);

  // Class distribution chart data
  const classDistData = detectionStats?.classCounts
    ? Object.entries(detectionStats.classCounts).map(([name, count]) => ({ name, count }))
    : [
        { name: 'BENIGN', count: 11800 },
        { name: 'DDoS', count: 1850 },
        { name: 'PortScan', count: 520 },
        { name: 'BruteForce', count: 210 },
        { name: 'WebAttack', count: 120 },
      ];

  const modelComparisonData = [
    { name: 'XGBoost', accuracy: 0.998, macroF1: 0.989 },
    { name: 'RandomForest', accuracy: 0.997, macroF1: 0.984 },
    { name: 'LogisticReg', accuracy: 0.942, macroF1: 0.812 },
    { name: 'SVM (Linear)', accuracy: 0.925, macroF1: 0.784 },
  ];

  const severityTimeData = [
    { time: '08:00', benign: 1200, critical: 45, high: 90 },
    { time: '09:00', benign: 1850, critical: 12, high: 40 },
    { time: '10:00', benign: 2400, critical: 180, high: 210 },
    { time: '11:00', benign: 2100, critical: 80, high: 110 },
    { time: '12:00', benign: 2900, critical: 25, high: 60 },
  ];

  const COLORS = ['#10b981', '#f43f5e', '#f59e0b', '#3b82f6', '#a855f7'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-mono font-bold text-slate-100 uppercase tracking-wide flex items-center space-x-2">
          <Activity className="w-5 h-5 text-cyan-400" />
          <span>AI-Based Cyber Threat Detection</span>
        </h1>
        <p className="text-xs font-mono text-slate-400 mt-1">
          AI-driven network threat detection and cyber threat intelligence platform
        </p>
      </div>

      <ResearchNotice />

      {/* 1. DATASET OVERVIEW */}
      <section className="space-y-2">
        <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400 flex items-center space-x-2">
          <Database className="w-4 h-4" />
          <span>Dataset Overview (Phase 1)</span>
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <MetricCard
            title="Dataset Name"
            value={datasetStats?.datasetName}
            icon={Database}
            accentColor="cyan"
            subtitle="Selected Benchmark Dataset"
          />
          <MetricCard
            title="Total Records"
            value={datasetStats?.totalRecords}
            icon={FileCode}
            accentColor="cyan"
            subtitle="Ingested Samples"
          />
          <MetricCard
            title="Total Features"
            value={datasetStats?.totalFeatures}
            icon={Layers}
            accentColor="cyan"
            subtitle="Extracted Flow Metrics"
          />
          <MetricCard
            title="Classes"
            value={datasetStats?.totalClasses}
            icon={BarChart2}
            accentColor="cyan"
            subtitle="Attack Categories"
          />
          <MetricCard
            title="Data Quality"
            value={datasetStats?.dataQualityScore}
            unit="%"
            icon={CheckCircle2}
            accentColor="emerald"
            subtitle="Audit Cleanliness"
          />
        </div>
      </section>

      {/* 2. DETECTION OVERVIEW */}
      <section className="space-y-2">
        <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-rose-400 flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4" />
          <span>Detection Overview (Phase 3 Baseline)</span>
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <MetricCard
            title="Total Events"
            value={detectionStats?.totalEvents}
            icon={Activity}
            accentColor="slate"
            subtitle="Processed Traffic Flows"
          />
          <MetricCard
            title="Benign Events"
            value={detectionStats?.benignEvents}
            icon={CheckCircle2}
            accentColor="emerald"
            subtitle="Normal Traffic Signals"
          />
          <MetricCard
            title="Malicious Events"
            value={detectionStats?.maliciousEvents}
            icon={AlertTriangle}
            accentColor="rose"
            subtitle="Detected Intrusion Vector"
          />
          <MetricCard
            title="Attack Classes"
            value={detectionStats?.attackClassesCount}
            icon={ShieldAlert}
            accentColor="amber"
            subtitle="Distinct Threat Categories"
          />
        </div>
      </section>

      {/* 3. MODEL OVERVIEW */}
      <section className="space-y-2">
        <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-emerald-400 flex items-center space-x-2">
          <Cpu className="w-4 h-4" />
          <span>Model Overview (Phase 3 Benchmark)</span>
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
          <MetricCard
            title="Best Model"
            value={modelStats?.bestModelName}
            icon={Cpu}
            accentColor="emerald"
            badge="Phase 3"
          />
          <MetricCard
            title="Accuracy"
            value={modelStats?.accuracy}
            icon={TrendingUp}
            accentColor="emerald"
          />
          <MetricCard
            title="Macro F1"
            value={modelStats?.macroF1}
            icon={BarChart2}
            accentColor="emerald"
          />
          <MetricCard
            title="MCC Score"
            value={modelStats?.mcc}
            icon={Cpu}
            accentColor="emerald"
          />
          <MetricCard
            title="Precision"
            value={modelStats?.precision}
            icon={CheckCircle2}
            accentColor="emerald"
          />
          <MetricCard
            title="Recall"
            value={modelStats?.recall}
            icon={ShieldAlert}
            accentColor="emerald"
          />
        </div>
      </section>

      {/* System Pipeline Visualization */}
      <PipelineVisualizer />

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Attack Class Distribution Chart */}
        <ChartCard
          title="Attack Class Distribution"
          subtitle="Frequency breakdown across benign and attack flow classes"
          icon={BarChart2}
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={classDistData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
              <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '11px', fontFamily: 'monospace' }}
              />
              <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Model Performance Comparison Chart */}
        <ChartCard
          title="Baseline Model Comparison"
          subtitle="Accuracy vs Macro F1 score across baseline classifiers"
          icon={Cpu}
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={modelComparisonData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
              <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
              <YAxis domain={[0.7, 1.0]} stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '11px', fontFamily: 'monospace' }}
              />
              <Bar dataKey="accuracy" name="Accuracy" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="macroF1" name="Macro F1" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Recent Detection Distribution Area Chart */}
        <ChartCard
          title="Recent Detection Distribution"
          subtitle="Timeline flow of benign vs critical threat events"
          icon={Activity}
        >
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={severityTimeData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
              <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '11px', fontFamily: 'monospace' }}
              />
              <Area type="monotone" dataKey="benign" stackId="1" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
              <Area type="monotone" dataKey="high" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.5} />
              <Area type="monotone" dataKey="critical" stackId="1" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.8} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
};

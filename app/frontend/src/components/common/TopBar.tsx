import React, { useState, useEffect } from 'react';
import { Activity, Database, AlertCircle, RefreshCw, CheckCircle2, Server, FlaskConical } from 'lucide-react';
import { apiClient } from '../../api/client';
import { isDemoModeActive, setDemoMode } from '../../mocks/mockAdapter';

interface TopBarProps {
  sidebarCollapsed: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({ sidebarCollapsed }) => {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState<boolean>(false);
  const [demoActive, setDemoActive] = useState<boolean>(isDemoModeActive());
  const [selectedDataset, setSelectedDataset] = useState<string>('CIC-IDS2017');

  const checkHealth = async () => {
    setIsChecking(true);
    try {
      await apiClient.get('/health', { timeout: 3000 });
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const handleDemoChange = () => setDemoActive(isDemoModeActive());
    window.addEventListener('demo-mode-change', handleDemoChange);
    return () => window.removeEventListener('demo-mode-change', handleDemoChange);
  }, []);

  const handleToggleDemo = () => {
    const next = !demoActive;
    setDemoMode(next);
    setDemoActive(next);
  };

  return (
    <header
      className={`fixed top-0 right-0 z-30 h-16 bg-[#0e1424]/90 backdrop-blur-md border-b border-soc-border transition-all duration-300 flex items-center justify-between px-6 ${
        sidebarCollapsed ? 'left-16' : 'left-64'
      }`}
    >
      {/* Target Dataset & Status Indicator */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 bg-slate-900/80 px-3 py-1.5 rounded-md border border-soc-border">
          <Database className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-slate-400 font-mono">Dataset:</span>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="bg-transparent text-xs font-mono font-medium text-slate-200 focus:outline-none cursor-pointer"
          >
            <option value="CIC-IDS2017" className="bg-slate-900 text-slate-200">UNB CIC-IDS2017</option>
            <option value="UNSW-NB15" className="bg-slate-900 text-slate-200">UNSW Canberra NB15</option>
          </select>
        </div>

        <div className="hidden md:flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-md bg-slate-900/60 border border-soc-border">
          <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
          <span className="text-slate-400">Max Backend Phase:</span>
          <span className="text-emerald-400 font-bold">PHASE 3 (Baselines)</span>
        </div>
      </div>

      {/* Right Controls: Backend Status & Demo Mode */}
      <div className="flex items-center space-x-3">
        {/* Demo Mode Toggle */}
        <button
          onClick={handleToggleDemo}
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-mono font-semibold transition-all border ${
            demoActive
              ? 'bg-amber-950/80 text-amber-300 border-amber-600/70 shadow-soc-glow-rose'
              : 'bg-slate-900 text-slate-400 border-slate-700 hover:text-slate-200'
          }`}
          title="Toggle Demo Data mode for UI preview"
        >
          <FlaskConical className="w-3.5 h-3.5" />
          <span>{demoActive ? 'DEMO DATA ACTIVE' : 'PROD API MODE'}</span>
        </button>

        {/* Backend API Health Status */}
        <div
          onClick={checkHealth}
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-mono cursor-pointer border transition-colors ${
            backendOnline === true
              ? 'bg-emerald-950/40 text-emerald-400 border-emerald-800/60'
              : backendOnline === false
              ? 'bg-slate-900/80 text-slate-400 border-slate-800'
              : 'bg-slate-900 text-slate-400 border-slate-800'
          }`}
          title="Click to re-check FastAPI backend health"
        >
          <Server className="w-3.5 h-3.5" />
          <span>API:</span>
          {isChecking ? (
            <RefreshCw className="w-3 h-3 animate-spin text-cyan-400" />
          ) : backendOnline === true ? (
            <span className="flex items-center text-emerald-400 font-medium">
              <CheckCircle2 className="w-3 h-3 mr-1" /> ONLINE
            </span>
          ) : (
            <span className="flex items-center text-slate-400">
              <AlertCircle className="w-3 h-3 mr-1" /> OFFLINE
            </span>
          )}
        </div>
      </div>
    </header>
  );
};

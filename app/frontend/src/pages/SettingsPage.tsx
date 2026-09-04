import React, { useState } from 'react';
import { Settings, Server, RefreshCw, CheckCircle2, AlertCircle, FlaskConical, Database } from 'lucide-react';
import { updateApiBaseUrl } from '../api/client';
import { isDemoModeActive, setDemoMode } from '../mocks/mockAdapter';

export const SettingsPage: React.FC = () => {
  const [baseUrl, setBaseUrl] = useState<string>(
    localStorage.getItem('CYBERTHREAT_API_BASE_URL') || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
  );
  const [demoActive, setDemoActive] = useState<boolean>(isDemoModeActive());
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);

  const handleSaveSettings = () => {
    updateApiBaseUrl(baseUrl);
    setDemoMode(demoActive);
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-xl font-mono font-bold text-slate-100 uppercase tracking-wide flex items-center space-x-2">
          <Settings className="w-5 h-5 text-cyan-400" />
          <span>System Settings & API Configuration</span>
        </h1>
        <p className="text-xs font-mono text-slate-400 mt-1">
          Configure backend FastAPI endpoints, demo data mode, and environment variables
        </p>
      </div>

      {/* API Configuration Card */}
      <div className="soc-card border border-soc-border p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-soc-border/60 pb-3">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
            <Server className="w-4 h-4 text-cyan-400" />
            <span>FastAPI Backend Connection Settings</span>
          </h3>
          <span className="text-[10px] font-mono text-slate-400">
            Environment: <span className="text-slate-200 font-semibold">{import.meta.env.MODE}</span>
          </span>
        </div>

        <div className="space-y-3">
          <div>
            <label className="block text-xs font-mono text-slate-400 mb-1.5 uppercase font-semibold">
              API Base URL (VITE_API_BASE_URL)
            </label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://localhost:8000/api"
              className="w-full bg-slate-900 border border-soc-border rounded-md px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500/60"
            />
            <p className="text-[10px] font-mono text-slate-500 mt-1">
              Base URL of the running FastAPI server serving Phase 1-3 endpoints.
            </p>
          </div>
        </div>
      </div>

      {/* Demo Mode Card */}
      <div className="soc-card border border-soc-border p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-soc-border/60 pb-3">
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
            <FlaskConical className="w-4 h-4 text-amber-400" />
            <span>Demo Data Mode (UI Presentation Sandbox)</span>
          </h3>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs font-mono font-semibold text-slate-200">
              Enable "DEMO DATA" Fallback Adapter
            </div>
            <p className="text-[11px] font-mono text-slate-400 mt-0.5 max-w-xl">
              When enabled, the frontend renders benchmark dataset statistics and baseline evaluation metrics explicitly tagged as "DEMO DATA". When disabled, the frontend operates strictly in production API mode.
            </p>
          </div>

          <button
            onClick={() => setDemoActive(!demoActive)}
            className={`px-4 py-2 rounded-md text-xs font-mono font-bold transition-all border ${
              demoActive
                ? 'bg-amber-950/80 text-amber-300 border-amber-600/70'
                : 'bg-slate-900 text-slate-400 border-slate-700'
            }`}
          >
            {demoActive ? 'DEMO MODE: ACTIVE' : 'DEMO MODE: DISABLED'}
          </button>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex items-center space-x-3">
        <button
          onClick={handleSaveSettings}
          className="px-5 py-2.5 rounded-md text-xs font-mono font-bold uppercase bg-cyan-950/90 text-cyan-300 border border-cyan-800/80 hover:bg-cyan-900 transition-colors flex items-center space-x-2"
        >
          <Settings className="w-4 h-4" />
          <span>Save Settings</span>
        </button>

        {saveSuccess && (
          <span className="text-xs font-mono text-emerald-400 flex items-center space-x-1">
            <CheckCircle2 className="w-4 h-4" />
            <span>Settings saved successfully!</span>
          </span>
        )}
      </div>
    </div>
  );
};

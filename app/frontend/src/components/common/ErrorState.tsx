import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = "Backend API Unreachable",
  message = "Could not fetch research data from FastAPI backend server at VITE_API_BASE_URL.",
  onRetry
}) => {
  return (
    <div className="soc-card border-rose-900/60 bg-rose-950/20 p-6 text-center flex flex-col items-center justify-center space-y-3 my-4">
      <div className="p-2 bg-rose-900/40 text-rose-400 border border-rose-800 rounded-full">
        <AlertTriangle className="w-5 h-5" />
      </div>
      <div>
        <h4 className="text-xs font-mono font-bold text-rose-300 uppercase tracking-wide">
          {title}
        </h4>
        <p className="text-xs font-mono text-slate-400 mt-1 max-w-md">
          {message}
        </p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-mono bg-slate-900 text-slate-300 border border-slate-700 hover:bg-slate-800 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Request</span>
        </button>
      )}
    </div>
  );
};

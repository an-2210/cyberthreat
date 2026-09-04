import React from 'react';
import { RefreshCw } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ message = "Loading research metrics..." }) => {
  return (
    <div className="soc-card p-12 text-center flex flex-col items-center justify-center space-y-3">
      <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin" />
      <p className="text-xs font-mono text-slate-400 animate-pulse">{message}</p>
    </div>
  );
};

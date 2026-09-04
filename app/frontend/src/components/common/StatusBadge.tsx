import React from 'react';
import { SeverityLevel } from '../../types/detection';

interface StatusBadgeProps {
  status: SeverityLevel | 'COMPLETED' | 'RUNNING' | 'FAILED' | 'PLANNED' | 'NOT IMPLEMENTED' | 'IMPLEMENTED' | string;
  type?: 'severity' | 'phase' | 'status';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, type = 'severity' }) => {
  const normalized = status ? status.toUpperCase() : 'UNKNOWN';

  let badgeClasses = 'bg-slate-800 text-slate-400 border-slate-700';

  if (type === 'severity') {
    switch (normalized) {
      case 'CRITICAL':
        badgeClasses = 'bg-rose-950/80 text-rose-300 border-rose-800/80 shadow-soc-glow-rose';
        break;
      case 'HIGH':
        badgeClasses = 'bg-rose-950/40 text-rose-400 border-rose-900/60';
        break;
      case 'MEDIUM':
        badgeClasses = 'bg-amber-950/50 text-amber-300 border-amber-800/60';
        break;
      case 'LOW':
        badgeClasses = 'bg-blue-950/50 text-blue-300 border-blue-800/60';
        break;
      case 'BENIGN':
        badgeClasses = 'bg-emerald-950/50 text-emerald-300 border-emerald-800/60';
        break;
      default:
        badgeClasses = 'bg-slate-900 text-slate-400 border-slate-800';
    }
  } else if (type === 'phase' || type === 'status') {
    switch (normalized) {
      case 'COMPLETED':
      case 'IMPLEMENTED':
      case 'READY':
        badgeClasses = 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60';
        break;
      case 'RUNNING':
      case 'TRAINING':
        badgeClasses = 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60 animate-pulse';
        break;
      case 'PLANNED':
        badgeClasses = 'bg-blue-950/60 text-blue-300 border-blue-800/60';
        break;
      case 'NOT IMPLEMENTED':
      case 'DISABLED':
        badgeClasses = 'bg-slate-900 text-slate-400 border-slate-800';
        break;
      case 'FAILED':
        badgeClasses = 'bg-rose-950/80 text-rose-400 border-rose-800/80';
        break;
      default:
        badgeClasses = 'bg-slate-900 text-slate-400 border-slate-800';
    }
  }

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${badgeClasses}`}>
      {normalized}
    </span>
  );
};

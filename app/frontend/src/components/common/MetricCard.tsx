import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number | null | undefined;
  subtitle?: string;
  icon?: LucideIcon;
  accentColor?: 'cyan' | 'emerald' | 'amber' | 'rose' | 'purple' | 'slate';
  unit?: string;
  badge?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  accentColor = 'cyan',
  unit,
  badge
}) => {
  const isValueAvailable = value !== null && value !== undefined && value !== '';

  const colorStyles = {
    cyan: 'border-cyan-900/40 text-cyan-400 bg-cyan-950/20',
    emerald: 'border-emerald-900/40 text-emerald-400 bg-emerald-950/20',
    amber: 'border-amber-900/40 text-amber-400 bg-amber-950/20',
    rose: 'border-rose-900/40 text-rose-400 bg-rose-950/20',
    purple: 'border-purple-900/40 text-purple-400 bg-purple-950/20',
    slate: 'border-slate-800 text-slate-400 bg-slate-900/40',
  };

  const iconStyles = {
    cyan: 'text-cyan-400 bg-cyan-950/60 border-cyan-800/40',
    emerald: 'text-emerald-400 bg-emerald-950/60 border-emerald-800/40',
    amber: 'text-amber-400 bg-amber-950/60 border-amber-800/40',
    rose: 'text-rose-400 bg-rose-950/60 border-rose-800/40',
    purple: 'text-purple-400 bg-purple-950/60 border-purple-800/40',
    slate: 'text-slate-400 bg-slate-900 border-slate-800',
  };

  return (
    <div className={`soc-card soc-card-hover border ${colorStyles[accentColor]} flex flex-col justify-between`}>
      <div className="flex items-start justify-between">
        <div>
          <span className="text-[11px] font-mono font-semibold tracking-wider text-slate-400 uppercase">
            {title}
          </span>
          {badge && (
            <span className="ml-2 px-1.5 py-0.5 text-[9px] font-mono bg-slate-800 text-slate-300 rounded border border-slate-700">
              {badge}
            </span>
          )}
        </div>
        {Icon && (
          <div className={`p-2 rounded-md border ${iconStyles[accentColor]}`}>
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="mt-3 mb-1">
        {isValueAvailable ? (
          <div className="flex items-baseline space-x-1">
            <span className="text-2xl font-mono font-bold text-slate-100 tracking-tight">
              {typeof value === 'number' && value < 1 && value > 0
                ? (value * 100).toFixed(2) + '%'
                : typeof value === 'number'
                ? value.toLocaleString()
                : value}
            </span>
            {unit && <span className="text-xs font-mono text-slate-400">{unit}</span>}
          </div>
        ) : (
          <div className="text-sm font-mono text-slate-500 italic">
            Not evaluated
          </div>
        )}
      </div>

      {subtitle && (
        <p className="text-[11px] font-mono text-slate-400 border-t border-slate-800/60 pt-2 mt-2">
          {subtitle}
        </p>
      )}
    </div>
  );
};

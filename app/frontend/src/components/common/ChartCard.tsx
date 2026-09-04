import React from 'react';
import { LucideIcon, BarChart2 } from 'lucide-react';

interface ChartCardProps {
  title: string;
  subtitle?: string;
  icon?: LucideIcon;
  children: React.ReactNode;
  action?: React.ReactNode;
  height?: string;
}

export const ChartCard: React.FC<ChartCardProps> = ({
  title,
  subtitle,
  icon: Icon = BarChart2,
  children,
  action,
  height = 'h-64'
}) => {
  return (
    <div className="soc-card border border-soc-border flex flex-col justify-between">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-soc-border/60">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-cyan-950/60 text-cyan-400 border border-cyan-800/40 rounded">
            <Icon className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              {title}
            </h3>
            {subtitle && (
              <p className="text-[11px] font-mono text-slate-400">
                {subtitle}
              </p>
            )}
          </div>
        </div>
        {action && <div>{action}</div>}
      </div>

      <div className={`w-full ${height} flex items-center justify-center`}>
        {children}
      </div>
    </div>
  );
};

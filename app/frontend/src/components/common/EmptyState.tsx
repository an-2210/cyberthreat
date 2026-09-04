import React from 'react';
import { Database, ArrowRight, ShieldAlert } from 'lucide-react';

interface EmptyStateProps {
  title: string;
  description: string;
  actionText?: string;
  onAction?: () => void;
  icon?: React.ElementType;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  actionText,
  onAction,
  icon: Icon = Database
}) => {
  return (
    <div className="soc-card border-dashed border-slate-800 p-8 text-center flex flex-col items-center justify-center space-y-3 bg-slate-950/40 my-4">
      <div className="p-3 bg-slate-900 text-slate-400 border border-slate-800 rounded-full">
        <Icon className="w-6 h-6 text-slate-400" />
      </div>
      <div className="max-w-md">
        <h4 className="text-sm font-mono font-bold text-slate-200 uppercase tracking-wide">
          {title}
        </h4>
        <p className="text-xs font-mono text-slate-400 mt-1 leading-relaxed">
          {description}
        </p>
      </div>
      {actionText && onAction && (
        <button
          onClick={onAction}
          className="mt-2 inline-flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-mono font-medium bg-cyan-950/80 text-cyan-300 border border-cyan-800/80 hover:bg-cyan-900 transition-colors"
        >
          <span>{actionText}</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
};

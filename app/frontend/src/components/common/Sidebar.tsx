import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  BarChart3,
  ShieldAlert,
  LineChart,
  FlaskConical,
  BrainCircuit,
  Crosshair,
  BookOpen,
  FileSpreadsheet,
  Settings,
  ShieldCheck,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggleCollapse }) => {
  const activeNavItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard, badge: null },
    { name: 'Data Analysis', path: '/data-analysis', icon: BarChart3, badge: 'Phase 1' },
    { name: 'Threat Detection', path: '/threat-detection', icon: ShieldAlert, badge: 'Phase 3' },
    { name: 'Model Performance', path: '/model-performance', icon: LineChart, badge: 'Phase 3' },
    { name: 'Experiments', path: '/experiments', icon: FlaskConical, badge: 'Tracking' },
  ];

  const futureNavItems = [
    { name: 'Explainability', path: '/explainability', icon: BrainCircuit, phase: 'Phase 8' },
    { name: 'MITRE ATT&CK', path: '/mitre-attack', icon: Crosshair, phase: 'Phase 9' },
    { name: 'Threat Intelligence', path: '/threat-intelligence', icon: BookOpen, phase: 'Phase 11' },
    { name: 'CTI Reports', path: '/cti-reports', icon: FileSpreadsheet, phase: 'Phase 12' },
  ];

  return (
    <aside
      className={`fixed top-0 left-0 bottom-0 z-40 bg-[#0d1322] border-r border-soc-border transition-all duration-300 flex flex-col ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div className="h-16 px-4 border-b border-soc-border flex items-center justify-between">
        <div className="flex items-center space-x-3 overflow-hidden">
          <div className="p-2 bg-cyan-950/60 text-cyan-400 border border-cyan-800/50 rounded-md shrink-0">
            <ShieldCheck className="w-5 h-5 text-cyan-400" />
          </div>
          {!collapsed && (
            <div className="truncate">
              <h1 className="font-mono text-sm font-bold tracking-wider text-slate-100 uppercase">
                CYBER<span className="text-cyan-400">THREAT</span>
              </h1>
              <p className="text-[10px] text-slate-400 tracking-tight font-mono">
                RESEARCH SOC v1.0
              </p>
            </div>
          )}
        </div>
        <button
          onClick={onToggleCollapse}
          className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition-colors"
          title={collapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Nav Content */}
      <div className="flex-1 overflow-y-auto py-4 px-2 space-y-6">
        {/* Implemented Active Modules */}
        <div>
          {!collapsed && (
            <div className="px-3 mb-2 text-[10px] font-mono font-semibold text-slate-400 tracking-wider uppercase">
              Core Modules (Phase 1–3)
            </div>
          )}
          <nav className="space-y-1">
            {activeNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center justify-between px-3 py-2.5 rounded-md text-sm font-medium transition-all group ${
                    isActive
                      ? 'bg-cyan-950/50 text-cyan-400 border border-cyan-800/60 shadow-soc-glow-cyan'
                      : 'text-slate-300 hover:bg-slate-800/50 hover:text-slate-100'
                  }`
                }
              >
                <div className="flex items-center space-x-3 truncate">
                  <item.icon className="w-4 h-4 text-slate-400 group-hover:text-cyan-400 transition-colors shrink-0" />
                  {!collapsed && <span className="truncate">{item.name}</span>}
                </div>
                {!collapsed && item.badge && (
                  <span className="px-1.5 py-0.5 text-[9px] font-mono font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/50 rounded">
                    {item.badge}
                  </span>
                )}
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Separator */}
        <div className="border-t border-soc-border/60 mx-2" />

        {/* Future Coming Soon Modules */}
        <div>
          {!collapsed && (
            <div className="px-3 mb-2 text-[10px] font-mono font-semibold text-slate-400 tracking-wider uppercase flex items-center justify-between">
              <span>Roadmap Modules</span>
              <span className="text-[9px] text-amber-400/80 font-normal">Phase 4+</span>
            </div>
          )}
          <nav className="space-y-1">
            {futureNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center justify-between px-3 py-2.5 rounded-md text-sm font-medium transition-all group ${
                    isActive
                      ? 'bg-slate-800/80 text-slate-100 border border-slate-700'
                      : 'text-slate-400 hover:bg-slate-800/30 hover:text-slate-300'
                  }`
                }
              >
                <div className="flex items-center space-x-3 truncate">
                  <item.icon className="w-4 h-4 text-slate-500 group-hover:text-slate-400 transition-colors shrink-0" />
                  {!collapsed && <span className="truncate">{item.name}</span>}
                </div>
                {!collapsed && (
                  <span className="px-1.5 py-0.5 text-[9px] font-mono bg-slate-800 text-slate-400 border border-slate-700/60 rounded">
                    Coming Soon
                  </span>
                )}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>

      {/* Bottom Settings Link */}
      <div className="p-2 border-t border-soc-border bg-[#0a0e17]">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `flex items-center space-x-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all ${
              isActive
                ? 'bg-slate-800 text-cyan-400 border border-cyan-900/60'
                : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
            }`
          }
        >
          <Settings className="w-4 h-4 shrink-0" />
          {!collapsed && <span className="truncate">Settings & API</span>}
        </NavLink>
      </div>
    </aside>
  );
};

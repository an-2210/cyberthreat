import { useState } from 'react';
import { OverviewTab } from './components/OverviewTab';
import { DataTab } from './components/DataTab';
import { ModelsTab } from './components/ModelsTab';
import { TryItTab } from './components/TryItTab';

// One page, four tabs, in the order the project runs: overview -> data -> models -> try it.
const TABS = [
  { id: 'overview', label: 'Overview', Component: OverviewTab },
  { id: 'data', label: 'Data', Component: DataTab },
  { id: 'models', label: 'Models', Component: ModelsTab },
  { id: 'try', label: 'Try it', Component: TryItTab },
] as const;

export function App() {
  const [active, setActive] = useState<(typeof TABS)[number]['id']>('overview');
  const { Component } = TABS.find((t) => t.id === active)!;

  return (
    <div className="app">
      <div className="header">
        <h1>CyberThreat · network intrusion detection</h1>
        <span className="muted">CIC-IDS2017 · Random Forest and XGBoost</span>
      </div>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={t.id === active ? 'active' : ''} onClick={() => setActive(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      <Component />
    </div>
  );
}

import { useState } from 'react';
import { OverviewTab } from './components/OverviewTab';
import { DataTab } from './components/DataTab';
import { ModelsTab } from './components/ModelsTab';
import { ResearchTab } from './components/ResearchTab';
import { UnseenTab } from './components/UnseenTab';
import { UnswTab } from './components/UnswTab';
import { TryItTab } from './components/TryItTab';

// One page, seven tabs, in the order the project runs: overview -> data -> models -> research -> unknown attacks
// -> UNSW-NB15 -> try it.
const TABS = [
  { id: 'overview', label: 'Overview', Component: OverviewTab },
  { id: 'data', label: 'Data', Component: DataTab },
  { id: 'models', label: 'Models', Component: ModelsTab },
  { id: 'research', label: 'Research', Component: ResearchTab },
  { id: 'unseen', label: 'Unknown attacks', Component: UnseenTab },
  { id: 'unsw', label: 'UNSW-NB15', Component: UnswTab },
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

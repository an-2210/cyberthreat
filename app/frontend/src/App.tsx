import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/common/Sidebar';
import { TopBar } from './components/common/TopBar';
import { DashboardPage } from './pages/DashboardPage';
import { DataAnalysisPage } from './pages/DataAnalysisPage';
import { ThreatDetectionPage } from './pages/ThreatDetectionPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { ComingSoonPage } from './pages/ComingSoonPage';
import { SettingsPage } from './pages/SettingsPage';

export const App: React.FC = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);

  return (
    <Router>
      <div className="min-h-screen bg-soc-bg text-slate-100 flex">
        {/* Persistent Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />

        {/* Top Header Bar */}
        <TopBar sidebarCollapsed={sidebarCollapsed} />

        {/* Main Application Area */}
        <main
          className={`flex-1 transition-all duration-300 pt-20 px-6 pb-12 overflow-y-auto ${
            sidebarCollapsed ? 'ml-16' : 'ml-64'
          }`}
        >
          <Routes>
            {/* Active Implemented Phase 1-3 Pages */}
            <Route path="/" element={<DashboardPage />} />
            <Route path="/data-analysis" element={<DataAnalysisPage />} />
            <Route path="/threat-detection" element={<ThreatDetectionPage />} />
            <Route path="/model-performance" element={<ModelPerformancePage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />

            {/* Future Phase 4+ Coming Soon Modules */}
            <Route path="/explainability" element={<ComingSoonPage moduleKey="explainability" />} />
            <Route path="/mitre-attack" element={<ComingSoonPage moduleKey="mitre-attack" />} />
            <Route path="/threat-intelligence" element={<ComingSoonPage moduleKey="threat-intelligence" />} />
            <Route path="/cti-reports" element={<ComingSoonPage moduleKey="cti-reports" />} />

            {/* Settings Page */}
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;

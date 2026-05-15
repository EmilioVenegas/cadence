import React, { useState, useEffect, useMemo } from 'react';
import PatientSelector from './components/PatientSelector';
import LiveInferenceForm from './components/LiveInferenceForm';
import PatientSummaryDashboard from './components/PatientSummaryDashboard';
import InterventionChart from './components/InterventionChart';
import RankingPanel from './components/RankingPanel';
import { Activity, ShieldCheck, Heart } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [rankingData, setRankingData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Visibility control for interventions
  const [visibleInterventions, setVisibleInterventions] = useState(new Set([0, 1, 2]));
  const [hoveredIntervention, setHoveredIntervention] = useState(null);
  const [mode, setMode] = useState('registry'); // 'registry' or 'live'

  const fetchLiveInference = async (formData) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/rank/live`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      if (!res.ok) throw new Error("Custom patient already at optimal targets or invalid data.");
      const data = await res.json();
      setRankingData(data);
      setSelectedPatient({ isLive: true });
      setVisibleInterventions(new Set([0, 1, 2]));
    } catch (err) {
      setError(err.message);
      setRankingData(null);
    } finally {
      setLoading(false);
    }
  };

  const fetchRanking = async (p) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/rank/${p.cunicah}/${p.np}`);
      if (!res.ok) throw new Error("Patient already at optimal targets or not found.");
      const data = await res.json();
      setRankingData(data);
      setSelectedPatient(p);
      // Reset visibility to top 3
      setVisibleInterventions(new Set([0, 1, 2]));
    } catch (err) {
      setError(err.message);
      setRankingData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <header className="header">
        <div className="logo-group">
          <h1>🌋 Latent aging velocity atlas</h1>
          <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem', maxWidth: '500px' }}>
            Calculating cognitive & physical decline trajectories to simulate and propose optimal continuous-time interventions.
          </p>
        </div>
        <div className="status-group" style={{ display: 'flex', gap: '1rem' }}>
          <div className="glass-card" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} color="var(--accent-primary)" />
            <span style={{ fontSize: '0.8rem' }}>Server Status: Online</span>
          </div>
        </div>
      </header>

      <aside className="sidebar">
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
          <button
            onClick={() => setMode('registry')}
            style={{ flex: 1, padding: '0.6rem', background: mode === 'registry' ? 'var(--accent-primary)' : 'rgba(255,255,255,0.05)', color: mode === 'registry' ? '#fff' : 'var(--text-dim)', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
          >
            Registry
          </button>
          <button
            onClick={() => setMode('live')}
            style={{ flex: 1, padding: '0.6rem', background: mode === 'live' ? 'var(--accent-primary)' : 'rgba(255,255,255,0.05)', color: mode === 'live' ? '#fff' : 'var(--text-dim)', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
          >
            Live Inference
          </button>
        </div>

        {mode === 'registry' ? (
          <PatientSelector
            onSelect={fetchRanking}
            currentSelection={selectedPatient}
          />
        ) : (
          <div className="glass-card" style={{ padding: 0, flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <LiveInferenceForm onSubmit={fetchLiveInference} />
          </div>
        )}

        {selectedPatient && mode === 'registry' && (
          <div className="glass-card">
            <h3 style={{ marginBottom: '1rem' }}>Selected Profile</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem' }}>
              <div>Record ID: <span style={{ color: 'var(--accent-primary)', fontWeight: 'bold' }}>{selectedPatient.cunicah}-{selectedPatient.np}</span></div>
              <div>Wave (Ronda): {selectedPatient.ronda}</div>
              <div>Selection Age: {selectedPatient.edad?.toFixed(1)} y/o</div>
              <div>Interview Year: {selectedPatient.a_o_ent?.toFixed(0)}</div>
              <div>Gender: {selectedPatient.sexo === 1 ? 'Male' : 'Female'}</div>
            </div>
          </div>
        )}

        <div className="glass-card" style={{ marginTop: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: 0.7 }}>
            <ShieldCheck size={16} />
            <span style={{ fontSize: '0.8rem' }}>Mahalanobis Guardrails Active</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <div className="glass-card" style={{ height: '100%' }}>
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <div className="spinner" style={{ marginBottom: '1rem' }}>Analysing Digital Twin...</div>
              <p style={{ fontSize: '0.8rem', opacity: 0.7 }}>Running high-momentum Neural ODE simulation</p>
            </div>
          ) : error ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'var(--danger)' }}>
              {error}
            </div>
          ) : rankingData ? (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '1rem' }}>
              <div style={{ flex: 2, minHeight: 0 }}>
                <InterventionChart
                  data={rankingData}
                  visibleInterventions={visibleInterventions}
                  hoveredIntervention={hoveredIntervention}
                />
              </div>
              <div style={{ flex: 1, borderTop: '1px solid var(--border-glass)', paddingTop: '1rem' }}>
                <PatientSummaryDashboard ranking={rankingData} />
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%', opacity: 0.5 }}>
              <Heart size={48} style={{ marginBottom: '1rem' }} />
              <p>Select a latest patient record to initiate real-time inference</p>
            </div>
          )}
        </div>
      </main>

      <div className="glass-card" style={{ overflow: 'hidden', padding: 0 }}>
        <RankingPanel
          ranking={rankingData}
          visibleInterventions={visibleInterventions}
          setVisibleInterventions={setVisibleInterventions}
          hoveredIntervention={hoveredIntervention}
          setHoveredIntervention={setHoveredIntervention}
        />
      </div>

      <style>{`
        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid var(--border-glass);
          border-top: 4px solid var(--accent-primary);
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default App;

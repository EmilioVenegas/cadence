import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, ChevronsDown, Sparkles } from 'lucide-react';

const PatientSummaryDashboard = ({ ranking }) => {
    const [aiSummary, setAiSummary] = useState(null);
    const [loadingAI, setLoadingAI] = useState(false);

    useEffect(() => {
        setAiSummary(null);
        setLoadingAI(false);
    }, [ranking]);

    if (!ranking || !ranking.ranked_interventions || ranking.ranked_interventions.length === 0) return null;

    const baseAuc = ranking.auc_baseline;
    const bestIntervention = ranking.ranked_interventions[0];

    const generateSummary = async () => {
        setLoadingAI(true);
        try {
            const API_BASE = 'http://localhost:8000/api';
            const res = await fetch(`${API_BASE}/summary`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    actionable_deficits: ranking.actionable_deficits || [],
                    best_intervention: bestIntervention,
                    base_auc: baseAuc,
                    new_auc: bestIntervention.auc,
                    patient_context: ranking.patient_context || ""
                })
            });
            const data = await res.json();
            setAiSummary(data.action_plan_summary);
        } catch (e) {
            setAiSummary("Failed to generate AI summary.");
        } finally {
            setLoadingAI(false);
        }
    };

    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', height: '100%' }}>

            <div className="glass-card" style={{ background: 'rgba(255, 77, 77, 0.05)', padding: '1rem', border: '1px solid rgba(255, 77, 77, 0.2)' }}>
                <div style={{ color: 'var(--text-dim)', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <Activity size={16} /> BASELINE TRAJECTORY AUC
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#fff' }}>
                    {baseAuc.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.5rem' }}>
                    Cumulative 5-Year Predicted Decline
                </div>
            </div>

            <div className="glass-card" style={{ background: 'rgba(46, 204, 113, 0.05)', padding: '1rem', border: '1px solid rgba(46, 204, 113, 0.2)' }}>
                <div style={{ color: 'var(--text-dim)', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <ChevronsDown size={16} /> MAX IMPACT POTENTIAL
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--success)' }}>
                    -{bestIntervention.auc_reduction_pct.toFixed(1)}%
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.5rem' }}>
                    New AUC: {bestIntervention.auc.toFixed(2)}
                </div>
            </div>

            <div className="glass-card" style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '1rem' }}>
                <div style={{ color: 'var(--text-dim)', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <ShieldCheck size={16} color="var(--accent-secondary)" /> RECOMMENDED INTERVENTION
                </div>
                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: 'var(--accent-secondary)', wordWrap: 'break-word' }}>
                    {bestIntervention.label}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.5rem', display: 'flex', gap: '0.5rem' }}>
                    <span style={{ padding: '0.2rem 0.5rem', background: 'rgba(0,0,0,0.5)', borderRadius: '4px' }}>
                        Conf: {bestIntervention.confidence}
                    </span>
                </div>
            </div>

            <div className="glass-card" style={{ gridColumn: 'span 3', background: 'rgba(255, 255, 255, 0.02)', padding: '1rem', border: '1px solid var(--border-glass)' }}>
                <div style={{ color: 'var(--accent-primary)', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', fontWeight: 'bold' }}>
                    <Sparkles size={16} /> AI GENERATED ACTION PLAN
                </div>

                {!aiSummary && !loadingAI && (
                    <button
                        onClick={generateSummary}
                        style={{ padding: '0.6rem 1rem', background: 'var(--accent-primary)', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 'bold' }}>
                        Generate AI Insight Summary
                    </button>
                )}

                {loadingAI && (
                    <div style={{ fontSize: '0.9rem', color: 'var(--text-dim)', fontStyle: 'italic' }}>
                        Analyzing clinical trajectory using Gemma-4-31B...
                    </div>
                )}

                {aiSummary && (
                    <div style={{ fontSize: '0.9rem', color: 'var(--text-main)', lineHeight: '1.5' }}>
                        {aiSummary}
                    </div>
                )}
            </div>

        </div>
    );
};

export default PatientSummaryDashboard;

import React, { useState } from 'react';
import { Award, AlertCircle, Eye, EyeOff, ChevronDown } from 'lucide-react';

const RankingPanel = ({ ranking, visibleInterventions, setVisibleInterventions }) => {
    const [showDropdown, setShowDropdown] = useState(false);

    if (!ranking) return null;

    const toggleVisibility = (idx) => {
        const newSet = new Set(visibleInterventions);
        if (newSet.has(idx)) {
            newSet.delete(idx);
        } else {
            newSet.add(idx);
        }
        setVisibleInterventions(newSet);
    };

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <header style={{
                padding: '1rem',
                borderBottom: '1px solid var(--border-glass)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Award size={18} color="var(--accent-secondary)" /> Clinical Strategy Ranking
                </h4>

                {/* Dropdown Toggle for Visibility */}
                <div style={{ position: 'relative' }}>
                    <button
                        className="btn-glass"
                        onClick={() => setShowDropdown(!showDropdown)}
                        style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.8rem', padding: '0.3rem 0.6rem' }}
                    >
                        Manage Views <ChevronDown size={14} />
                    </button>

                    {showDropdown && (
                        <div style={{
                            position: 'absolute',
                            right: 0,
                            top: '120%',
                            background: 'var(--bg-dark)',
                            border: '1px solid var(--border-glass)',
                            borderRadius: '8px',
                            zIndex: 1000,
                            width: '200px',
                            padding: '0.5rem',
                            boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
                        }}>
                            <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginBottom: '0.5rem', padding: '0 0.5rem' }}>Toggle Chart visibility</p>
                            {ranking.ranked_interventions.map((r, i) => (
                                <div
                                    key={i}
                                    onClick={() => toggleVisibility(i)}
                                    style={{
                                        padding: '0.4rem 0.5rem',
                                        fontSize: '0.8rem',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.5rem',
                                        borderRadius: '4px',
                                        background: visibleInterventions.has(i) ? 'rgba(255,255,255,0.05)' : 'transparent'
                                    }}
                                >
                                    {visibleInterventions.has(i) ? <Eye size={14} /> : <EyeOff size={14} opacity={0.5} />}
                                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.label}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </header>

            <div style={{ overflowY: 'auto', flex: 1, position: 'relative' }}>
                <table style={{
                    width: '100%',
                    borderCollapse: 'separate',
                    borderSpacing: '0',
                    fontSize: '0.85rem',
                    tableLayout: 'fixed'
                }}>
                    <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-dark)', zIndex: 10 }}>
                        <tr>
                            <th style={{ ...thStyle, width: '40px' }}>#</th>
                            <th style={{ ...thStyle }}>Intervention</th>
                            <th style={{ ...thStyle, width: '90px' }}>Δ Velocity</th>
                            <th style={{ ...thStyle, width: '100px' }}>Confidence</th>
                            <th style={{ ...thStyle, width: '70px' }}>Maha</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ranking.ranked_interventions.map((r, i) => (
                            <tr
                                key={i}
                                className={visibleInterventions.has(i) ? 'row-active' : 'row-hidden'}
                                style={{
                                    background: i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
                                    height: '45px'
                                }}
                            >
                                <td style={tdStyle}>{i + 1}</td>
                                <td style={{ ...tdStyle, fontWeight: 'bold', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {r.label}
                                </td>
                                <td style={{ ...tdStyle, color: r.auc_reduction_pct > 0 ? 'var(--success)' : 'var(--danger)' }}>
                                    {r.auc_reduction_pct > 0 ? '↓' : '↑'} {Math.abs(r.auc_reduction_pct).toFixed(1)}%
                                </td>
                                <td style={tdStyle}>
                                    <span className={`badge ${r.confidence === 'High' ? 'badge-high' : 'badge-low'}`}>
                                        {r.confidence.split(' ')[0]}
                                    </span>
                                </td>
                                <td style={tdStyle}>
                                    {r.mahalanobis === Infinity ? 'OOD' : r.mahalanobis.toFixed(1)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div style={{ padding: '0.8rem 1rem', background: 'rgba(255, 77, 77, 0.08)', borderTop: '1px solid var(--border-glass)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <AlertCircle size={16} color="var(--accent-primary)" />
                    <strong style={{ fontSize: '0.8rem' }}>PRIMARY CLINICAL TARGET:</strong>
                    <span style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{ranking.ranked_interventions[0].label}</span>
                </div>
            </div>

            <style>{`
        .btn-glass {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid var(--border-glass);
          color: var(--text-main);
        }
        .btn-glass:hover {
          background: rgba(255, 255, 255, 0.1);
        }
        .row-hidden { opacity: 0.5; }
        .row-active { opacity: 1; }
      `}</style>
        </div>
    );
};

const thStyle = {
    textAlign: 'left',
    padding: '0.8rem 1rem',
    color: 'var(--text-dim)',
    fontSize: '0.65rem',
    textTransform: 'uppercase',
    letterSpacing: '1px',
    borderBottom: '1px solid var(--border-glass)'
};

const tdStyle = {
    padding: '0.5rem 1rem',
    verticalAlign: 'middle',
    borderBottom: '1px solid rgba(255,255,255,0.02)'
};

export default RankingPanel;

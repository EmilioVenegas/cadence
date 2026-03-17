import React, { useState, useEffect } from 'react';
import { Search, User } from 'lucide-react';

const PatientSelector = ({ onSelect, currentSelection }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const delayDebounceFn = setTimeout(() => {
            fetchPatients(searchTerm);
        }, 300);

        return () => clearTimeout(delayDebounceFn);
    }, [searchTerm]);

    const fetchPatients = async (q) => {
        setLoading(true);
        try {
            const res = await fetch(`http://localhost:8000/api/patients?q=${q}`);
            const data = await res.json();
            setResults(data);
        } catch (err) {
            console.error("Search failed", err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', height: '100%', maxHeight: '450px' }}>
            <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <User size={20} /> Patient Registry
            </h3>

            <div style={{ position: 'relative', marginBottom: '1rem' }}>
                <Search size={18} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', opacity: 0.5 }} />
                <input
                    type="text"
                    placeholder="Search by ID or Age..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={{ paddingLeft: '2.5rem' }}
                />
            </div>

            <div style={{ overflowY: 'auto', flex: 1 }}>
                {loading ? (
                    <p style={{ textAlign: 'center', opacity: 0.5, marginTop: '1rem' }}>Searching...</p>
                ) : results.length > 0 ? (
                    results.map(p => (
                        <div
                            key={`${p.cunicah}-${p.np}`}
                            className={`list-item ${currentSelection?.cunicah === p.cunicah && currentSelection?.np === p.np ? 'active' : ''}`}
                            onClick={() => onSelect(p)}
                        >
                            <div style={{ fontWeight: 'bold', fontSize: '1rem' }}>ID: {p.cunicah}-{p.np}</div>
                            <div style={{ fontSize: '0.75rem', opacity: 0.8, marginTop: '0.2rem' }}>
                                Wave (Ronda): <span style={{ color: 'var(--accent-secondary)' }}>{p.ronda}</span>
                            </div>
                            <div style={{ fontSize: '0.75rem', opacity: 0.6 }}>
                                Age: {p.edad?.toFixed(1)} | Year: {p.a_o_ent?.toFixed(0)}
                            </div>
                        </div>
                    ))
                ) : (
                    <p style={{ textAlign: 'center', opacity: 0.5, marginTop: '2rem' }}>No matches found</p>
                )}
            </div>
        </div>
    );
};

export default PatientSelector;

import React, { useState } from 'react';
import { Play } from 'lucide-react';

const CATEGORIES = {
    'Demographics': [
        { key: 'edad', label: 'Age (Years)', type: 'number', default: 65 },
        { key: 'sexo', label: 'Gender', type: 'select', options: [{ value: 1, label: 'Male' }, { value: 2, label: 'Female' }], default: 1 },
        { key: 'educacion', label: 'Education (Years)', type: 'number', default: 12 }
    ],
    'Lifestyle & Control': [
        { key: 'tabaco', label: 'Tobacco Use', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'alcohol', label: 'Heavy Alcohol Use', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'ejer_3_por_sem', label: 'Exercises ≥3x/week', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 1 },
        { key: 'asiste_club', label: 'Attends Social Club', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 1 },
        { key: 'voluntario', label: 'Engages in Volunteering', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 }
    ],
    'Comorbidities & Health': [
        { key: 'hipertension', label: 'Hypertension', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'diabetes', label: 'Diabetes', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'enf_pulm', label: 'Pulmonary Disease', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'artritis', label: 'Arthritis', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'infarto', label: 'Heart Attack', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'embolia', label: 'Stroke', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'cancer', label: 'Cancer', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'hospitalizacion', label: 'Recent Hospitalization', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'visita_medica', label: 'Recent Medical Visit', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'bmi_imp', label: 'BMI Impaired', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'salud_glob', label: 'Poor Global Health', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 }
    ],
    'ADL & Mobility': [
        { key: 'n_abvd', label: 'Difficulty Basic ADL', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'n_aivd', label: 'Difficulty Instrumental ADL', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'n_mov', label: 'Mobility Difficulty', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'n_img', label: 'Strength/Grip Impaired', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'motoras_gruesas', label: 'Gross Motor Impaired', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'motoras_finas', label: 'Fine Motor Impaired', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 }
    ],
    'Psychological & Mood': [
        { key: 'deprimido', label: 'Depressed', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'esfuerzo', label: 'Everything is an Effort', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'intranquilo', label: 'Restless/Anxious', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'triste', label: 'Sad', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'cansado', label: 'Tired', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'solo', label: 'Lonely', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'feliz', label: 'Unhappy (Not Happy)', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'disf_vida', label: 'Poor Quality of Life', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'energia', label: 'Low Energy', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 }
    ],
    'Cognitive Function': [
        { key: 'recuerdo1', label: 'Poor Recall 1', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'recuerdo2', label: 'Poor Recall 2', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'copiafiguras1', label: 'Poor Drawing 1', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'copiafiguras2', label: 'Poor Drawing 2', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'orientacion', label: 'Disoriented', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'serial7', label: 'Poor Serial 7s', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'visualscan', label: 'Poor Visual Scan', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
        { key: 'memoria', label: 'Poor Overall Memory', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 }
    ]
};

const LiveInferenceForm = ({ onSubmit }) => {
    // Generate initial state from defaults
    const initialState = {};
    Object.values(CATEGORIES).flat().forEach(field => {
        initialState[field.key] = field.default;
    });

    const [formData, setFormData] = useState(initialState);
    const [activeTab, setActiveTab] = useState('Demographics');

    const handleChange = (key, value, type) => {
        let parsed = value;
        if (type === 'number') {
            parsed = parseFloat(value);
            if (isNaN(parsed)) parsed = initialState[key];
        } else if (type === 'select') {
            parsed = parseInt(value, 10);
        }
        setFormData(prev => ({ ...prev, [key]: parsed }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit(formData);
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ display: 'flex', padding: '1rem', gap: '0.5rem', overflowX: 'auto', borderBottom: '1px solid var(--border-glass)' }}>
                {Object.keys(CATEGORIES).map(cat => (
                    <button
                        key={cat}
                        onClick={() => setActiveTab(cat)}
                        style={{
                            padding: '0.4rem 0.8rem',
                            border: 'none',
                            borderRadius: '4px',
                            background: activeTab === cat ? 'var(--accent-primary)' : 'rgba(255,255,255,0.05)',
                            color: activeTab === cat ? '#fff' : 'var(--text-dim)',
                            cursor: 'pointer',
                            fontSize: '0.8rem',
                            whiteSpace: 'nowrap'
                        }}
                    >
                        {cat}
                    </button>
                ))}
            </div>

            <div style={{ padding: '1rem', flex: 1, overflowY: 'auto' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
                    {CATEGORIES[activeTab].map(field => (
                        <div key={field.key} style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                            <label style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>{field.label}</label>
                            {field.type === 'select' ? (
                                <select
                                    value={formData[field.key]}
                                    onChange={(e) => handleChange(field.key, e.target.value, field.type)}
                                    style={inputStyle}
                                >
                                    {field.options.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                            ) : (
                                <input
                                    type="number"
                                    value={formData[field.key]}
                                    onChange={(e) => handleChange(field.key, e.target.value, field.type)}
                                    style={inputStyle}
                                />
                            )}
                        </div>
                    ))}
                </div>
            </div>

            <div style={{ padding: '1rem', borderTop: '1px solid var(--border-glass)' }}>
                <button
                    onClick={handleSubmit}
                    style={{
                        width: '100%',
                        padding: '0.8rem',
                        background: 'var(--accent-secondary)',
                        color: '#000',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontWeight: 'bold',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.5rem'
                    }}
                >
                    <Play size={18} />
                    Simulate Live Trajectory
                </button>
            </div>
        </div>
    );
};

const inputStyle = {
    padding: '0.5rem',
    background: 'rgba(0,0,0,0.2)',
    border: '1px solid var(--border-glass)',
    color: '#fff',
    borderRadius: '4px',
    fontSize: '0.9rem'
};

export default LiveInferenceForm;

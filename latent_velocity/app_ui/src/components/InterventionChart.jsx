import React from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const InterventionChart = ({ data, visibleInterventions, hoveredIntervention }) => {
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false // We use our own legend/toggle in RankingPanel
            },
            title: {
                display: true,
                text: `Predicted Aging Velocity Trajectory for Interventions - Patient ${data.patient_id}`,
                color: '#e0e0e0',
                font: { size: 16, weight: 'bold' }
            },
            tooltip: {
                backgroundColor: 'rgba(0,0,0,0.8)',
                titleColor: '#ff4d4d',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 10,
                callbacks: {
                    label: (context) => {
                        return ` ${context.dataset.label}: ${context.parsed.y.toFixed(3)}`;
                    }
                }
            }
        },
        scales: {
            y: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#a0a0a0' },
                title: {
                    display: true,
                    text: 'Velocity Magnitude (||v||)',
                    color: '#a0a0a0'
                }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#a0a0a0' },
                title: {
                    display: true,
                    text: 'Forecast Horizon (Years)',
                    color: '#a0a0a0'
                }
            }
        },
        animation: {
            duration: 500
        }
    };

    const palette = [
        '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c',
        '#e84393', '#fdcb6e', '#00cec9', '#d63031', '#636e72'
    ];

    const datasets = [
        {
            label: `Baseline`,
            data: data.v_mag_baseline,
            borderColor: hoveredIntervention ? '#ff4d4d33' : '#ff4d4d',
            borderWidth: 4,
            pointRadius: 0,
            tension: 0.3,
            z: 10
        },
        ...data.ranked_interventions.map((r, i) => {
            const isHovered = hoveredIntervention === r.label;
            const isDimmed = hoveredIntervention && !isHovered;
            const baseColor = palette[i % palette.length];
            const displayColor = isDimmed ? baseColor + '33' : baseColor;

            return {
                label: `${r.label}`,
                data: r.v_mag,
                borderColor: displayColor,
                borderWidth: isHovered ? 5 : 3,
                borderDash: r.confidence === "High" ? [] : [5, 5],
                pointRadius: 0,
                tension: 0.3,
                hidden: !visibleInterventions.has(i)
            }
        })
    ];

    const chartData = {
        labels: data.t.map(t => t.toFixed(1)),
        datasets
    };

    return (
        <div style={{ height: '100%', width: '100%' }}>
            <Line options={options} data={chartData} />
        </div>
    );
};

export default InterventionChart;

import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export default function Plot({ data, layout, style, config }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !data) return;

    // Apply dark-theme standards to Plotly figures
    const responsiveLayout = {
      ...layout,
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { 
        color: '#94a3b8', 
        family: "'Inter', system-ui, sans-serif" 
      },
      grid: {
        color: 'rgba(255,255,255,0.05)'
      },
      xaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.05)',
        zerolinecolor: 'rgba(255, 255, 255, 0.08)',
        linecolor: 'rgba(255, 255, 255, 0.08)',
        ...layout?.xaxis
      },
      yaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.05)',
        zerolinecolor: 'rgba(255, 255, 255, 0.08)',
        linecolor: 'rgba(255, 255, 255, 0.08)',
        ...layout?.yaxis
      },
      margin: { t: 40, r: 20, b: 40, l: 50, ...layout?.margin }
    };

    const defaultConfig = {
      responsive: true,
      displayModeBar: false,
      ...config
    };

    Plotly.newPlot(containerRef.current, data, responsiveLayout, defaultConfig);

    const handleResize = () => {
      if (containerRef.current) {
        Plotly.Plots.resize(containerRef.current);
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [data, layout, config]);

  return (
    <div 
      ref={containerRef} 
      style={{ 
        width: '100%', 
        height: '350px', 
        minHeight: '250px',
        ...style 
      }} 
    />
  );
}

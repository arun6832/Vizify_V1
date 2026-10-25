import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export default function Plot({ data, layout, style, config }) {
  const containerRef = useRef(null);
  const theme = document.documentElement.dataset.theme || 'dark';

  useEffect(() => {
    if (!containerRef.current || !data) return;

    const light = theme === 'light';
    const textColor = light ? '#526078' : '#a8b4c7';
    const gridColor = light ? 'rgba(82, 96, 120, 0.14)' : 'rgba(255,255,255,0.06)';
    const responsiveLayout = {
      ...layout,
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: textColor, family: "'DM Sans', system-ui, sans-serif" },
      grid: {
        color: gridColor
      },
      xaxis: {
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        linecolor: gridColor,
        tickfont: { color: textColor },
        title: { font: { color: textColor } },
        ...layout?.xaxis
      },
      yaxis: {
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        linecolor: gridColor,
        tickfont: { color: textColor },
        title: { font: { color: textColor } },
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
  }, [data, layout, config, theme]);

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

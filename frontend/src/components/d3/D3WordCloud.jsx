import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export default function D3WordCloud({ data, width = 600, height = 350, onWordClick }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const maxVal = d3.max(data, d => d.value) || 1;
    const minVal = d3.min(data, d => d.value) || 1;

    const fontScale = d3.scaleLinear()
      .domain([minVal, maxVal])
      .range([14, 42]);

    const colorScale = d3.scaleOrdinal()
      .range(['#8b5cf6', '#ec4899', '#3b82f6', '#10b981', '#f59e0b', '#06b6d4', '#a855f7']);

    // Layout simulation placing words in spiraling spiral cloud layout
    const words = data.map((d, i) => {
      const fontSize = fontScale(d.value);
      const angle = (i * 137.5) * (Math.PI / 180);
      const radius = 12 * Math.sqrt(i);
      return {
        text: d.text,
        value: d.value,
        fontSize,
        x: (width / 2) + radius * Math.cos(angle),
        y: (height / 2) + radius * Math.sin(angle),
        color: colorScale(i % 7)
      };
    });

    const g = svg.append('g');

    const wordNodes = g.selectAll('text')
      .data(words)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('x', d => d.x)
      .attr('y', d => d.y)
      .style('font-size', d => `${d.fontSize}px`)
      .style('font-weight', '700')
      .style('font-family', 'sans-serif')
      .style('fill', d => d.color)
      .style('cursor', 'pointer')
      .style('opacity', 0)
      .text(d => d.text);

    wordNodes.transition()
      .duration(800)
      .delay((d, i) => i * 15)
      .style('opacity', 0.9)
      .attr('transform', d => `rotate(${(d.value % 2 === 0 ? 0 : (Math.random() > 0.7 ? 90 : 0))}, ${d.x}, ${d.y})`);

    wordNodes
      .on('mouseover', function (event, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .style('opacity', 1)
          .style('font-size', `${d.fontSize * 1.25}px`)
          .style('filter', 'drop-shadow(0 0 8px rgba(139,92,246,0.6))');
      })
      .on('mouseout', function (event, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .style('opacity', 0.9)
          .style('font-size', `${d.fontSize}px`)
          .style('filter', 'none');
      })
      .on('click', (event, d) => {
        if (onWordClick) onWordClick(d.text);
      });

    // Zoom & pan support
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

  }, [data, width, height, onWordClick]);

  return (
    <div style={{ position: 'relative', width: '100%', display: 'flex', justifyContent: 'center' }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          background: 'rgba(15, 23, 42, 0.4)',
          borderRadius: '12px',
          border: '1px solid var(--border-color)',
          overflow: 'hidden'
        }}
      />
    </div>
  );
}

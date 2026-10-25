import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export default function D3BarChart({ data, width = 600, height = 350, title = '' }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 30, right: 30, bottom: 60, left: 50 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const x = d3.scaleBand()
      .domain(data.map(d => d.text || d.x))
      .range([0, innerW])
      .padding(0.25);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value || d.y) || 1])
      .nice()
      .range([innerH, 0]);

    const color = d3.scaleOrdinal()
      .range(['#8b5cf6', '#ec4899', '#3b82f6', '#10b981', '#f59e0b', '#06b6d4']);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Axes
    const xAxis = d3.axisBottom(x);
    const yAxis = d3.axisLeft(y).ticks(5);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(xAxis)
      .selectAll('text')
      .attr('transform', 'rotate(-30)')
      .style('text-anchor', 'end')
      .style('fill', '#94a3b8')
      .style('font-size', '11px');

    g.append('g')
      .call(yAxis)
      .selectAll('text')
      .style('fill', '#94a3b8')
      .style('font-size', '11px');

    // Bars with smooth grow animation
    g.selectAll('.bar')
      .data(data)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.text || d.x))
      .attr('width', x.bandwidth())
      .attr('y', innerH)
      .attr('height', 0)
      .attr('rx', 4)
      .attr('fill', (d, i) => color(i % 6))
      .style('opacity', 0.85)
      .transition()
      .duration(750)
      .delay((d, i) => i * 30)
      .attr('y', d => y(d.value || d.y))
      .attr('height', d => innerH - y(d.value || d.y));

    g.selectAll('.bar-label')
      .data(data)
      .enter().append('text')
      .attr('x', d => x(d.text || d.x) + x.bandwidth() / 2)
      .attr('y', d => y(d.value || d.y) - 6)
      .attr('text-anchor', 'middle')
      .text(d => d.value || d.y)
      .style('fill', '#e2e8f0')
      .style('font-size', '10px')
      .style('font-weight', '600')
      .style('opacity', 0)
      .transition()
      .duration(750)
      .delay((d, i) => i * 30 + 300)
      .style('opacity', 1);

  }, [data, width, height]);

  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      {title && <h4 style={{ color: 'var(--text-primary)', marginBottom: '8px', fontSize: '0.95rem' }}>{title}</h4>}
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

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export default function D3TreeMap({ data, width = 600, height = 350 }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const rootData = {
      name: 'root',
      children: data
    };

    const root = d3.hierarchy(rootData)
      .sum(d => d.value || 1)
      .sort((a, b) => b.value - a.value);

    d3.treemap()
      .size([width, height])
      .paddingInner(3)
      .round(true)(root);

    const color = d3.scaleOrdinal()
      .range(['#8b5cf6', '#ec4899', '#3b82f6', '#10b981', '#f59e0b', '#06b6d4', '#6366f1']);

    const g = svg.append('g');

    const leaf = g.selectAll('g')
      .data(root.leaves())
      .enter().append('g')
      .attr('transform', d => `translate(${d.x0},${d.y0})`)
      .style('cursor', 'pointer');

    leaf.append('rect')
      .attr('width', d => Math.max(0, d.x1 - d.x0))
      .attr('height', d => Math.max(0, d.y1 - d.y0))
      .attr('fill', (d, i) => color(i % 7))
      .attr('rx', 6)
      .style('opacity', 0.85)
      .style('transition', 'all 0.2s ease');

    leaf.append('text')
      .selectAll('tspan')
      .data(d => {
        const w = d.x1 - d.x0;
        const h = d.y1 - d.y0;
        if (w < 40 || h < 25) return [];
        return [
          { text: d.data.text || d.data.name, bold: true },
          { text: `(${d.data.value})`, bold: false }
        ];
      })
      .enter().append('tspan')
      .attr('x', 6)
      .attr('y', (d, i) => 16 + i * 14)
      .text(d => d.text)
      .style('font-size', '11px')
      .style('font-weight', d => d.bold ? '700' : '400')
      .style('fill', '#ffffff');

    leaf.on('mouseover', function (event, d) {
      d3.select(this).select('rect')
        .style('opacity', 1)
        .style('filter', 'brightness(1.2) drop-shadow(0 0 8px rgba(255,255,255,0.3))');
    }).on('mouseout', function (event, d) {
      d3.select(this).select('rect')
        .style('opacity', 0.85)
        .style('filter', 'none');
    });

  }, [data, width, height]);

  return (
    <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
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

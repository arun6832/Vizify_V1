import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export default function D3ForceGraph({ data, width = 600, height = 380 }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || !data.nodes || data.nodes.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const nodes = data.nodes.map(d => ({ ...d }));
    const links = data.links.map(d => ({ ...d }));

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(90))
      .force('charge', d3.forceManyBody().strength(-180))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => Math.max(15, (d.radius || 1) * 2)));

    const g = svg.append('g');

    const link = g.append('g')
      .attr('stroke', 'rgba(139, 92, 246, 0.35)')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke-width', d => Math.sqrt(d.value || 1) + 1);

    const nodeGroup = g.append('g')
      .selectAll('g')
      .data(nodes)
      .enter().append('g')
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    const radiusScale = d3.scaleLinear()
      .domain([1, d3.max(nodes, d => d.radius) || 1])
      .range([8, 22]);

    nodeGroup.append('circle')
      .attr('r', d => radiusScale(d.radius || 1))
      .attr('fill', (d, i) => color(i % 10))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 1.5)
      .style('filter', 'drop-shadow(0 0 6px rgba(139, 92, 246, 0.5))');

    nodeGroup.append('text')
      .text(d => d.id)
      .attr('x', 14)
      .attr('y', 4)
      .style('fill', '#e2e8f0')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .style('pointer-events', 'none');

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      nodeGroup
        .attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    const zoom = d3.zoom()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

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

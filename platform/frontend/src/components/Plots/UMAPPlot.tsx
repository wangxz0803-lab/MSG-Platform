import { useMemo } from 'react';
import Plot from './plotlyFactory';

export interface UMAPPoint {
  x: number;
  y: number;
  sample_id: string;
  source: string;
  sinr?: number | null;
}

interface Props {
  points: UMAPPoint[];
  colorBy?: 'source' | 'sinr';
  height?: number;
  title?: string;
}

export default function UMAPPlot({
  points,
  colorBy = 'source',
  height = 520,
  title = 'Latent UMAP',
}: Props) {
  const data = useMemo(() => {
    if (colorBy === 'sinr') {
      return [
        {
          x: points.map((p) => p.x),
          y: points.map((p) => p.y),
          mode: 'markers' as const,
          type: 'scattergl' as const,
          text: points.map(
            (p) => `${p.sample_id}<br>source: ${p.source}<br>sinr: ${p.sinr ?? 'n/a'}`,
          ),
          hoverinfo: 'text' as const,
          marker: {
            size: 6,
            color: points.map((p) => (p.sinr == null ? 0 : p.sinr)),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { title: { text: 'SINR (dB)' } },
          },
        },
      ];
    }
    const bySource = new Map<string, UMAPPoint[]>();
    points.forEach((p) => {
      const arr = bySource.get(p.source) ?? [];
      arr.push(p);
      bySource.set(p.source, arr);
    });
    return Array.from(bySource.entries()).map(([source, pts]) => ({
      x: pts.map((p) => p.x),
      y: pts.map((p) => p.y),
      mode: 'markers' as const,
      type: 'scattergl' as const,
      name: source,
      text: pts.map(
        (p) => `${p.sample_id}<br>source: ${p.source}<br>sinr: ${p.sinr ?? 'n/a'}`,
      ),
      hoverinfo: 'text' as const,
      marker: { size: 6 },
    }));
  }, [points, colorBy]);

  return (
    <Plot
      data={data}
      useResizeHandler
      style={{ width: '100%', height }}
      config={{ displaylogo: false, responsive: true }}
      layout={{
        title: { text: title },
        autosize: true,
        margin: { t: 40, l: 48, r: 24, b: 48 },
        xaxis: { title: { text: 'UMAP-1' }, zeroline: false },
        yaxis: { title: { text: 'UMAP-2' }, zeroline: false },
        legend: { orientation: 'h', y: -0.2 },
      }}
    />
  );
}

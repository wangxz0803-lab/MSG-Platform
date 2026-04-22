import { useMemo } from 'react';
import Plot from './plotlyFactory';
import type { ScalarSeries } from '@/api/types';

interface Props {
  series: ScalarSeries[];
  title?: string;
  height?: number;
  yAxisTitle?: string;
}

const COLORS = ['#1677ff', '#ff7a45', '#52c41a', '#722ed1', '#faad14', '#13c2c2'];

export default function LossChart({
  series,
  title = 'Training curves',
  height = 360,
  yAxisTitle = 'value',
}: Props) {
  const data = useMemo(
    () =>
      series.map((s, i) => ({
        x: s.steps,
        y: s.values,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: s.tag,
        line: { color: COLORS[i % COLORS.length], width: 2 },
      })),
    [series],
  );

  return (
    <Plot
      data={data}
      useResizeHandler
      style={{ width: '100%', height }}
      config={{ displaylogo: false, responsive: true }}
      layout={{
        title: { text: title },
        autosize: true,
        margin: { t: 40, l: 56, r: 24, b: 48 },
        xaxis: { title: { text: 'step' }, zeroline: false },
        yaxis: { title: { text: yAxisTitle }, zeroline: false },
        legend: { orientation: 'h', y: -0.2 },
      }}
    />
  );
}

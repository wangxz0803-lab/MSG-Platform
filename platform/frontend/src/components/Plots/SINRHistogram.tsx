import { useMemo } from 'react';
import Plot from './plotlyFactory';

interface Props {
  values: (number | null | undefined)[];
  title?: string;
  height?: number;
  xAxis?: string;
  nbinsx?: number;
}

export default function SINRHistogram({
  values,
  title = 'SINR distribution',
  height = 320,
  xAxis = 'SINR (dB)',
  nbinsx = 40,
}: Props) {
  const clean = useMemo(
    () => values.filter((v): v is number => v !== null && v !== undefined && !Number.isNaN(v)),
    [values],
  );
  return (
    <Plot
      data={[
        {
          x: clean,
          type: 'histogram',
          marker: { color: '#1677ff' },
          // @ts-expect-error plotly.js typings miss `nbinsx` on histograms but the runtime accepts it
          nbinsx,
        },
      ]}
      useResizeHandler
      style={{ width: '100%', height }}
      config={{ displaylogo: false, responsive: true }}
      layout={{
        title: { text: title },
        autosize: true,
        margin: { t: 40, l: 56, r: 24, b: 48 },
        xaxis: { title: { text: xAxis }, zeroline: false },
        yaxis: { title: { text: 'count' }, zeroline: false },
        bargap: 0.02,
      }}
    />
  );
}

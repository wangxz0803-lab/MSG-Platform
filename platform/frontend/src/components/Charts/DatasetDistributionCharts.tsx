import { Card, Col, Row } from 'antd';
import Plot from '@/components/Plots/plotlyFactory';
import type { DatasetSummary } from '@/api/types';

interface Props {
  datasets: DatasetSummary[];
}

const COLORS = [
  '#1677ff', '#52c41a', '#faad14', '#f5222d', '#722ed1',
  '#13c2c2', '#eb2f96', '#fa8c16', '#2f54eb', '#a0d911',
];

const FONT = { family: "'PingFang SC', 'Microsoft YaHei', sans-serif" };

const PLOTLY_CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
};

const BASE_LAYOUT: Partial<Plotly.Layout> = {
  autosize: true,
  font: FONT,
  margin: { t: 40, l: 56, r: 24, b: 64 },
};

export default function DatasetDistributionCharts({ datasets }: Props) {
  if (!datasets.length) return null;

  const sources = datasets.map((d) => d.source);
  const counts = datasets.map((d) => d.count);

  return (
    <Row gutter={[16, 16]}>
      {/* 1. Source Distribution Pie Chart */}
      <Col xs={24} lg={12}>
        <Card title="来源分布">
          <Plot
            data={[
              {
                values: counts,
                labels: sources,
                type: 'pie',
                marker: { colors: COLORS },
                textinfo: 'label+percent',
                hovertemplate: '%{label}: %{value} 个样本 (%{percent})<extra></extra>',
              },
            ]}
            useResizeHandler
            style={{ width: '100%', height: 360 }}
            config={PLOTLY_CONFIG}
            layout={{
              ...BASE_LAYOUT,
              margin: { t: 20, l: 20, r: 20, b: 20 },
              showlegend: true,
              legend: { orientation: 'h', y: -0.1 },
            }}
          />
        </Card>
      </Col>

      {/* 2. SNR Distribution Bar Chart with Error Bars */}
      <Col xs={24} lg={12}>
        <Card title="SNR 分布">
          <Plot
            data={[
              {
                x: sources,
                y: datasets.map((d) => d.snr_mean),
                error_y: {
                  type: 'data' as const,
                  array: datasets.map((d) => d.snr_std),
                  visible: true,
                  color: '#333',
                },
                type: 'bar',
                marker: {
                  color: sources.map((_, i) => COLORS[i % COLORS.length]),
                },
                hovertemplate: '%{x}<br>SNR: %{y:.2f} dB<extra></extra>',
              },
            ]}
            useResizeHandler
            style={{ width: '100%', height: 360 }}
            config={PLOTLY_CONFIG}
            layout={{
              ...BASE_LAYOUT,
              xaxis: { title: { text: '数据来源' } },
              yaxis: { title: { text: 'SNR (dB)' }, zeroline: false },
              bargap: 0.3,
            }}
          />
        </Card>
      </Col>

      {/* 3. Link Type Stacked Bar Chart */}
      <Col xs={24} lg={12}>
        <Card title="链路类型分布">
          <Plot
            data={[
              {
                x: sources,
                y: datasets.map((d) =>
                  d.links?.includes('UL') ? 1 : 0,
                ),
                name: 'UL (上行)',
                type: 'bar',
                marker: { color: '#1677ff' },
                hovertemplate: '%{x}: UL<extra></extra>',
              },
              {
                x: sources,
                y: datasets.map((d) =>
                  d.links?.includes('DL') ? 1 : 0,
                ),
                name: 'DL (下行)',
                type: 'bar',
                marker: { color: '#52c41a' },
                hovertemplate: '%{x}: DL<extra></extra>',
              },
            ]}
            useResizeHandler
            style={{ width: '100%', height: 360 }}
            config={PLOTLY_CONFIG}
            layout={{
              ...BASE_LAYOUT,
              barmode: 'stack',
              xaxis: { title: { text: '数据来源' } },
              yaxis: {
                title: { text: '链路类型数' },
                dtick: 1,
                zeroline: false,
              },
              legend: { orientation: 'h', y: 1.12 },
            }}
          />
        </Card>
      </Col>

      {/* 4. Signal Quality Overview Grouped Bar */}
      <Col xs={24} lg={12}>
        <Card title="信号质量概览">
          <Plot
            data={[
              {
                x: sources,
                y: datasets.map((d) => d.snr_mean),
                name: 'SNR 均值',
                type: 'bar',
                marker: { color: '#1677ff' },
                hovertemplate: '%{x}<br>SNR: %{y:.2f} dB<extra></extra>',
              },
              {
                x: sources,
                y: datasets.map((d) => d.sir_mean ?? 0),
                name: 'SIR 均值',
                type: 'bar',
                marker: { color: '#faad14' },
                hovertemplate: '%{x}<br>SIR: %{y:.2f} dB<extra></extra>',
              },
              {
                x: sources,
                y: datasets.map((d) => d.sinr_mean ?? 0),
                name: 'SINR 均值',
                type: 'bar',
                marker: { color: '#52c41a' },
                hovertemplate: '%{x}<br>SINR: %{y:.2f} dB<extra></extra>',
              },
            ]}
            useResizeHandler
            style={{ width: '100%', height: 360 }}
            config={PLOTLY_CONFIG}
            layout={{
              ...BASE_LAYOUT,
              barmode: 'group',
              xaxis: { title: { text: '数据来源' } },
              yaxis: { title: { text: '均值 (dB)' }, zeroline: false },
              legend: { orientation: 'h', y: 1.12 },
            }}
          />
        </Card>
      </Col>
    </Row>
  );
}

import { useMemo } from 'react';
import { Card, Col, Row, Typography } from 'antd';
import Plot from '@/components/Plots/plotlyFactory';
import type { Sample } from '@/api/types';

const { Text } = Typography;

interface Props {
  samples: Sample[];
  source: string;
}

const FONT = { family: "'PingFang SC', 'Microsoft YaHei', sans-serif" };

const PLOTLY_CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
};

const BASE_LAYOUT: Partial<Plotly.Layout> = {
  autosize: true,
  font: FONT,
  margin: { t: 40, l: 56, r: 24, b: 56 },
};

function cleanNumbers(values: (number | null | undefined)[]): number[] {
  return values.filter(
    (v): v is number => v !== null && v !== undefined && !Number.isNaN(v),
  );
}

export default function SampleDistributionCharts({ samples, source }: Props) {
  if (!samples.length) return null;

  const snrValues = useMemo(() => cleanNumbers(samples.map((s) => s.snr_dB)), [samples]);
  const sirValues = useMemo(() => cleanNumbers(samples.map((s) => s.sir_dB)), [samples]);
  const sinrValues = useMemo(() => cleanNumbers(samples.map((s) => s.sinr_dB)), [samples]);

  const ulCount = useMemo(() => samples.filter((s) => s.link === 'UL').length, [samples]);
  const dlCount = useMemo(() => samples.filter((s) => s.link === 'DL').length, [samples]);

  // For scatter plot: SNR vs SINR colored by link type
  const scatterData = useMemo(() => {
    const ul = samples.filter((s) => s.link === 'UL' && s.sinr_dB != null);
    const dl = samples.filter((s) => s.link === 'DL' && s.sinr_dB != null);
    return { ul, dl };
  }, [samples]);

  // For heatmap: SNR vs SIR
  const hasHeatmapData = snrValues.length > 5 && sirValues.length > 5;

  const titleSuffix = source ? ` - ${source}` : '';

  return (
    <Row gutter={[16, 16]}>
      {/* 1. SNR Histogram */}
      <Col xs={24} lg={12}>
        <Card title={`SNR 分布直方图${titleSuffix}`}>
          <Plot
            data={[
              {
                x: snrValues,
                type: 'histogram',
                marker: { color: '#1677ff' },
                // @ts-expect-error plotly.js typings miss nbinsx on histograms
                nbinsx: 40,
                hovertemplate: 'SNR: %{x:.1f} dB<br>数量: %{y}<extra></extra>',
              },
            ]}
            useResizeHandler
            style={{ width: '100%', height: 320 }}
            config={PLOTLY_CONFIG}
            layout={{
              ...BASE_LAYOUT,
              xaxis: { title: { text: 'SNR (dB)' }, zeroline: false },
              yaxis: { title: { text: '样本数' }, zeroline: false },
              bargap: 0.02,
            }}
          />
        </Card>
      </Col>

      {/* 2. SIR Histogram */}
      <Col xs={24} lg={12}>
        <Card title="SIR 分布直方图">
          {sirValues.length > 0 ? (
            <Plot
              data={[
                {
                  x: sirValues,
                  type: 'histogram',
                  marker: { color: '#faad14' },
                  // @ts-expect-error plotly.js typings miss nbinsx on histograms
                  nbinsx: 40,
                  hovertemplate: 'SIR: %{x:.1f} dB<br>数量: %{y}<extra></extra>',
                },
              ]}
              useResizeHandler
              style={{ width: '100%', height: 320 }}
              config={PLOTLY_CONFIG}
              layout={{
                ...BASE_LAYOUT,
                xaxis: { title: { text: 'SIR (dB)' }, zeroline: false },
                yaxis: { title: { text: '样本数' }, zeroline: false },
                bargap: 0.02,
              }}
            />
          ) : (
            <div style={{ height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Text type="secondary">暂无 SIR 数据</Text>
            </div>
          )}
        </Card>
      </Col>

      {/* 3. SINR Histogram */}
      <Col xs={24} lg={12}>
        <Card title="SINR 分布直方图">
          {sinrValues.length > 0 ? (
            <Plot
              data={[
                {
                  x: sinrValues,
                  type: 'histogram',
                  marker: { color: '#52c41a' },
                  // @ts-expect-error plotly.js typings miss nbinsx on histograms
                  nbinsx: 40,
                  hovertemplate: 'SINR: %{x:.1f} dB<br>数量: %{y}<extra></extra>',
                },
              ]}
              useResizeHandler
              style={{ width: '100%', height: 320 }}
              config={PLOTLY_CONFIG}
              layout={{
                ...BASE_LAYOUT,
                xaxis: { title: { text: 'SINR (dB)' }, zeroline: false },
                yaxis: { title: { text: '样本数' }, zeroline: false },
                bargap: 0.02,
              }}
            />
          ) : (
            <div style={{ height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Text type="secondary">暂无 SINR 数据</Text>
            </div>
          )}
        </Card>
      </Col>

      {/* 4. Link Type Pie Chart */}
      <Col xs={24} lg={12}>
        <Card title="链路类型占比">
          <Plot
            data={[
              {
                values: [ulCount, dlCount],
                labels: ['UL (上行)', 'DL (下行)'],
                type: 'pie',
                marker: { colors: ['#1677ff', '#52c41a'] },
                textinfo: 'label+percent',
                hovertemplate: '%{label}: %{value} 个样本 (%{percent})<extra></extra>',
              },
            ]}
            useResizeHandler
            style={{ width: '100%', height: 320 }}
            config={PLOTLY_CONFIG}
            layout={{
              ...BASE_LAYOUT,
              margin: { t: 20, l: 20, r: 20, b: 20 },
              showlegend: true,
              legend: { orientation: 'h', y: -0.05 },
            }}
          />
        </Card>
      </Col>

      {/* 5. SNR vs SINR Scatter Plot */}
      <Col xs={24} lg={12}>
        <Card title="信号质量散点图">
          {(scatterData.ul.length > 0 || scatterData.dl.length > 0) ? (
            <Plot
              data={[
                ...(scatterData.ul.length > 0
                  ? [
                      {
                        x: scatterData.ul.map((s) => s.snr_dB),
                        y: scatterData.ul.map((s) => s.sinr_dB),
                        mode: 'markers' as const,
                        type: 'scatter' as const,
                        name: 'UL (上行)',
                        marker: { color: '#1677ff', size: 5, opacity: 0.6 },
                        hovertemplate:
                          'SNR: %{x:.2f} dB<br>SINR: %{y:.2f} dB<extra>UL</extra>',
                      },
                    ]
                  : []),
                ...(scatterData.dl.length > 0
                  ? [
                      {
                        x: scatterData.dl.map((s) => s.snr_dB),
                        y: scatterData.dl.map((s) => s.sinr_dB),
                        mode: 'markers' as const,
                        type: 'scatter' as const,
                        name: 'DL (下行)',
                        marker: { color: '#52c41a', size: 5, opacity: 0.6 },
                        hovertemplate:
                          'SNR: %{x:.2f} dB<br>SINR: %{y:.2f} dB<extra>DL</extra>',
                      },
                    ]
                  : []),
              ]}
              useResizeHandler
              style={{ width: '100%', height: 320 }}
              config={PLOTLY_CONFIG}
              layout={{
                ...BASE_LAYOUT,
                xaxis: { title: { text: 'SNR (dB)' }, zeroline: false },
                yaxis: { title: { text: 'SINR (dB)' }, zeroline: false },
                legend: { orientation: 'h', y: 1.12 },
              }}
            />
          ) : (
            <div style={{ height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Text type="secondary">暂无散点数据</Text>
            </div>
          )}
        </Card>
      </Col>

      {/* 6. SNR vs SIR 2D Histogram Heatmap */}
      <Col xs={24} lg={12}>
        <Card title="数据覆盖热力图">
          {hasHeatmapData ? (
            <>
              <Plot
                data={[
                  {
                    x: samples.filter((s) => s.sir_dB != null).map((s) => s.snr_dB),
                    y: samples
                      .filter((s) => s.sir_dB != null)
                      .map((s) => s.sir_dB as number),
                    type: 'histogram2d',
                    colorscale: 'YlGnBu',
                    reversescale: true,
                    colorbar: { title: { text: '密度', side: 'right' } },
                    hovertemplate:
                      'SNR: %{x:.1f} dB<br>SIR: %{y:.1f} dB<br>数量: %{z}<extra></extra>',
                  },
                ]}
                useResizeHandler
                style={{ width: '100%', height: 320 }}
                config={PLOTLY_CONFIG}
                layout={{
                  ...BASE_LAYOUT,
                  xaxis: { title: { text: 'SNR (dB)' } },
                  yaxis: { title: { text: 'SIR (dB)' } },
                }}
              />
              <Text
                type="secondary"
                style={{ display: 'block', marginTop: 8, fontSize: 12 }}
              >
                颜色越深表示该区域数据越密集，空白区域可能需要补充采集
              </Text>
            </>
          ) : (
            <div style={{ height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Text type="secondary">
                数据量不足，无法生成热力图（需要 SNR 和 SIR 数据）
              </Text>
            </div>
          )}
        </Card>
      </Col>
    </Row>
  );
}

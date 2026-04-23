import { useCallback, useMemo, useRef, useEffect, useState } from 'react';
import {
  Breadcrumb,
  Button,
  Card,
  Col,
  Descriptions,
  Row,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { useSearchParams } from 'react-router-dom';
import { useChannels, useChannelDetail } from '@/api/queries';
import type { ChannelListItem, FeatureInfo } from '@/api/types';
import { formatNumber } from '@/utils/format';
import Plot from '@/components/Plots/plotlyFactory';

const { Title, Text } = Typography;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map a value in [min, max] to a blue-white-red RGB string. */
function blueRedColor(value: number, min: number, max: number): string {
  if (max === min) return 'rgb(255,255,255)';
  const t = (value - min) / (max - min); // 0..1
  // blue (0) -> white (0.5) -> red (1)
  if (t < 0.5) {
    const s = t / 0.5;
    const r = Math.round(s * 255);
    const g = Math.round(s * 255);
    const b = 255;
    return `rgb(${r},${g},${b})`;
  }
  const s = (t - 0.5) / 0.5;
  const r = 255;
  const g = Math.round(255 * (1 - s));
  const b = Math.round(255 * (1 - s));
  return `rgb(${r},${g},${b})`;
}

// ---------------------------------------------------------------------------
// HeatmapCanvas: renders a 2-D number array as a colored canvas
// ---------------------------------------------------------------------------

function HeatmapCanvas({
  data,
  title,
  width = 400,
  height = 220,
}: {
  data: number[][];
  title: string;
  width?: number;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = data.length;
    const cols = data[0].length;

    canvas.width = width;
    canvas.height = height;

    // Find global min/max
    let min = Infinity;
    let max = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r][c];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    const cellW = width / cols;
    const cellH = height / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = blueRedColor(data[r][c], min, max);
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
  }, [data, width, height]);

  return (
    <div style={{ marginBottom: 12 }}>
      <Text strong style={{ display: 'block', marginBottom: 4 }}>
        {title}
      </Text>
      {data.length === 0 ? (
        <Text type="secondary">No data</Text>
      ) : (
        <canvas
          ref={canvasRef}
          style={{ border: '1px solid #d9d9d9', borderRadius: 4, maxWidth: '100%' }}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// FeatureChart: renders a single FeatureInfo as a bar or line chart via Plotly
// ---------------------------------------------------------------------------

function FeatureChart({ info }: { info: FeatureInfo }) {
  const values = info.magnitude ?? info.values ?? [];
  if (values.length === 0) return <Text type="secondary">No data for {info.name}</Text>;

  const isLine = info.name.toLowerCase().includes('srs')
    || info.name.toLowerCase().includes('pmi')
    || info.name.toLowerCase().includes('dft');

  return (
    <Plot
      data={[
        {
          x: values.map((_, i) => i),
          y: values,
          type: isLine ? 'scatter' : 'bar',
          mode: isLine ? 'lines+markers' : undefined,
          marker: { color: '#1677ff', size: 3 },
          line: isLine ? { width: 1.5 } : undefined,
        },
      ]}
      useResizeHandler
      style={{ width: '100%', height: 180 }}
      config={{ displaylogo: false, responsive: true }}
      layout={{
        title: { text: `${info.name} (${info.shape.join('x')})`, font: { size: 13 } },
        autosize: true,
        margin: { t: 32, l: 48, r: 16, b: 32 },
        xaxis: { zeroline: false },
        yaxis: { zeroline: false },
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// DetailView
// ---------------------------------------------------------------------------

function DetailView({ sampleIndex, onBack }: { sampleIndex: number; onBack: () => void }) {
  const { data, isLoading, error } = useChannelDetail(sampleIndex);

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: 80 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <Text type="danger">Failed to load sample #{sampleIndex}.</Text>
        <br />
        <Button onClick={onBack} style={{ marginTop: 12 }}>
          Back
        </Button>
      </Card>
    );
  }

  const { meta, features } = data;

  // Group features by type for display ordering
  const featureEntries = Object.entries(features);

  return (
    <div className="msg-page">
      <Breadcrumb
        className="msg-breadcrumb"
        items={[
          {
            title: (
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  onBack();
                }}
              >
                信道浏览
              </a>
            ),
          },
          { title: `Sample #${sampleIndex}` },
        ]}
      />

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Button icon={<ArrowLeftOutlined />} onClick={onBack}>
            Back to list
          </Button>
          <Title level={3} style={{ margin: 0 }}>
            Channel Explorer: Sample #{sampleIndex}
          </Title>
        </div>

        <Row gutter={24}>
          {/* Left panel: Channel heatmaps + metadata */}
          <Col xs={24} lg={12}>
            <Card
              title="信道矩阵 (Channel Matrix)"
              size="small"
              styles={{ body: { padding: 16 } }}
            >
              <HeatmapCanvas data={data.channel_ideal} title="|H_ideal|" />
              <HeatmapCanvas data={data.channel_est} title="|H_est|" />
              <HeatmapCanvas data={data.channel_error} title="|Error| = |ideal - est|" />
              <Text type="secondary">
                Shape: {data.shape.join(' x ')}
              </Text>
            </Card>

            <Card
              title="元数据"
              size="small"
              style={{ marginTop: 16 }}
              styles={{ body: { padding: 16 } }}
            >
              <Descriptions column={2} size="small" bordered>
                <Descriptions.Item label="SNR">
                  {formatNumber(meta.snr_dB, 1)} dB
                </Descriptions.Item>
                <Descriptions.Item label="SIR">
                  {formatNumber(meta.sir_dB, 1)} dB
                </Descriptions.Item>
                <Descriptions.Item label="SINR">
                  {formatNumber(meta.sinr_dB, 1)} dB
                </Descriptions.Item>
                {meta.ul_sir_dB != null && (
                  <Descriptions.Item label="UL SIR">
                    {formatNumber(meta.ul_sir_dB, 1)} dB
                  </Descriptions.Item>
                )}
                {meta.dl_sir_dB != null && (
                  <Descriptions.Item label="DL SIR">
                    {formatNumber(meta.dl_sir_dB, 1)} dB
                  </Descriptions.Item>
                )}
                <Descriptions.Item label="Source">
                  <Tag color="blue">{meta.source}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Link">
                  <Tag>{meta.link}</Tag>
                  {meta.link_pairing === 'paired' && <Tag color="blue">配对</Tag>}
                </Descriptions.Item>
                <Descriptions.Item label="Est Mode">
                  <Tag color="green">{meta.channel_est_mode}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Serving Cell">
                  {meta.serving_cell_id}
                </Descriptions.Item>
                {meta.num_interfering_ues != null && (
                  <Descriptions.Item label="干扰 UE 数">
                    {meta.num_interfering_ues}
                  </Descriptions.Item>
                )}
                {meta.mobility_mode && (
                  <Descriptions.Item label="运动模式">
                    <Tag>{meta.mobility_mode}</Tag>
                  </Descriptions.Item>
                )}
                <Descriptions.Item label="Position">
                  {meta.ue_position
                    ? `[${meta.ue_position.map((v) => formatNumber(v, 1)).join(', ')}]`
                    : 'N/A'}
                </Descriptions.Item>
                <Descriptions.Item label="SSB RSRP (dBm)" span={2}>
                  {meta.ssb_rsrp_dBm?.length
                    ? meta.ssb_rsrp_dBm.map((v) => formatNumber(v, 1)).join(', ')
                    : 'N/A'}
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>

          {/* Right panel: Feature visualization */}
          <Col xs={24} lg={12}>
            <Card
              title="模型输入 (Model Input Features)"
              size="small"
              styles={{ body: { padding: 16 } }}
            >
              {featureEntries.length === 0 ? (
                <Text type="secondary">No features available.</Text>
              ) : (
                featureEntries.map(([key, info]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <FeatureChart info={info} />
                  </div>
                ))
              )}
            </Card>
          </Col>
        </Row>
      </Space>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ListView
// ---------------------------------------------------------------------------

function ListView({ onSelect }: { onSelect: (index: number) => void }) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);

  const { data, isLoading } = useChannels({
    limit: pageSize,
    offset: (page - 1) * pageSize,
  });

  const columns: ColumnsType<ChannelListItem> = useMemo(
    () => [
      {
        title: 'Index',
        dataIndex: 'index',
        key: 'index',
        width: 80,
        align: 'right',
      },
      {
        title: 'Source',
        key: 'source',
        render: (_: unknown, record: ChannelListItem) => (
          <Tag color="blue">{record.meta.source}</Tag>
        ),
      },
      {
        title: 'Link',
        key: 'link',
        render: (_: unknown, record: ChannelListItem) => <Tag>{record.meta.link}</Tag>,
      },
      {
        title: 'SNR (dB)',
        key: 'snr',
        align: 'right',
        render: (_: unknown, record: ChannelListItem) => formatNumber(record.meta.snr_dB, 1),
      },
      {
        title: 'SINR (dB)',
        key: 'sinr',
        align: 'right',
        render: (_: unknown, record: ChannelListItem) => formatNumber(record.meta.sinr_dB, 1),
      },
      {
        title: 'Channel Est Mode',
        key: 'est_mode',
        render: (_: unknown, record: ChannelListItem) => (
          <Tag color="green">{record.meta.channel_est_mode}</Tag>
        ),
      },
    ],
    [],
  );

  return (
    <div className="msg-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Title level={3} style={{ margin: 0 }}>
          信道浏览 (Channel Explorer)
        </Title>

        <Card>
          <Table<ChannelListItem>
            columns={columns}
            dataSource={data?.items ?? []}
            rowKey="index"
            loading={isLoading}
            onRow={(record) => ({
              onClick: () => onSelect(record.index),
              style: { cursor: 'pointer' },
            })}
            pagination={{
              total: data?.total ?? 0,
              pageSize,
              current: page,
              showSizeChanger: true,
              onChange: (p, ps) => {
                setPage(p);
                setPageSize(ps);
              },
            }}
            scroll={{ y: 540 }}
          />
        </Card>
      </Space>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChannelExplorer (top-level page)
// ---------------------------------------------------------------------------

export default function ChannelExplorer() {
  const [searchParams, setSearchParams] = useSearchParams();
  const sampleParam = searchParams.get('sample');
  const sampleIndex = sampleParam !== null ? Number(sampleParam) : null;

  const handleSelect = useCallback(
    (index: number) => {
      setSearchParams({ sample: String(index) });
    },
    [setSearchParams],
  );

  const handleBack = useCallback(() => {
    setSearchParams({});
  }, [setSearchParams]);

  if (sampleIndex !== null && !Number.isNaN(sampleIndex)) {
    return <DetailView sampleIndex={sampleIndex} onBack={handleBack} />;
  }

  return <ListView onSelect={handleSelect} />;
}

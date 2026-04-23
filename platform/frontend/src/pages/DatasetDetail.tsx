import { useMemo, useState } from 'react';
import { Breadcrumb, Card, Col, Row, Space, Statistic, Table, Tag, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { Link, useParams } from 'react-router-dom';
import { useDatasetSamples, useDatasets } from '@/api/queries';
import type { Sample } from '@/api/types';
import { formatDateTime, formatNumber } from '@/utils/format';
import SINRHistogram from '@/components/Plots/SINRHistogram';
import SampleDistributionCharts from '@/components/Charts/SampleDistributionCharts';

const { Title } = Typography;

export default function DatasetDetail() {
  const { source = '' } = useParams();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);

  const { data: datasets } = useDatasets({ source });
  const summary = datasets?.items.find((d) => d.source === source);

  const { data, isLoading } = useDatasetSamples(source, {
    limit: pageSize,
    offset: (page - 1) * pageSize,
  });

  const columns: ColumnsType<Sample> = useMemo(
    () => [
      { title: '样本 ID', dataIndex: 'sample_id', key: 'sample_id', ellipsis: true },
      { title: '链路', dataIndex: 'link', key: 'link' },
      {
        title: 'SNR (dB)',
        dataIndex: 'snr_dB',
        key: 'snr_dB',
        align: 'right',
        render: (v: number) => formatNumber(v, 2),
      },
      {
        title: 'SIR (dB)',
        dataIndex: 'sir_dB',
        key: 'sir_dB',
        align: 'right',
        render: (v: number | null) => formatNumber(v, 2),
      },
      {
        title: 'UL SIR',
        dataIndex: 'ul_sir_dB',
        key: 'ul_sir_dB',
        align: 'right',
        render: (v: number | null) => (v != null ? formatNumber(v, 2) : '-'),
      },
      {
        title: 'DL SIR',
        dataIndex: 'dl_sir_dB',
        key: 'dl_sir_dB',
        align: 'right',
        render: (v: number | null) => (v != null ? formatNumber(v, 2) : '-'),
      },
      {
        title: 'SINR (dB)',
        dataIndex: 'sinr_dB',
        key: 'sinr_dB',
        align: 'right',
        render: (v: number | null) => formatNumber(v, 2),
      },
      {
        title: '配对',
        dataIndex: 'link_pairing',
        key: 'link_pairing',
        render: (v: string) => (v === 'paired' ? <Tag color="blue">配对</Tag> : '-'),
      },
      {
        title: '时间戳',
        dataIndex: 'timestamp',
        key: 'timestamp',
        render: (v: string) => formatDateTime(v),
      },
      {
        title: '标签',
        dataIndex: 'tags',
        key: 'tags',
        render: (tags: string[]) => (
          <Space wrap>
            {tags?.map((t) => <Tag key={t}>{t}</Tag>)}
          </Space>
        ),
      },
    ],
    [],
  );

  const sinrValues = data?.items.map((s) => s.sinr_dB) ?? [];

  return (
    <div className="msg-page">
      <Breadcrumb
        className="msg-breadcrumb"
        items={[
          { title: <Link to="/datasets">数据集</Link> },
          { title: source },
        ]}
      />
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Title level={3} style={{ margin: 0 }}>
          {source}
        </Title>

        <Row gutter={16}>
          <Col span={4}>
            <Card>
              <Statistic title="总样本数" value={summary?.count ?? data?.total ?? 0} />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="SNR 均值"
                value={formatNumber(summary?.snr_mean, 2)}
                suffix="dB"
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="SIR 均值"
                value={formatNumber(summary?.sir_mean, 2)}
                suffix="dB"
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="UL SIR 均值"
                value={formatNumber(summary?.ul_sir_mean, 2)}
                suffix="dB"
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="DL SIR 均值"
                value={formatNumber(summary?.dl_sir_mean, 2)}
                suffix="dB"
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="SINR 均值"
                value={formatNumber(summary?.sinr_mean, 2)}
                suffix="dB"
              />
            </Card>
          </Col>
        </Row>
        {summary?.has_paired && (
          <Tag color="blue" style={{ marginTop: 8 }}>含配对 UL+DL 样本</Tag>
        )}

        <Card title="SINR 分布">
          <SINRHistogram values={sinrValues} />
        </Card>

        {data?.items && data.items.length > 0 && (
          <SampleDistributionCharts samples={data.items} source={source} />
        )}

        <Card title="样本列表">
          <Table<Sample>
            columns={columns}
            dataSource={data?.items ?? []}
            rowKey="sample_id"
            loading={isLoading}
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
            scroll={{ y: 480 }}
          />
        </Card>
      </Space>
    </div>
  );
}

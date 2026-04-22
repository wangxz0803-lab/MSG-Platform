import { Table } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import type { RunMetrics } from '@/api/types';
import { formatNumber } from '@/utils/format';

interface Row {
  key: string;
  metric: string;
  value: number | undefined;
}

const PRIORITY = ['ct', 'tw', 'knn_acc', 'nmse_dB'];

const LABELS: Record<string, string> = {
  ct: '连续性 (CT)',
  tw: '可信度 (TW)',
  knn_acc: 'kNN 准确率',
  nmse_dB: 'NMSE (dB)',
};

interface Props {
  metrics: RunMetrics;
}

export default function MetricsTable({ metrics }: Props) {
  const keys = Object.keys(metrics);
  const ordered = [
    ...PRIORITY.filter((k) => k in metrics),
    ...keys.filter((k) => !PRIORITY.includes(k)).sort(),
  ];
  const rows: Row[] = ordered.map((k) => ({
    key: k,
    metric: LABELS[k] ?? k,
    value: metrics[k],
  }));

  const columns: ColumnsType<Row> = [
    { title: '指标', dataIndex: 'metric', key: 'metric' },
    {
      title: '值',
      dataIndex: 'value',
      key: 'value',
      align: 'right',
      render: (v: number | undefined) => formatNumber(v, 6),
    },
  ];

  return <Table<Row> dataSource={rows} columns={columns} pagination={false} size="small" bordered />;
}

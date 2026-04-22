import { Table } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import type { RunMetrics } from '@/api/types';
import { formatNumber, shortSha } from '@/utils/format';

interface RunEntry {
  run_id: string;
  metrics: RunMetrics;
}

interface Props {
  runs: RunEntry[];
}

interface Row {
  key: string;
  metric: string;
  [runId: string]: string | number | undefined;
}

const LABELS: Record<string, string> = {
  ct: 'Continuity (CT)',
  tw: 'Trustworthiness (TW)',
  knn_acc: 'kNN accuracy',
  nmse_dB: 'NMSE (dB)',
};

export default function MetricsCompareTable({ runs }: Props) {
  const allKeys = Array.from(
    new Set(runs.flatMap((r) => Object.keys(r.metrics ?? {}))),
  ).sort();

  const rows: Row[] = allKeys.map((k) => {
    const row: Row = { key: k, metric: LABELS[k] ?? k };
    runs.forEach((r) => {
      row[r.run_id] = r.metrics?.[k];
    });
    return row;
  });

  const columns: ColumnsType<Row> = [
    { title: 'Metric', dataIndex: 'metric', key: 'metric', fixed: 'left', width: 200 },
    ...runs.map((r) => ({
      title: shortSha(r.run_id),
      dataIndex: r.run_id,
      key: r.run_id,
      align: 'right' as const,
      render: (v: number | undefined) => formatNumber(v, 6),
    })),
  ];

  return (
    <Table<Row>
      dataSource={rows}
      columns={columns}
      pagination={false}
      size="small"
      bordered
      scroll={{ x: 'max-content' }}
    />
  );
}

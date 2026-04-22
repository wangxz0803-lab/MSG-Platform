import { useMemo } from 'react';
import { Alert, Card, Space, Typography } from 'antd';
import { useSearchParams } from 'react-router-dom';
import { useCompareRuns, useRun } from '@/api/queries';
import MetricsCompareTable from '@/components/Metrics/MetricsCompareTable';
import LossChart from '@/components/Plots/LossChart';
import LoadingBox from '@/components/Common/LoadingBox';
import type { ScalarSeries } from '@/api/types';

const { Title } = Typography;

function useRunsScalars(ids: string[]) {
  // Run N hooks in a stable order; ids is derived from URL so it's stable per render.
  const runA = useRun(ids[0]);
  const runB = useRun(ids[1]);
  const runC = useRun(ids[2]);
  const runD = useRun(ids[3]);
  const all = [runA, runB, runC, runD].slice(0, ids.length);
  return all;
}

export default function Compare() {
  const [searchParams] = useSearchParams();
  const idsParam = searchParams.get('ids') ?? '';
  const ids = useMemo(
    () =>
      idsParam
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, 4),
    [idsParam],
  );
  const compare = useCompareRuns(ids);
  const runs = useRunsScalars(ids);

  if (ids.length < 2) {
    return (
      <div className="msg-page">
        <Alert
          type="info"
          message="请从训练记录页面选择 2 条或更多记录进行对比"
          showIcon
        />
      </div>
    );
  }

  if (compare.isLoading) {
    return (
      <div className="msg-page">
        <LoadingBox tip="加载对比数据中..." />
      </div>
    );
  }

  const overlaid: ScalarSeries[] = runs
    .flatMap((q, i) => {
      const run = q.data;
      if (!run?.tb_scalars) return [];
      const lossSeries = run.tb_scalars.find((s) => /loss/i.test(s.tag));
      if (!lossSeries) return [];
      return [
        {
          ...lossSeries,
          tag: `${ids[i].slice(0, 7)}/${lossSeries.tag}`,
        },
      ];
    })
    .filter(Boolean);

  return (
    <div className="msg-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Title level={3} style={{ margin: 0 }}>
          对比训练记录
        </Title>
        <Card title="指标">
          <MetricsCompareTable runs={compare.data?.runs ?? []} />
        </Card>
        <Card title="损失曲线">
          {overlaid.length > 0 ? (
            <LossChart series={overlaid} title="Loss (overlay)" />
          ) : (
            <Alert type="info" message="所选训练记录中无可用的损失曲线。" />
          )}
        </Card>
      </Space>
    </div>
  );
}

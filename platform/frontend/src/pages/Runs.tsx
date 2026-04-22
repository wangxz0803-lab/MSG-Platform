import { useMemo, useState } from 'react';
import { Button, Card, Form, Input, Popconfirm, Space, Table, Tag, Typography, message } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import type { ColumnsType, TableRowSelection } from 'antd/es/table/interface';
import { Link, useNavigate } from 'react-router-dom';
import { useDeleteRun, useRuns } from '@/api/queries';
import type { Run, RunFilters } from '@/api/types';
import { formatDateTime, formatNumber, shortSha } from '@/utils/format';

const { Title } = Typography;

export default function Runs() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState<RunFilters>({ limit: 20, offset: 0 });
  const [selected, setSelected] = useState<string[]>([]);
  const { data, isLoading } = useRuns(filters);
  const deleteMutation = useDeleteRun();

  const handleDelete = async (runId: string) => {
    try {
      await deleteMutation.mutateAsync(runId);
      message.success('训练记录已删除');
      setSelected((keys) => keys.filter((k) => k !== runId));
    } catch (e) {
      message.error((e as Error).message);
    }
  };

  const columns: ColumnsType<Run> = useMemo(
    () => [
      {
        title: '训练记录',
        dataIndex: 'run_id',
        key: 'run_id',
        render: (id: string) => <Link to={`/runs/${id}`}>{shortSha(id)}</Link>,
      },
      {
        title: '创建时间',
        dataIndex: 'created_at',
        key: 'created_at',
        render: (v: string) => formatDateTime(v),
      },
      {
        title: 'Git SHA',
        dataIndex: 'git_sha',
        key: 'git_sha',
        render: (v: string | null) => shortSha(v),
      },
      {
        title: 'Ckpt',
        key: 'ckpt_path',
        ellipsis: true,
        render: (_v, r) => r.ckpt_path ?? r.ckpt_best ?? '-',
      },
      { title: 'CT', key: 'ct', render: (_v, r) => formatNumber(r.metrics?.ct, 4) },
      { title: 'TW', key: 'tw', render: (_v, r) => formatNumber(r.metrics?.tw, 4) },
      {
        title: 'NMSE (dB)',
        key: 'nmse',
        render: (_v, r) => formatNumber(r.metrics?.nmse_dB, 2),
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
      {
        title: '操作',
        key: 'action',
        render: (_, record) => (
          <Popconfirm
            title={`确定要删除训练记录 ${shortSha(record.run_id)} 吗？`}
            description="此操作不可恢复。"
            onConfirm={() => handleDelete(record.run_id)}
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
          >
            <Button size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        ),
      },
    ],
    [deleteMutation],
  );

  const rowSelection: TableRowSelection<Run> = {
    selectedRowKeys: selected,
    onChange: (keys) => setSelected(keys.map(String)),
  };

  const compare = () => {
    if (selected.length < 2) {
      message.warning('请至少选择两条训练记录进行对比');
      return;
    }
    navigate(`/compare?ids=${selected.join(',')}`);
  };

  return (
    <div className="msg-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            训练记录
          </Title>
          <Button type="primary" disabled={selected.length < 2} onClick={compare}>
            对比 {selected.length ? `(${selected.length})` : ''}
          </Button>
        </Space>

        <Card>
          <Form
            layout="inline"
            initialValues={filters}
            onValuesChange={(_cv, av) => setFilters((f) => ({ ...f, ...av, offset: 0 }))}
          >
            <Form.Item label="标签" name="tag">
              <Input placeholder="按标签筛选" allowClear />
            </Form.Item>
          </Form>
        </Card>

        <Card>
          <Table<Run>
            rowSelection={rowSelection}
            columns={columns}
            dataSource={data?.items ?? []}
            rowKey="run_id"
            loading={isLoading}
            pagination={{
              total: data?.total ?? 0,
              pageSize: filters.limit,
              current: Math.floor((filters.offset ?? 0) / (filters.limit ?? 20)) + 1,
              onChange: (page, pageSize) =>
                setFilters((f) => ({ ...f, limit: pageSize, offset: (page - 1) * pageSize })),
            }}
          />
        </Card>
      </Space>
    </div>
  );
}

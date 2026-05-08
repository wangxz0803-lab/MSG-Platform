import { useState } from 'react';
import {
  Breadcrumb,
  Button,
  Card,
  Input,
  Radio,
  Space,
  Steps,
  Switch,
  Tabs,
  Typography,
  message,
} from 'antd';
import { Link, useNavigate } from 'react-router-dom';
import { useCollectDataset, useCreateBatchJobs, useCreateJob } from '@/api/queries';
import type { CreateCollectRequest, JobType } from '@/api/types';
import ConfigForm from '@/components/ConfigForm/ConfigForm';

const { Title, Paragraph, Text } = Typography;

const JOB_TYPE_OPTIONS: { value: JobType; label: string; section: string; description: string }[] = [
  { value: 'convert', label: 'Convert', section: 'convert', description: '将原始采集数据转换为标准数据集格式。' },
  { value: 'bridge', label: 'Bridge', section: 'bridge', description: '构建预训练桥接数据集。' },
  { value: 'eval', label: 'Eval', section: 'eval', description: '评估已导入模型的检查点。' },
  { value: 'infer', label: 'Infer', section: 'infer', description: '为数据集生成嵌入向量。' },
  { value: 'export', label: 'Export', section: 'export', description: '将检查点导出为 ONNX / TorchScript。' },
  { value: 'report', label: 'Report', section: 'report', description: '生成 HTML/Markdown 报告。' },
  { value: 'simulate', label: 'Simulate', section: 'simulate', description: '端到端运行仿真流水线。' },
];

export default function JobCreate() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [type, setType] = useState<JobType>('simulate');
  const [displayName, setDisplayName] = useState<string>('');
  const [formData, setFormData] = useState<Record<string, unknown>>({});

  const [batchMode, setBatchMode] = useState(false);
  const [batchConfigs, setBatchConfigs] = useState<Record<string, unknown>[]>([{}]);
  const [activeBatchIdx, setActiveBatchIdx] = useState(0);

  const createJob = useCreateJob();
  const collect = useCollectDataset();
  const createBatch = useCreateBatchJobs();

  const typeOpt = JOB_TYPE_OPTIONS.find((o) => o.value === type)!;
  const isCollectFlow = type === 'simulate';

  const submit = async () => {
    try {
      if (batchMode) {
        const res = await createBatch.mutateAsync({
          type,
          configs: batchConfigs,
          display_name_prefix: displayName || `batch-${type}`,
        });
        message.success(`已创建 ${res.jobs.length} 个任务`);
        navigate('/jobs');
      } else if (isCollectFlow) {
        const res = await collect.mutateAsync({ source: type as CreateCollectRequest['source'], config_overrides: formData });
        message.success(`任务 ${res.job_id} 已排队`);
        navigate(`/jobs/${res.job_id}`);
      } else {
        const job = await createJob.mutateAsync({
          type,
          config_overrides: formData,
          display_name: displayName || undefined,
        });
        message.success(`任务 ${job.job_id} 已创建`);
        navigate(`/jobs/${job.job_id}`);
      }
    } catch (e) {
      message.error((e as Error).message);
    }
  };

  return (
    <div className="msg-page">
      <Breadcrumb
        className="msg-breadcrumb"
        items={[{ title: <Link to="/jobs">任务</Link> }, { title: '新建' }]}
      />
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Title level={3} style={{ margin: 0 }}>
          新建任务
        </Title>

        <Steps
          current={step}
          items={[{ title: '选择类型' }, { title: '配置' }, { title: '检查并提交' }]}
        />

        {step === 0 && (
          <Card>
            <Paragraph type="secondary">选择要运行的任务类型。</Paragraph>
            <Radio.Group
              value={type}
              onChange={(e) => setType(e.target.value)}
              optionType="button"
              buttonStyle="solid"
              style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}
            >
              {JOB_TYPE_OPTIONS.map((o) => (
                <Radio.Button value={o.value} key={o.value}>
                  {o.label}
                </Radio.Button>
              ))}
            </Radio.Group>
            <Paragraph type="secondary" style={{ marginTop: 16 }}>
              {typeOpt.description}
            </Paragraph>
            <Space style={{ marginTop: 16 }}>
              <Text>显示名称 (可选):</Text>
              <Input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                style={{ width: 320 }}
                placeholder="例如 baseline-msg-v1"
              />
            </Space>
              <Space style={{ marginTop: 16 }}>
                <Text>批量模式:</Text>
                <Switch checked={batchMode} onChange={setBatchMode} />
                {batchMode && (
                  <Text type="secondary">提交多组不同参数配置进行批量运行</Text>
                )}
              </Space>
            <div style={{ marginTop: 24 }}>
              <Button type="primary" onClick={() => setStep(1)}>
                下一步
              </Button>
            </div>
          </Card>
        )}

        {step === 1 && !batchMode && (
          <Card title={`配置 ${typeOpt.label}`}>
            <ConfigForm
              section={typeOpt.section.includes(',') ? typeOpt.section.split(',') : typeOpt.section}
              formData={formData}
              onChange={setFormData}
              onSubmit={(d) => {
                setFormData(d);
                setStep(2);
              }}
              submitText="预览"
            />
            <div style={{ marginTop: 16 }}>
              <Button onClick={() => setStep(0)}>上一步</Button>
            </div>
          </Card>
        )}

        {step === 1 && batchMode && (
          <Card title={`批量配置 ${typeOpt.label}`}>
            <Tabs
              type="editable-card"
              activeKey={String(activeBatchIdx)}
              onChange={(k) => setActiveBatchIdx(Number(k))}
              onEdit={(targetKey, action) => {
                if (action === 'add') {
                  setBatchConfigs([...batchConfigs, { ...batchConfigs[batchConfigs.length - 1] }]);
                  setActiveBatchIdx(batchConfigs.length);
                } else if (action === 'remove' && batchConfigs.length > 1) {
                  const idx = Number(targetKey);
                  const next = batchConfigs.filter((_, i) => i !== idx);
                  setBatchConfigs(next);
                  setActiveBatchIdx(Math.min(activeBatchIdx, next.length - 1));
                }
              }}
              items={batchConfigs.map((cfg, i) => ({
                key: String(i),
                label: `配置 ${i + 1}`,
                children: (
                  <ConfigForm
                    section={typeOpt.section.includes(',') ? typeOpt.section.split(',') : typeOpt.section}
                    formData={cfg}
                    onChange={(d) => {
                      const next = [...batchConfigs];
                      next[i] = d;
                      setBatchConfigs(next);
                    }}
                    onSubmit={(d) => {
                      const next = [...batchConfigs];
                      next[i] = d;
                      setBatchConfigs(next);
                      setStep(2);
                    }}
                    submitText="预览全部"
                  />
                ),
              }))}
            />
            <div style={{ marginTop: 16 }}>
              <Button onClick={() => setStep(0)}>上一步</Button>
            </div>
          </Card>
        )}

        {step === 2 && !batchMode && (
          <Card title="检查">
            <Paragraph>
              <Text strong>类型:</Text> {type}
            </Paragraph>
            {displayName && (
              <Paragraph>
                <Text strong>显示名称:</Text> {displayName}
              </Paragraph>
            )}
            <Paragraph>
              <Text strong>配置覆盖:</Text>
            </Paragraph>
            <pre className="msg-yaml-block">{JSON.stringify(formData, null, 2)}</pre>
            <Space>
              <Button onClick={() => setStep(1)}>上一步</Button>
              <Button
                type="primary"
                loading={createJob.isPending || collect.isPending}
                onClick={submit}
              >
                提交
              </Button>
            </Space>
          </Card>
        )}

        {step === 2 && batchMode && (
          <Card title="检查 (批量)">
            <Paragraph>
              <Text strong>类型:</Text> {type}
            </Paragraph>
            <Paragraph>
              <Text strong>配置数量:</Text> {batchConfigs.length}
            </Paragraph>
            {batchConfigs.map((cfg, i) => (
              <div key={i}>
                <Text strong>配置 {i + 1}:</Text>
                <pre className="msg-yaml-block">{JSON.stringify(cfg, null, 2)}</pre>
              </div>
            ))}
            <Space>
              <Button onClick={() => setStep(1)}>上一步</Button>
              <Button
                type="primary"
                loading={createBatch.isPending}
                onClick={submit}
              >
                提交 {batchConfigs.length} 个任务
              </Button>
            </Space>
          </Card>
        )}
      </Space>
    </div>
  );
}

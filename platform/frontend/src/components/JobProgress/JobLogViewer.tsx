import { useEffect, useRef } from 'react';
import { Card, Space, Switch, Typography } from 'antd';
import { useState } from 'react';

const { Text } = Typography;

interface Props {
  lines: string[];
  title?: string;
  loading?: boolean;
}

export default function JobLogViewer({ lines, title = '日志尾部', loading }: Props) {
  const [autoScroll, setAutoScroll] = useState(true);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (autoScroll && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [lines, autoScroll]);

  return (
    <Card
      title={title}
      extra={
        <Space>
          <Text type="secondary">自动滚动</Text>
          <Switch size="small" checked={autoScroll} onChange={setAutoScroll} />
        </Space>
      }
      loading={loading}
    >
      <div className="msg-log-viewer" ref={ref}>
        {lines.length === 0 ? (
          <div style={{ color: '#6e7681' }}>[暂无日志输出]</div>
        ) : (
          lines.map((line, i) => (
            <div className="log-line" key={i}>
              {line}
            </div>
          ))
        )}
      </div>
    </Card>
  );
}

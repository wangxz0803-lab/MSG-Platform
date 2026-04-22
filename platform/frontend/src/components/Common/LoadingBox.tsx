import { Spin } from 'antd';

interface Props {
  tip?: string;
  height?: number | string;
}

export default function LoadingBox({ tip = '加载中...', height = 240 }: Props) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
        height,
      }}
    >
      <Spin tip={tip}>
        <div style={{ padding: 32 }} />
      </Spin>
    </div>
  );
}

import { Empty } from 'antd';
import type { ReactNode } from 'react';

interface Props {
  description?: ReactNode;
  children?: ReactNode;
}

export default function EmptyState({ description = '暂无数据', children }: Props) {
  return (
    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={description}>
      {children}
    </Empty>
  );
}

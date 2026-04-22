import { Layout, Space, Switch, Tag, Tooltip, Typography } from 'antd';
import { BulbFilled, BulbOutlined, GithubOutlined } from '@ant-design/icons';
import { useUIStore } from '@/store';
import { useHealth } from '@/api/queries';

const { Header } = Layout;
const { Text } = Typography;

export default function AppHeader() {
  const themeMode = useUIStore((s) => s.themeMode);
  const toggleTheme = useUIStore((s) => s.toggleTheme);
  const { data, isError } = useHealth();

  const healthColor =
    isError || !data ? 'red' : data.status === 'ok' ? 'green' : data.status === 'degraded' ? 'orange' : 'red';
  const healthLabel = isError ? 'down' : data?.status ?? '...';

  return (
    <Header
      style={{
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid rgba(5,5,5,0.06)',
        position: 'sticky',
        top: 0,
        zIndex: 9,
      }}
    >
      <Space size="middle">
        <Text strong style={{ fontSize: 16 }}>
          研究平台
        </Text>
        <Tag color={healthColor}>API {healthLabel}</Tag>
        {data?.version && <Tag>v{data.version}</Tag>}
        {data?.db && <Tag color={data.db === 'ok' ? 'green' : 'red'}>db {data.db}</Tag>}
      </Space>
      <Space size="large">
        <Tooltip title="切换深色模式">
          <Switch
            checked={themeMode === 'dark'}
            checkedChildren={<BulbFilled />}
            unCheckedChildren={<BulbOutlined />}
            onChange={toggleTheme}
          />
        </Tooltip>
        <a
          href="https://github.com/"
          target="_blank"
          rel="noreferrer"
          style={{ color: 'inherit' }}
          aria-label="GitHub"
        >
          <GithubOutlined style={{ fontSize: 18 }} />
        </a>
      </Space>
    </Header>
  );
}

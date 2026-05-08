import { Layout, Menu } from 'antd';
import {
  DashboardOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  DeploymentUnitOutlined,
  DiffOutlined,
  AppstoreOutlined,
  PlusCircleOutlined,
  ToolOutlined,
} from '@ant-design/icons';
import { useLocation, useNavigate } from 'react-router-dom';
import { useUIStore } from '@/store';

const { Sider } = Layout;

const MENU_ITEMS = [
  { key: '/', icon: <DashboardOutlined />, label: '仪表盘' },
  { key: '/datasets', icon: <DatabaseOutlined />, label: '数据集' },
  { key: '/channels', icon: <EyeOutlined />, label: '信道浏览' },
  { key: '/collect', icon: <PlusCircleOutlined />, label: '数据采集' },
  { key: '/process', icon: <ToolOutlined />, label: '数据处理' },
  { key: '/jobs', icon: <ThunderboltOutlined />, label: '任务' },
  { key: '/runs', icon: <ExperimentOutlined />, label: '运行记录' },
  { key: '/compare', icon: <DiffOutlined />, label: '对比' },
  { key: '/models', icon: <DeploymentUnitOutlined />, label: '模型' },
];

function deriveSelectedKey(pathname: string): string {
  if (pathname === '/') return '/';
  if (pathname.startsWith('/datasets')) return '/datasets';
  if (pathname.startsWith('/channels')) return '/channels';
  if (pathname.startsWith('/collect')) return '/collect';
  if (pathname.startsWith('/process')) return '/process';
  if (pathname.startsWith('/jobs')) return '/jobs';
  if (pathname.startsWith('/runs')) return '/runs';
  if (pathname.startsWith('/compare')) return '/compare';
  if (pathname.startsWith('/models')) return '/models';
  return '/';
}

export default function AppSider() {
  const navigate = useNavigate();
  const location = useLocation();
  const collapsed = useUIStore((s) => s.siderCollapsed);
  const setCollapsed = useUIStore((s) => s.setSiderCollapsed);
  const selectedKey = deriveSelectedKey(location.pathname);

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      width={220}
      style={{
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
        height: '100vh',
        overflow: 'auto',
        zIndex: 10,
      }}
    >
      <div
        style={{
          height: 56,
          color: '#ffffff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 600,
          letterSpacing: 1,
          borderBottom: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        <AppstoreOutlined style={{ marginRight: collapsed ? 0 : 8 }} />
        {collapsed ? '' : 'ChannelHub'}
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[selectedKey]}
        onClick={(info) => navigate(info.key)}
        items={MENU_ITEMS}
      />
    </Sider>
  );
}

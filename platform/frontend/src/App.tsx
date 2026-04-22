import { Layout } from 'antd';
import { Route, Routes, Navigate } from 'react-router-dom';
import AppSider from './components/Layout/AppSider';
import AppHeader from './components/Layout/AppHeader';
import ErrorBoundary from './components/Common/ErrorBoundary';
import Dashboard from './pages/Dashboard';
import Datasets from './pages/Datasets';
import DatasetDetail from './pages/DatasetDetail';
import Jobs from './pages/Jobs';
import JobDetail from './pages/JobDetail';
import JobCreate from './pages/JobCreate';
import Runs from './pages/Runs';
import RunDetail from './pages/RunDetail';
import Compare from './pages/Compare';
import Models from './pages/Models';
import CollectWizard from './pages/CollectWizard';
import DataProcess from './pages/DataProcess';
import ChannelExplorer from './pages/ChannelExplorer';
import { useUIStore } from './store';

const { Content } = Layout;

export default function App() {
  const collapsed = useUIStore((s) => s.siderCollapsed);
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <AppSider />
      <Layout style={{ marginLeft: collapsed ? 80 : 220, transition: 'margin-left 0.2s' }}>
        <AppHeader />
        <Content>
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/datasets" element={<Datasets />} />
              <Route path="/datasets/:source" element={<DatasetDetail />} />
              <Route path="/jobs" element={<Jobs />} />
              <Route path="/jobs/new" element={<JobCreate />} />
              <Route path="/jobs/:jobId" element={<JobDetail />} />
              <Route path="/runs" element={<Runs />} />
              <Route path="/runs/:runId" element={<RunDetail />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/collect" element={<CollectWizard />} />
              <Route path="/process" element={<DataProcess />} />
              <Route path="/models" element={<Models />} />
              <Route path="/channels" element={<ChannelExplorer />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ErrorBoundary>
        </Content>
      </Layout>
    </Layout>
  );
}

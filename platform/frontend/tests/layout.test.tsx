import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import AppSider from '../src/components/Layout/AppSider';

// Guard ResizeObserver for AntD in jsdom.
class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as unknown as { ResizeObserver: typeof RO }).ResizeObserver = RO;

vi.mock('../src/api/queries', async () => {
  return {
    useHealth: () => ({ data: { status: 'ok', version: '0.1', db: 'ok' }, isError: false }),
  };
});

describe('AppSider', () => {
  it('renders all main nav items', () => {
    const qc = new QueryClient();
    render(
      <ConfigProvider>
        <QueryClientProvider client={qc}>
          <MemoryRouter initialEntries={['/']}>
            <AppSider />
          </MemoryRouter>
        </QueryClientProvider>
      </ConfigProvider>,
    );
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Datasets')).toBeInTheDocument();
    expect(screen.getByText('Jobs')).toBeInTheDocument();
    expect(screen.getByText('Runs')).toBeInTheDocument();
    expect(screen.getByText('Compare')).toBeInTheDocument();
    expect(screen.getByText('Models')).toBeInTheDocument();
  });
});

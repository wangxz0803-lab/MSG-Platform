import { describe, it, expect } from 'vitest';
import {
  isActiveStatus,
  statusToBadge,
  statusToColor,
  statusToLabel,
} from '../src/utils/jobStatus';

describe('jobStatus mapping', () => {
  it('maps all statuses to a badge color', () => {
    expect(statusToBadge('queued')).toBe('default');
    expect(statusToBadge('running')).toBe('processing');
    expect(statusToBadge('completed')).toBe('success');
    expect(statusToBadge('failed')).toBe('error');
    expect(statusToBadge('cancelled')).toBe('warning');
  });

  it('maps statuses to hex colors', () => {
    expect(statusToColor('running')).toMatch(/^#/);
    expect(statusToColor('failed')).toMatch(/^#/);
  });

  it('labels statuses human-readable', () => {
    expect(statusToLabel('running')).toBe('Running');
    expect(statusToLabel('completed')).toBe('Completed');
  });

  it('detects active statuses', () => {
    expect(isActiveStatus('running')).toBe(true);
    expect(isActiveStatus('queued')).toBe(true);
    expect(isActiveStatus('completed')).toBe(false);
    expect(isActiveStatus('failed')).toBe(false);
    expect(isActiveStatus('cancelled')).toBe(false);
  });
});

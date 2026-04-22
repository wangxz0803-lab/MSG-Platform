import { describe, it, expect } from 'vitest';
import {
  formatBytes,
  formatDuration,
  formatNumber,
  shortSha,
  formatDateTime,
} from '../src/utils/format';

describe('formatDuration', () => {
  it('returns dash for null/undefined', () => {
    expect(formatDuration(null)).toBe('-');
    expect(formatDuration(undefined)).toBe('-');
  });
  it('formats sub-minute seconds', () => {
    expect(formatDuration(12)).toBe('12s');
  });
  it('formats minutes and seconds', () => {
    expect(formatDuration(125)).toBe('2m 5s');
  });
  it('formats hours and minutes', () => {
    expect(formatDuration(3665)).toBe('1h 1m');
  });
});

describe('formatBytes', () => {
  it('handles zero and bytes', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(500)).toBe('500 B');
  });
  it('formats KB', () => {
    expect(formatBytes(2048)).toBe('2.00 KB');
  });
  it('formats MB', () => {
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.00 MB');
  });
  it('handles nullish', () => {
    expect(formatBytes(null)).toBe('-');
  });
});

describe('formatNumber', () => {
  it('formats with default digits', () => {
    expect(formatNumber(1.23456789)).toBe('1.2346');
  });
  it('respects custom digits', () => {
    expect(formatNumber(1.5, 2)).toBe('1.50');
  });
  it('handles nullish', () => {
    expect(formatNumber(undefined)).toBe('-');
  });
});

describe('shortSha', () => {
  it('truncates to 7 chars', () => {
    expect(shortSha('abcdef0123456789')).toBe('abcdef0');
  });
  it('handles empty', () => {
    expect(shortSha(null)).toBe('-');
  });
});

describe('formatDateTime', () => {
  it('returns dash for empty', () => {
    expect(formatDateTime(null)).toBe('-');
  });
  it('formats ISO strings', () => {
    const out = formatDateTime('2024-01-02T03:04:05Z');
    expect(out).toMatch(/2024-01-0[12] \d{2}:\d{2}:\d{2}/);
  });
});

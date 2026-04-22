import { describe, it, expect, expectTypeOf } from 'vitest';
import type {
  DatasetFilters,
  Job,
  JobStatus,
  Run,
  ModelArtifact,
} from '../src/api/types';

// Pure compile-time type assertions. Keep this file minimal so we don't depend
// on backend behaviour, but validate that the schemas we claim are real.
describe('api types', () => {
  it('JobStatus is a union', () => {
    const s: JobStatus = 'queued';
    expect(s).toBe('queued');
  });

  it('Job has required fields', () => {
    expectTypeOf<Job>().toHaveProperty('job_id');
    expectTypeOf<Job>().toHaveProperty('type');
    expectTypeOf<Job>().toHaveProperty('status');
    expectTypeOf<Job>().toHaveProperty('progress_pct');
  });

  it('Run has metrics dict', () => {
    expectTypeOf<Run>().toHaveProperty('metrics');
    expectTypeOf<Run>().toHaveProperty('run_id');
  });

  it('ModelArtifact has format and size', () => {
    expectTypeOf<ModelArtifact>().toHaveProperty('format');
    expectTypeOf<ModelArtifact>().toHaveProperty('size_bytes');
  });

  it('DatasetFilters is all-optional', () => {
    const f: DatasetFilters = {};
    expectTypeOf(f).toEqualTypeOf<DatasetFilters>();
  });
});

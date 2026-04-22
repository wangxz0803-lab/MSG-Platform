import { useMemo } from 'react';
import { Alert, Skeleton } from 'antd';
import Form from '@rjsf/antd';
import validator from '@rjsf/validator-ajv8';
import type { IChangeEvent } from '@rjsf/core';
import type { RJSFSchema, UiSchema } from '@rjsf/utils';
import { useConfigDefaults, useConfigSchema } from '@/api/queries';

interface Props {
  /**
   * Optional section key; narrows the JSON schema to one top-level key of the Hydra config
   * (e.g. "train", "data", "simulate"). When undefined the full schema is used.
   */
  section?: string | string[];
  formData?: Record<string, unknown>;
  onChange?: (data: Record<string, unknown>) => void;
  onSubmit?: (data: Record<string, unknown>) => void;
  submitText?: string;
  disabled?: boolean;
  uiSchema?: UiSchema;
}

function pickSection(schema: RJSFSchema, section?: string | string[]): RJSFSchema {
  if (!section) return schema;
  if (!schema || typeof schema !== 'object') return schema;
  const props = (schema.properties ?? {}) as Record<string, RJSFSchema>;

  if (Array.isArray(section)) {
    const merged: Record<string, RJSFSchema> = {};
    for (const s of section) {
      const sub = props[s];
      if (sub && typeof sub === 'object' && sub.properties) {
        Object.assign(merged, sub.properties as Record<string, RJSFSchema>);
      }
    }
    if (Object.keys(merged).length === 0) return schema;
    return { type: 'object', properties: merged } as RJSFSchema;
  }

  const sub = props[section];
  if (sub) return sub;
  return schema;
}

export default function ConfigForm({
  section,
  formData,
  onChange,
  onSubmit,
  submitText = '提交',
  disabled,
  uiSchema,
}: Props) {
  const { data: schema, isLoading: schemaLoading, error: schemaError } = useConfigSchema();
  const defaultsSection = Array.isArray(section) ? section[0] : section;
  const { data: defaults, isLoading: defaultsLoading } = useConfigDefaults(defaultsSection);

  const effectiveSchema = useMemo(
    () => (schema ? pickSection(schema as RJSFSchema, section) : ({} as RJSFSchema)),
    [schema, section],
  );

  const effectiveData = useMemo(() => {
    const base = (defaults ?? {}) as Record<string, unknown>;
    return { ...base, ...(formData ?? {}) };
  }, [defaults, formData]);

  if (schemaError) {
    return <Alert type="error" message="加载配置 schema 失败" description={String(schemaError)} />;
  }
  if (schemaLoading || defaultsLoading) {
    return <Skeleton active paragraph={{ rows: 6 }} />;
  }

  return (
    <Form
      schema={effectiveSchema}
      uiSchema={uiSchema}
      formData={effectiveData}
      validator={validator}
      disabled={disabled}
      onChange={(e: IChangeEvent) => onChange?.(e.formData as Record<string, unknown>)}
      onSubmit={(e: IChangeEvent) => onSubmit?.(e.formData as Record<string, unknown>)}
      showErrorList={false}
    >
      <button type="submit" className="ant-btn ant-btn-primary" disabled={disabled}>
        {submitText}
      </button>
    </Form>
  );
}

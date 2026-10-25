import { Play, X } from 'lucide-react'
import { FormEvent, useEffect, useMemo, useState } from 'react'
import type { PipelinePlugin } from '../types'

interface PipelineDialogProps {
  open: boolean
  serial: string
  plugins: PipelinePlugin[]
  eyebrow?: string
  title?: string
  submitLabel?: string
  parameterOverrides?: Record<string, unknown>
  onClose: () => void
  onSubmit: (pipelineKey: string, parameters: Record<string, unknown>) => Promise<void>
}

const EMPTY_PARAMETER_OVERRIDES: Record<string, unknown> = {}

function datetimeLocalValue(value: unknown): string {
  if (!value) return ''
  const parsed = new Date(String(value))
  if (Number.isNaN(parsed.getTime())) return String(value)
  return parsed.toISOString().slice(0, 16)
}

function utcIsoValue(value: unknown): string {
  const text = String(value)
  const parsed = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(text) ? text : `${text}Z`)
  return parsed.toISOString()
}

function initialValues(
  plugin: PipelinePlugin | undefined,
  serial: string,
  parameterOverrides: Record<string, unknown>,
): Record<string, unknown> {
  if (!plugin) return {}
  return Object.fromEntries(
    plugin.parameters.map((parameter) => {
      const hasSerialDefault = Object.prototype.hasOwnProperty.call(
        parameter.serial_defaults ?? {},
        serial,
      )
      const defaultValue = hasSerialDefault ? parameter.serial_defaults?.[serial] : parameter.default
      const value = Object.prototype.hasOwnProperty.call(parameterOverrides, parameter.key)
        ? parameterOverrides[parameter.key]
        : defaultValue
      if (parameter.key === 'serial') return [parameter.key, serial]
      if (parameter.type === 'datetime') {
        return [parameter.key, datetimeLocalValue(value)]
      }
      return [parameter.key, value ?? '']
    }),
  )
}

export function PipelineDialog({
  open,
  serial,
  plugins,
  eyebrow = 'New analysis job',
  title = 'Run battery pipeline',
  submitLabel = 'Start job',
  parameterOverrides = EMPTY_PARAMETER_OVERRIDES,
  onClose,
  onSubmit,
}: PipelineDialogProps) {
  const [pluginKey, setPluginKey] = useState(plugins[0]?.key ?? '')
  const plugin = plugins.find((item) => item.key === pluginKey) ?? plugins[0]
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (plugin) setValues(initialValues(plugin, serial, parameterOverrides))
  }, [plugin, serial, parameterOverrides])

  useEffect(() => {
    if (open && !plugins.some((item) => item.key === pluginKey)) {
      setPluginKey(plugins[0]?.key ?? '')
    }
  }, [open, pluginKey, plugins])

  useEffect(() => {
    if (open) setError('')
  }, [open, pluginKey, serial])

  const groups = useMemo(() => {
    const result = new Map<string, PipelinePlugin['parameters']>()
    plugin?.parameters.forEach((parameter) => {
      result.set(parameter.group, [...(result.get(parameter.group) ?? []), parameter])
    })
    return [...result.entries()]
  }, [plugin])

  if (!open || !plugin) return null

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const cleaned = Object.fromEntries(
        Object.entries(values)
          .filter(([, value]) => value !== '' && value !== null)
          .map(([key, value]) => {
            const definition = plugin.parameters.find((item) => item.key === key)
            if (definition?.type === 'datetime' && value) {
              return [key, utcIsoValue(value)]
            }
            if (definition?.type === 'number' || definition?.type === 'integer') {
              return [key, Number(value)]
            }
            return [key, value]
          }),
      )
      await onSubmit(plugin.key, cleaned)
      onClose()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="pipeline-dialog" role="dialog" aria-modal="true" aria-labelledby="pipeline-title" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div>
            <span className="eyebrow">{eyebrow}</span>
            <h2 id="pipeline-title">{title}</h2>
          </div>
          <button className="icon-button" type="button" onClick={onClose} title="Close">
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <form onSubmit={submit}>
          <label className="field full-field">
            <span>Pipeline</span>
            <select value={plugin.key} onChange={(event) => setPluginKey(event.target.value)}>
              {plugins.map((item) => <option key={item.key} value={item.key}>{item.name}</option>)}
            </select>
            <small>{plugin.description}</small>
          </label>

          <div className="dialog-scroll">
            {groups.map(([group, parameters]) => (
              <fieldset key={group}>
                <legend>{group}</legend>
                <div className="field-grid">
                  {parameters.map((parameter) => {
                    const value = values[parameter.key]
                    if (parameter.type === 'boolean') {
                      return (
                        <label className="toggle-field" key={parameter.key}>
                          <input
                            type="checkbox"
                            checked={Boolean(value)}
                            onChange={(event) => setValues((current) => ({ ...current, [parameter.key]: event.target.checked }))}
                          />
                          <span>{parameter.label}</span>
                        </label>
                      )
                    }
                    return (
                      <label className="field" key={parameter.key}>
                        <span>{parameter.label}</span>
                        {parameter.type === 'choice' ? (
                          <select value={String(value ?? '')} onChange={(event) => setValues((current) => ({ ...current, [parameter.key]: event.target.value }))}>
                            {parameter.choices.map((choice) => <option key={choice} value={choice}>{choice.replaceAll('_', ' ')}</option>)}
                          </select>
                        ) : (
                          <input
                            type={parameter.type === 'datetime' ? 'datetime-local' : parameter.type === 'number' || parameter.type === 'integer' ? 'number' : 'text'}
                            step={parameter.type === 'integer' ? '1' : parameter.type === 'number' ? 'any' : undefined}
                            value={String(value ?? '')}
                            required={parameter.required}
                            onChange={(event) => setValues((current) => ({ ...current, [parameter.key]: event.target.value }))}
                          />
                        )}
                      </label>
                    )
                  })}
                </div>
              </fieldset>
            ))}
          </div>

          {error && <div className="form-error">{error}</div>}
          <footer>
            <button className="button secondary" type="button" onClick={onClose}>Cancel</button>
            <button className="button primary" type="submit" disabled={busy}>
              <Play size={16} aria-hidden="true" />
              {busy ? 'Starting...' : submitLabel}
            </button>
          </footer>
        </form>
      </div>
    </div>
  )
}

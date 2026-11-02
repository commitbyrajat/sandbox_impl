import { useId, useRef, useState } from 'react'
import { Highlight } from 'prism-react-renderer'
import type { PrismTheme } from 'prism-react-renderer'
import {
  Braces,
  Check,
  ChevronDown,
  Copy,
  Database,
  Download,
  FileCode2,
  Layers3,
  Play,
  Table2,
  Workflow,
  WrapText,
} from 'lucide-react'
import './execution-evidence.css'

interface Dataset {
  result_id: string
  row_count: number
  columns: string[]
  preview: Record<string, unknown>[]
  truncated?: boolean
}
interface Calculation {
  step_id: string
  executed: boolean
  code?: string | null
  result_ids?: string[]
  reused_from_step_id?: string | null
  output: unknown
}
interface Evidence {
  queries?: string[]
  datasets?: Dataset[]
  calculations?: Calculation[]
}

const codeTheme: PrismTheme = {
  plain: { color: '#d5e2f3', backgroundColor: '#10243e' },
  styles: [
    {
      types: ['comment', 'prolog', 'doctype'],
      style: { color: '#7b93ae', fontStyle: 'italic' },
    },
    { types: ['keyword', 'operator'], style: { color: '#94bdff' } },
    { types: ['string', 'char', 'attr-value'], style: { color: '#a3d8bf' } },
    { types: ['number', 'boolean', 'constant'], style: { color: '#f4c391' } },
    {
      types: ['function', 'class-name', 'builtin'],
      style: { color: '#d5b5f0' },
    },
    { types: ['punctuation'], style: { color: '#9cafc7' } },
  ],
}

function saveSource(source: string, filename: string) {
  const url = URL.createObjectURL(
    new Blob([source], { type: 'text/plain;charset=utf-8' }),
  )
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function CodeCard({
  code,
  language,
  title,
  subtitle,
  filename,
  expanded,
  onToggle,
  reused = false,
}: {
  code?: string | null
  language: 'sql' | 'python'
  title: string
  subtitle: string
  filename: string
  expanded: boolean
  onToggle: () => void
  reused?: boolean
}) {
  const id = useId()
  const [copied, setCopied] = useState(false)
  const [copyError, setCopyError] = useState(false)
  const [wrap, setWrap] = useState(language === 'sql')
  const source = useRef<HTMLPreElement>(null)
  const lines = code?.split('\n').length || 0
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code || '')
      setCopied(true)
      setCopyError(false)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      setCopyError(true)
      source.current?.focus()
    }
  }
  return (
    <article
      className={`execution-code-card ${language} ${expanded ? 'is-expanded' : ''}`}
      aria-label={title}
    >
      <div className="execution-code-header">
        <button
          className="code-disclosure"
          onClick={onToggle}
          aria-expanded={expanded}
          aria-controls={id}
        >
          <ChevronDown size={14} className="code-chevron" />
          <span className="code-language-mark">
            {language === 'sql' ? (
              <Database size={16} />
            ) : (
              <FileCode2 size={17} />
            )}
          </span>
          <span className="code-file-label">
            <strong>{title}</strong>
            <small>{subtitle}</small>
          </span>
        </button>
        <div className="code-header-actions">
          <span className={`code-run-status ${reused ? 'reused' : ''}`}>
            {reused ? <Layers3 size={11} /> : <Check size={11} />}
            {reused ? 'Reused' : 'Executed'}
          </span>
          <button
            onClick={copy}
            disabled={!code}
            className="code-action"
            aria-label={`Copy ${title}`}
            title="Copy exact source"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
          <button
            onClick={() => saveSource(code || '', filename)}
            disabled={!code}
            className="code-action"
            aria-label={`Download ${title}`}
            title={`Download ${filename}`}
          >
            <Download size={14} />
          </button>
        </div>
      </div>
      {expanded && (
        <div
          id={id}
          className={`execution-code-content ${wrap ? 'source-wrap' : ''}`}
        >
          {code ? (
            <Highlight code={code} language={language} theme={codeTheme}>
              {({ tokens, getLineProps, getTokenProps, style }) => (
                <pre
                  ref={source}
                  className="execution-source"
                  style={style}
                  tabIndex={0}
                  aria-label={`${title} source code`}
                >
                  <code>
                    {tokens.map((line, index) => (
                      <span
                        key={index}
                        {...getLineProps({ line })}
                        className="execution-source-line"
                      >
                        <span className="source-line-number" aria-hidden="true">
                          {index + 1}
                        </span>
                        <span className="source-line-content">
                          {line.map((token, i) => (
                            <span key={i} {...getTokenProps({ token })} />
                          ))}
                        </span>
                      </span>
                    ))}
                  </code>
                </pre>
              )}
            </Highlight>
          ) : (
            <div className="source-unavailable">
              <FileCode2 size={20} />
              <p>Source code was not retained for this result.</p>
            </div>
          )}
          <div className="code-card-footer">
            <span>
              <span className="code-language-dot" />
              {language === 'sql' ? 'SQL' : 'Python'}
              <i />
              {lines} {lines === 1 ? 'line' : 'lines'}
            </span>
            <button
              className="source-wrap-button"
              onClick={() => setWrap(!wrap)}
              aria-pressed={wrap}
            >
              <WrapText size={11} />
              Wrap lines
            </button>
            <span aria-live="polite">
              {copyError
                ? 'Clipboard unavailable. Use Download to save the source.'
                : copied
                  ? 'Exact source copied'
                  : reused
                    ? 'Previously executed source · result reused'
                    : 'Exact executed source'}
            </span>
          </div>
        </div>
      )}
    </article>
  )
}

export default function ExecutionEvidence({ data }: { data: Evidence }) {
  const queries = data.queries || []
  const calculations = data.calculations || []
  const datasets = data.datasets || []
  const [tab, setTab] = useState<'sql' | 'python' | 'results'>(
    queries.length ? 'sql' : calculations.length ? 'python' : 'results',
  )
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    'sql-0': true,
  })
  const tabId = useId()
  const tabs = [
    { id: 'sql' as const, label: 'SQL', count: queries.length, Icon: Database },
    ...(calculations.length
      ? [
          {
            id: 'python' as const,
            label: 'Python',
            count: calculations.length,
            Icon: FileCode2,
          },
        ]
      : []),
    {
      id: 'results' as const,
      label: 'Results',
      count: datasets.length,
      Icon: Table2,
    },
  ]
  const codeIds =
    tab === 'sql'
      ? queries.map((_, i) => `sql-${i}`)
      : calculations.map((_, i) => `python-${i}`)
  const isExpanded = (id: string) => expanded[id] ?? id.startsWith('python-')
  const allExpanded = codeIds.length > 0 && codeIds.every(isExpanded)
  const toggleAll = () =>
    setExpanded((value) => ({
      ...value,
      ...Object.fromEntries(codeIds.map((id) => [id, !allExpanded])),
    }))
  const stringValue = (value: unknown) =>
    value == null
      ? '—'
      : typeof value === 'object'
        ? JSON.stringify(value)
        : String(value)

  return (
    <div className="execution-evidence">
      <div className="execution-stats">
        <div>
          <strong>{queries.length}</strong>
          <span>SQL statements</span>
        </div>
        <div>
          <strong>
            {datasets.reduce((sum, dataset) => sum + dataset.row_count, 0)}
          </strong>
          <span>Result rows</span>
        </div>
        <div>
          <strong>{calculations.filter((run) => run.executed).length}</strong>
          <span>Python runs</span>
        </div>
      </div>
      <div
        className="evidence-tabs"
        role="tablist"
        aria-label="Execution evidence"
      >
        {tabs.map(({ id, label, count, Icon }, i) => (
          <button
            key={id}
            id={`${tabId}-${id}-tab`}
            role="tab"
            aria-selected={tab === id}
            aria-controls={`${tabId}-panel`}
            tabIndex={tab === id ? 0 : -1}
            className={tab === id ? 'active' : ''}
            onClick={() => setTab(id)}
            onKeyDown={(event) => {
              if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
                event.preventDefault()
                const next =
                  tabs[
                    (i + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) %
                      tabs.length
                  ]
                setTab(next.id)
                document.getElementById(`${tabId}-${next.id}-tab`)?.focus()
              }
            }}
          >
            <Icon size={14} />
            {label}
            <span>{count}</span>
          </button>
        ))}
      </div>
      <div
        id={`${tabId}-panel`}
        role="tabpanel"
        aria-labelledby={`${tabId}-${tab}-tab`}
      >
        {tab !== 'results' && (
          <>
            <div className="evidence-section-heading">
              <div>
                <h4>
                  {tab === 'sql'
                    ? 'Executed SQL statements'
                    : 'Python execution'}
                </h4>
                <p>
                  {tab === 'sql'
                    ? 'Every recorded statement, in execution order.'
                    : 'Sandbox source and the inputs used for each calculation.'}
                </p>
              </div>
              {codeIds.length > 1 && (
                <button onClick={toggleAll} className="evidence-expand-all">
                  {allExpanded ? 'Collapse all' : 'Expand all'}
                  <ChevronDown size={12} />
                </button>
              )}
            </div>
            {tab === 'sql' ? (
              queries.length ? (
                <div className="execution-code-stack">
                  {queries.map((code, index) => (
                    <CodeCard
                      key={index}
                      title={`Statement ${String(index + 1).padStart(2, '0')}`}
                      subtitle={`${String(index + 1).padStart(2, '0')} of ${String(queries.length).padStart(2, '0')} · Query execution`}
                      language="sql"
                      code={code}
                      filename={`statement-${String(index + 1).padStart(2, '0')}.sql`}
                      expanded={isExpanded(`sql-${index}`)}
                      onToggle={() =>
                        setExpanded((value) => ({
                          ...value,
                          [`sql-${index}`]: !isExpanded(`sql-${index}`),
                        }))
                      }
                    />
                  ))}
                </div>
              ) : (
                <div className="evidence-empty">
                  <Database size={24} />
                  <p>No SQL statements were recorded for this response.</p>
                </div>
              )
            ) : (
              <div className="execution-code-stack">
                {calculations.map((run, index) => (
                  <CodeCard
                    key={`${run.step_id}-${index}`}
                    title={`Python · ${run.step_id}`}
                    subtitle={
                      run.executed
                        ? `Monty sandbox${run.result_ids?.length ? ' · Inputs: ' + run.result_ids.join(', ') : ''}`
                        : `Result reused${run.reused_from_step_id ? ' from ' + run.reused_from_step_id : ''}`
                    }
                    language="python"
                    code={run.code}
                    reused={!run.executed}
                    filename={`analysis-${run.step_id.replace(/[^\w-]/g, '_')}.py`}
                    expanded={isExpanded(`python-${index}`)}
                    onToggle={() =>
                      setExpanded((value) => ({
                        ...value,
                        [`python-${index}`]: !isExpanded(`python-${index}`),
                      }))
                    }
                  />
                ))}
              </div>
            )}
            <div className="evidence-provenance">
              <Workflow size={12} />
              <span>
                {tab === 'sql'
                  ? 'Statements are captured by the analysis pipeline.'
                  : 'Source comes from retained execution records.'}
              </span>
            </div>
          </>
        )}
        {tab === 'results' && (
          <>
            <div className="evidence-section-heading">
              <div>
                <h4>Returned data & calculations</h4>
                <p>The retained output used to prepare your answer.</p>
              </div>
              <Table2 size={17} />
            </div>
            {datasets.map((dataset) => (
              <div className="evidence-dataset" key={dataset.result_id}>
                <div>
                  <span>
                    <Database size={13} />
                    {dataset.result_id}
                  </span>
                  <small>
                    {dataset.row_count} rows
                    {dataset.truncated ? ' · truncated' : ''}
                  </small>
                </div>
                <div className="data-table">
                  <table>
                    <thead>
                      <tr>
                        {dataset.columns.slice(0, 6).map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {dataset.preview.map((row, i) => (
                        <tr key={i}>
                          {dataset.columns.slice(0, 6).map((column) => (
                            <td key={column}>{stringValue(row[column])}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <small>Preview · up to 5 rows and 6 columns</small>
              </div>
            ))}
            {calculations.map((run, i) => (
              <details key={i} className="calculation-result">
                <summary>
                  <Braces size={14} />
                  <span>Calculation output · {run.step_id}</span>
                  <ChevronDown size={13} />
                </summary>
                <pre>{JSON.stringify(run.output, null, 2)}</pre>
              </details>
            ))}
            {!datasets.length && !calculations.length && (
              <div className="evidence-empty">
                <Play size={22} />
                <p>No result datasets were recorded for this response.</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

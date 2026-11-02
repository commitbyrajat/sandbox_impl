import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  AudioLines,
  BarChart3,
  BookOpen,
  Building2,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Copy,
  Database,
  FileCheck2,
  FileText,
  Fingerprint,
  History,
  Layers3,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Network,
  Plus,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
  TrendingUp,
  Users,
  Waypoints,
  X,
  TriangleAlert,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { consumeEvents, getJSON } from './api'
import type {
  Activity,
  ProgressEvent,
  Run,
  StageDefinition,
  StageState,
  Tenant,
  Workspace,
} from './api'

const ExecutionEvidence = lazy(() => import('./ExecutionEvidence'))

const DEFAULT_STAGES: StageDefinition[] = [
  {
    id: 'resolve',
    name: 'Connect workspace',
    description: 'Resolve your tenant and workspace.',
  },
  {
    id: 'classify',
    name: 'Understand question',
    description: 'Identify the objective and constraints.',
  },
  {
    id: 'plan',
    name: 'Build analysis plan',
    description: 'Choose the right metrics and approach.',
  },
  {
    id: 'publish',
    name: 'Record plan',
    description: 'Save a reviewable execution plan.',
  },
  {
    id: 'load_sources',
    name: 'Connect data',
    description: 'Load and verify your data sources.',
  },
  {
    id: 'execute',
    name: 'Run analysis',
    description: 'Execute queries and calculations.',
  },
  {
    id: 'validate_results',
    name: 'Validate results',
    description: 'Check the evidence and results.',
  },
  {
    id: 'summarize',
    name: 'Prepare answer',
    description: 'Bring the findings together.',
  },
]
const STAGE_ICONS = [
  Fingerprint,
  MessageSquareText,
  Waypoints,
  FileText,
  Database,
  BarChart3,
  ShieldCheck,
  Sparkles,
]
const SHORT_NAMES = [
  'Connect',
  'Understand',
  'Plan',
  'Record',
  'Load data',
  'Analyze',
  'Validate',
  'Answer',
]
const formatTime = (seconds: number) =>
  seconds < 60
    ? `${Math.floor(seconds)}s`
    : `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
const pretty = (value: unknown) => JSON.stringify(value, null, 2)
const display = (value: unknown) =>
  value === null || value === undefined
    ? '—'
    : typeof value === 'object'
      ? JSON.stringify(value)
      : String(value)
const asList = (value: unknown) => (Array.isArray(value) ? value : [])

function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand ${compact ? 'compact' : ''}`}>
      <img src="/brand-mark.svg" alt="HDFC Bank mark" width="38" height="38" />
      {!compact && (
        <div>
          <strong>HDFC BANK</strong>
          <span>INSIGHT WORKSPACE</span>
        </div>
      )}
    </div>
  )
}

function Outcome({ stage }: { stage: StageState }) {
  const data = stage.outcome
  if (!data || Object.keys(data).length === 0) return null
  return (
    <div className="outcome">
      {stage.id === 'resolve' && (
        <div className="key-values">
          {['tenant', 'workspace', 'model'].map((key) => (
            <div key={key}>
              <span>{key}</span>
              <strong>{display(data[key])}</strong>
            </div>
          ))}
        </div>
      )}
      {stage.id === 'classify' && (
        <>
          <div className="detail-callout">{display(data.objective)}</div>
          <div className="detail-tags">
            {asList(data.constraints).map((item, i) => (
              <span key={i}>{display(item)}</span>
            ))}
          </div>
        </>
      )}
      {stage.id === 'plan' && (
        <>
          <span className="route-tag">
            <Network size={13} />
            {data.execution_mode === 'sql_python'
              ? 'SQL + Python'
              : data.execution_mode === 'sql'
                ? 'SQL analysis'
                : 'Clarification'}
          </span>
          <p>{display(data.execution_reason || data.response)}</p>
          <div className="plan-steps">
            {asList(data.steps).map((item: Record<string, unknown>, i) => (
              <div key={i}>
                <span>{String(i + 1).padStart(2, '0')}</span>
                <div>
                  <b>{display(item.action)}</b>
                  <p>{display(item.instruction)}</p>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
      {stage.id === 'publish' && (
        <div className="detail-callout">
          <FileCheck2 size={18} />
          <div>
            <strong>Plan saved for review</strong>
            <p className="mono">{display(data.publication)}</p>
          </div>
        </div>
      )}
      {stage.id === 'load_sources' && (
        <>
          <p>
            {data.connected
              ? 'Your workspace sources are loaded and ready for schema inspection.'
              : 'Data execution is not needed for this response.'}
          </p>
          <div className="table-chips">
            {asList(data.tables).map((table, i) => (
              <span key={i}>
                <Database size={12} />
                {display(table)}
              </span>
            ))}
          </div>
        </>
      )}
      {stage.id === 'execute' && (
        <Suspense
          fallback={<div className="muted">Loading execution evidence…</div>}
        >
          <ExecutionEvidence data={data} />
        </Suspense>
      )}
      {stage.id === 'validate_results' && (
        <>
          <div className="detail-callout success">
            <ShieldCheck size={20} />
            <strong>{display(data.summary)}</strong>
          </div>
          {asList(data.checks).map((check, i) => (
            <div className="validation-check" key={i}>
              <CheckCheck size={15} />
              <span>
                {display(check.boundary)} · {asList(check.checks).join(', ')}
              </span>
            </div>
          ))}
        </>
      )}
      {stage.id === 'summarize' && (
        <div className="detail-callout success">
          <CheckCheck size={20} />
          <strong>{display(data.summary)}</strong>
        </div>
      )}
      <details className="raw-outcome">
        <summary>View structured outcome</summary>
        <pre>{pretty(data)}</pre>
      </details>
    </div>
  )
}

function Journey({
  run,
  stages,
}: {
  run: Run | null
  stages: StageDefinition[]
}) {
  const [selection, setSelection] = useState<string | null>(null)
  const [follow, setFollow] = useState(true)
  const [tab, setTab] = useState<'overview' | 'activity'>('overview')
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    setSelection(null)
    setFollow(true)
    setTab('overview')
  }, [run?.id])
  useEffect(() => {
    if (run?.status !== 'running') return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [run?.status])
  const items: StageState[] =
    run?.stages ||
    stages.map((stage) => ({ ...stage, status: 'pending' as const }))
  const live =
    items.find((stage) =>
      ['running', 'failed', 'timeout', 'cancelled'].includes(stage.status),
    ) || [...items].reverse().find((stage) => stage.status === 'completed')
  const current =
    items.find((stage) => stage.id === (follow ? live?.id : selection)) ||
    items[0]
  const complete = items.filter(
    (stage) => stage.status === 'completed' || stage.status === 'skipped',
  ).length
  const activity =
    run?.activity.filter((item) => item.stage === current.id) || []
  return (
    <section
      className={`journey card ${run ? 'has-run' : ''}`}
      aria-label="Analysis journey"
    >
      <div className="section-head">
        <div>
          <div className="eyebrow">
            <Waypoints size={14} /> FULL VISIBILITY
          </div>
          <h2>
            {run
              ? 'Your analysis, step by step'
              : 'A clear path from question to answer'}
          </h2>
          <p>
            {run
              ? `${complete} of ${items.length} stages completed`
              : 'Follow the process. Inspect the evidence. Understand the outcome.'}
          </p>
        </div>
        {run ? (
          <div className="journey-time">
            <Clock3 size={14} />
            {formatTime(
              Math.max(0, ((run.finished || now) - run.started) / 1000),
            )}
            <span className={`status-pill ${run.status}`}>
              {run.status === 'running' && <i />}
              {run.status === 'completed'
                ? 'Complete'
                : run.status === 'failed'
                  ? 'Needs attention'
                  : run.status === 'cancelled'
                    ? 'Stopped'
                    : 'Live analysis'}
            </span>
          </div>
        ) : (
          <span className="quiet-badge">8 connected stages</span>
        )}
      </div>
      {!run ? (
        <div className="journey-preview">
          {items.map((stage, i) => {
            const Icon = STAGE_ICONS[i]
            return (
              <div key={stage.id}>
                <span className="preview-icon">
                  <Icon size={19} />
                </span>
                <span className="preview-index">0{i + 1}</span>
                <strong>{SHORT_NAMES[i]}</strong>
              </div>
            )
          })}
        </div>
      ) : (
        <>
          <div
            className="stage-segments"
            aria-label={`${complete} stages completed`}
          >
            {items.map((stage) => (
              <span className={stage.status} key={stage.id} />
            ))}
          </div>
          <div className="journey-content">
            <div className="stage-list">
              {items.map((stage, i) => {
                const Icon = STAGE_ICONS[i]
                return (
                  <button
                    className={`stage-row ${stage.status} ${current.id === stage.id ? 'selected' : ''}`}
                    key={stage.id}
                    onClick={() => {
                      setSelection(stage.id)
                      setFollow(false)
                    }}
                    aria-label={`${stage.name}: ${stage.status}`}
                  >
                    <span className="stage-icon">
                      {stage.status === 'completed' ? (
                        <Check size={16} />
                      ) : stage.status === 'running' ? (
                        <LoaderCircle size={17} className="spin" />
                      ) : stage.status === 'failed' ||
                        stage.status === 'timeout' ? (
                        <TriangleAlert size={16} />
                      ) : (
                        <Icon size={17} />
                      )}
                    </span>
                    <span className="stage-text">
                      <strong>{stage.name}</strong>
                      <small>
                        {stage.status === 'running'
                          ? 'In progress…'
                          : stage.status === 'pending'
                            ? 'Waiting'
                            : stage.status === 'completed'
                              ? 'Completed'
                              : stage.status === 'skipped'
                                ? 'Not required'
                                : stage.status === 'cancelled'
                                  ? 'Stopped'
                                  : 'Could not complete'}
                      </small>
                    </span>
                    {stage.duration_seconds !== undefined && (
                      <span className="stage-duration">
                        {formatTime(stage.duration_seconds)}
                      </span>
                    )}
                    <ChevronRight size={13} className="stage-chevron" />
                  </button>
                )
              })}
            </div>
            <div className="stage-detail">
              <div className="detail-heading">
                <div>
                  <span className="eyebrow">
                    STAGE {items.findIndex((s) => s.id === current.id) + 1} OF{' '}
                    {items.length}
                  </span>
                  <h3>{current.name}</h3>
                </div>
                <button
                  className={`follow-button ${follow ? 'active' : ''}`}
                  onClick={() => setFollow(!follow)}
                  aria-pressed={follow}
                >
                  <AudioLines size={13} />
                  Follow live
                </button>
              </div>
              <div className="detail-tabs">
                <button
                  className={tab === 'overview' ? 'active' : ''}
                  onClick={() => setTab('overview')}
                >
                  Outcome
                </button>
                <button
                  className={tab === 'activity' ? 'active' : ''}
                  onClick={() => setTab('activity')}
                >
                  Activity <span>{activity.length}</span>
                </button>
              </div>
              {tab === 'overview' ? (
                <>
                  {current.outcome &&
                  Object.keys(current.outcome).length > 0 ? (
                    <Outcome key={`${run.id}-${current.id}`} stage={current} />
                  ) : (
                    <div className="stage-empty">
                      {current.status === 'running' ? (
                        <LoaderCircle className="spin" size={25} />
                      ) : (
                        <Layers3 size={25} />
                      )}
                      <strong>
                        {current.status === 'running'
                          ? 'Analysis in progress'
                          : current.status === 'failed' ||
                              current.status === 'timeout'
                            ? 'This stage needs attention'
                            : current.status === 'cancelled'
                              ? 'Analysis stopped'
                              : 'Ready when you are'}
                      </strong>
                      <p>
                        {current.status === 'running'
                          ? current.description
                          : current.status === 'failed' ||
                              current.status === 'timeout'
                            ? run.error
                            : 'The outcome will appear here when this stage completes.'}
                      </p>
                      {current.status === 'running' && activity.length > 0 && (
                        <span className="latest-activity">
                          {activity.at(-1)?.message}
                        </span>
                      )}
                      <small>
                        Stages report actual work. Some model requests can take
                        several minutes.
                      </small>
                    </div>
                  )}
                </>
              ) : (
                <div className="activity-feed">
                  {activity.length === 0 ? (
                    <p className="muted">
                      No tool activity reported for this stage yet.
                    </p>
                  ) : (
                    activity.map((entry, i) => (
                      <div key={i}>
                        <span />
                        <p>{entry.message}</p>
                        <time>
                          {entry.timestamp
                            ? new Date(entry.timestamp).toLocaleTimeString([], {
                                hour: '2-digit',
                                minute: '2-digit',
                                second: '2-digit',
                              })
                            : ''}
                        </time>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  )
}

function Answer({ run }: { run: Run }) {
  const [copied, setCopied] = useState(false)
  const [copyError, setCopyError] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(run.answer || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      setCopyError(true)
    }
  }
  const download = () => {
    const url = URL.createObjectURL(
      new Blob(
        [`# ${run.question}\n\nWorkspace: ${run.workspace}\n\n${run.answer}`],
        { type: 'text/markdown;charset=utf-8' },
      ),
    )
    const link = document.createElement('a')
    link.href = url
    link.download = `insight-${run.id.slice(0, 8)}.md`
    link.click()
    URL.revokeObjectURL(url)
  }
  return (
    <section className="answer card" aria-label="Final answer">
      <div className="answer-head">
        <div className="answer-title">
          <span>
            <Sparkles size={20} />
          </span>
          <div>
            <span className="eyebrow">HDFC INSIGHT</span>
            <h2>Your answer</h2>
          </div>
        </div>
        <div className="answer-actions">
          <button onClick={copy} title="Copy answer">
            {copied ? <Check size={15} /> : <Copy size={15} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button onClick={download} title="Download answer">
            <ArrowDownToLine size={15} />
            Export
          </button>
        </div>
      </div>
      <div className="answer-question">{run.question}</div>
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{run.answer}</ReactMarkdown>
      </div>
      {copyError && (
        <p className="inline-error">
          Clipboard access is unavailable. Use Export to save your answer.
        </p>
      )}
      <div className="answer-footer">
        <ShieldCheck size={14} />
        <span>Analysis completed · {run.workspace}</span>
        <span>
          {run.finished &&
            new Date(run.finished).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
        </span>
      </div>
    </section>
  )
}

export default function App() {
  const [tenants, setTenants] = useState<Tenant[]>([])
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [stages, setStages] = useState(DEFAULT_STAGES)
  const [tenantId, setTenantId] = useState('')
  const [workspaceId, setWorkspaceId] = useState('')
  const [question, setQuestion] = useState('')
  const [maxTokens, setMaxTokens] = useState(8192)
  const [advanced, setAdvanced] = useState(false)
  const [catalogError, setCatalogError] = useState('')
  const [health, setHealth] = useState<'checking' | 'online' | 'offline'>(
    'checking',
  )
  const [workspaceLoading, setWorkspaceLoading] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [runs, setRuns] = useState<Run[]>([])
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [page, setPage] = useState<'studio' | 'history' | 'catalog'>('studio')
  const [help, setHelp] = useState(false)
  const [menu, setMenu] = useState(false)
  const controller = useRef<AbortController | null>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const run = runs.find((item) => item.id === currentId) || null
  const running = runs.some((item) => item.status === 'running')
  const workspace = workspaces.find((item) => item.id === workspaceId)
  const tenant = tenants.find((item) => item.id === tenantId)

  useEffect(() => {
    const abort = new AbortController()
    getJSON<{ tenants: Tenant[]; stages: StageDefinition[] }>(
      '/api/tenants',
      abort.signal,
    )
      .then((data) => {
        setTenants(data.tenants)
        setStages(data.stages)
        setTenantId((old) =>
          data.tenants.some((t) => t.id === old)
            ? old
            : data.tenants[0]?.id || '',
        )
        setCatalogError('')
      })
      .catch((error) => {
        if (!abort.signal.aborted) setCatalogError(error.message)
      })
    return () => abort.abort()
  }, [refresh])
  useEffect(() => {
    const abort = new AbortController()
    const check = () =>
      getJSON('/api/health', abort.signal)
        .then(() => setHealth('online'))
        .catch(() => {
          if (!abort.signal.aborted) setHealth('offline')
        })
    void check()
    const timer = setInterval(check, 20000)
    return () => {
      clearInterval(timer)
      abort.abort()
    }
  }, [refresh])
  useEffect(() => {
    if (!tenantId) return
    const abort = new AbortController()
    setWorkspaceLoading(true)
    setWorkspaces([])
    setWorkspaceId('')
    getJSON<{ workspaces: Workspace[] }>(
      `/api/tenants/${encodeURIComponent(tenantId)}/workspaces`,
      abort.signal,
    )
      .then((data) => {
        setWorkspaces(data.workspaces)
        setWorkspaceId(data.workspaces[0]?.id || '')
        setCatalogError('')
      })
      .catch((error) => {
        if (!abort.signal.aborted) setCatalogError(error.message)
      })
      .finally(() => {
        if (!abort.signal.aborted) setWorkspaceLoading(false)
      })
    return () => abort.abort()
  }, [tenantId, refresh])
  useEffect(() => () => controller.current?.abort(), [])
  useEffect(() => {
    if (!help) return
    const previous = document.activeElement as HTMLElement | null
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    document.querySelector<HTMLButtonElement>('.modal-close')?.focus()
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setHelp(false)
      if (event.key === 'Tab') {
        const buttons =
          document.querySelectorAll<HTMLButtonElement>('.help-modal button')
        const first = buttons[0],
          last = buttons[buttons.length - 1]
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault()
          last?.focus()
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault()
          first?.focus()
        }
      }
    }
    document.addEventListener('keydown', keyboard)
    return () => {
      document.removeEventListener('keydown', keyboard)
      document.body.style.overflow = overflow
      previous?.focus()
    }
  }, [help])

  const updateRun = (id: string, update: (value: Run) => Run) =>
    setRuns((all) =>
      all.map((value) => (value.id === id ? update(value) : value)),
    )
  const receive = (id: string, event: ProgressEvent) =>
    updateRun(id, (value) => {
      if (event.kind === 'ready')
        return {
          ...value,
          stages: (event.stages || stages).map((stage) => ({
            ...stage,
            status: 'pending',
          })),
        }
      if (event.kind === 'stage')
        return {
          ...value,
          stages: value.stages.map((stage) =>
            stage.id === event.stage
              ? {
                  ...stage,
                  status: event.status || stage.status,
                  duration_seconds: event.duration_seconds,
                  outcome: event.outcome,
                }
              : stage,
          ),
        }
      if (event.kind === 'activity')
        return {
          ...value,
          activity: [
            ...value.activity,
            {
              stage: event.stage || 'execute',
              message: event.message || '',
              timestamp: event.timestamp,
            } as Activity,
          ],
        }
      if (event.kind === 'result')
        return {
          ...value,
          status: 'completed',
          answer: event.answer || '',
          finished: Date.now(),
        }
      return value
    })
  const submit = async (event?: FormEvent) => {
    event?.preventDefault()
    if (
      !question.trim() ||
      !workspace ||
      !tenant ||
      running ||
      health === 'offline'
    )
      return
    const id = crypto.randomUUID(),
      abort = new AbortController()
    controller.current = abort
    const fresh: Run = {
      id,
      question: question.trim(),
      tenant: tenant.name,
      workspace: workspace.name,
      started: Date.now(),
      status: 'running',
      activity: [],
      stages: stages.map((stage) => ({ ...stage, status: 'pending' })),
    }
    setRuns((all) => [fresh, ...all].slice(0, 20))
    setCurrentId(id)
    setPage('studio')
    try {
      const response = await fetch('/api/questions/stream', {
        method: 'POST',
        signal: abort.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: tenantId,
          workspace_id: workspaceId,
          question: fresh.question,
          max_tokens: maxTokens,
        }),
      })
      await consumeEvents(response, (event) => receive(id, event))
    } catch (error) {
      const cancelled = abort.signal.aborted
      updateRun(id, (value) =>
        value.status === 'completed'
          ? value
          : {
              ...value,
              status: cancelled ? 'cancelled' : 'failed',
              finished: Date.now(),
              error: cancelled
                ? undefined
                : error instanceof Error
                  ? error.message
                  : 'Analysis could not complete.',
              stages: value.stages.map((stage) =>
                stage.status === 'running'
                  ? { ...stage, status: cancelled ? 'cancelled' : 'failed' }
                  : stage,
              ),
            },
      )
    } finally {
      if (controller.current === abort) controller.current = null
    }
  }
  const reset = () => {
    setCurrentId(null)
    setQuestion('')
    setPage('studio')
    setMenu(false)
    textarea.current?.focus()
  }
  const chooseRun = (id: string) => {
    setCurrentId(id)
    setPage('studio')
    setMenu(false)
  }
  const suggestions =
    workspace?.domain === 'smartwealth'
      ? [
          {
            icon: TrendingUp,
            label: 'Holdings overview',
            text: 'How is holding value distributed across AMCs?',
            question: 'What is the total current holding value by AMC?',
          },
          {
            icon: BarChart3,
            label: 'Cash flow trends',
            text: 'How did monthly net inflows vary?',
            question:
              'How variable were completed monthly net inflows from January through June 2026? Show all six monthly totals, the average, minimum and maximum.',
          },
          {
            icon: Users,
            label: 'Review groups',
            text: 'Create balanced AMC review groups.',
            question:
              'Split AMCs into three review groups with roughly equal total holding value. Show members and totals.',
          },
        ]
      : [
          {
            icon: TrendingUp,
            label: 'Portfolio performance',
            text: 'Find what’s driving profitability.',
            question:
              'Which card variants are driving portfolio profitability?',
          },
          {
            icon: ShieldCheck,
            label: 'Risk outlook',
            text: 'Understand potential credit losses.',
            question:
              "What could next month's credit losses be? Show expected loss and a loss level exceeded in only 5% of scenarios.",
          },
          {
            icon: Users,
            label: 'Customer insights',
            text: 'Discover meaningful customer segments.',
            question:
              'Group customers into three segments with similar spending, profitability and operating cost per card. Show members and typical values.',
          },
        ]

  return (
    <div className="app-shell">
      {menu && (
        <button
          className="sidebar-overlay"
          aria-label="Close navigation"
          onClick={() => setMenu(false)}
        />
      )}
      <aside className={`sidebar ${menu ? 'open' : ''}`}>
        <div className="sidebar-brand">
          <Logo />
        </div>
        <button className="new-analysis" onClick={reset} disabled={running}>
          <Plus size={17} />
          New analysis<kbd>↗</kbd>
        </button>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {(
            [
              ['studio', Sparkles, 'Analysis studio'],
              ['history', History, 'Analysis history'],
              ['catalog', Database, 'Data catalog'],
            ] as const
          ).map(([id, Icon, label]) => (
            <button
              key={id}
              className={`nav-item ${page === id ? 'active' : ''}`}
              onClick={() => {
                setPage(id)
                setMenu(false)
              }}
            >
              <Icon size={18} />
              <span>{label}</span>
              {id === 'studio' && <ChevronRight size={14} />}
              {id === 'history' && runs.length > 0 && (
                <span className="nav-count">{runs.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="recent-label">RECENT ANALYSES</div>
        <div className="recent-list">
          {runs.length === 0 ? (
            <div className="no-recent">
              Your analysis history
              <br />
              will appear here.
            </div>
          ) : (
            runs.slice(0, 4).map((item) => (
              <button
                key={item.id}
                onClick={() => chooseRun(item.id)}
                className={item.id === currentId ? 'selected' : ''}
              >
                <MessageSquareText size={14} />
                <span>{item.question}</span>
                {item.status === 'running' && <i className="live-dot" />}
              </button>
            ))
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <div className="note-symbol">
              <Waypoints size={23} />
            </div>
            <strong>Insight with context.</strong>
            <p>
              Your question. Your workspace.
              <br />
              Every step of the analysis.
            </p>
            <button onClick={() => setHelp(true)}>
              How it works <ArrowUpRight size={13} />
            </button>
          </div>
          <div className="profile">
            <div className="avatar">WA</div>
            <div>
              <strong>Workspace analyst</strong>
              <span>Local workspace</span>
            </div>
            <span className="profile-dot" />
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="mobile-menu icon-button"
              onClick={() => setMenu(true)}
              aria-label="Open navigation"
            >
              <Menu size={20} />
            </button>
            <Building2 size={16} />
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>
              {page === 'studio'
                ? 'Analysis studio'
                : page === 'history'
                  ? 'Analysis history'
                  : 'Data catalog'}
            </strong>
          </div>
          <div className="topbar-right">
            <span className={`service-state ${health}`}>
              <i />
              {health === 'online'
                ? 'Service online'
                : health === 'offline'
                  ? 'Service offline'
                  : 'Connecting'}
            </span>
            <span className="topbar-divider" />
            <button
              className="help-button"
              aria-label="Help & guide"
              onClick={() => setHelp(true)}
            >
              <CircleHelp size={17} />
              <span>Help & guide</span>
            </button>
          </div>
        </header>
        <main>
          {page === 'studio' && (
            <>
              <div className="page-heading">
                <div className="eyebrow">
                  <span className="red-rule" />
                  YOUR INTELLIGENCE WORKSPACE
                </div>
                <h1>
                  A better perspective starts
                  <br className="hero-break" /> with the right question.
                </h1>
                <p>
                  Explore your data with clarity. Follow every step from
                  question to insight.
                </p>
              </div>
              {(catalogError || health === 'offline') && (
                <div className="connection-error" role="alert">
                  <TriangleAlert size={18} />
                  <div>
                    <strong>
                      We couldn’t connect to the analysis service.
                    </strong>
                    <p>
                      {catalogError ||
                        'Check that the Python API is running, then reconnect.'}
                    </p>
                  </div>
                  <button onClick={() => setRefresh((v) => v + 1)}>
                    Reconnect
                  </button>
                </div>
              )}
              <div className="context-bar">
                <div className="context-select">
                  <label htmlFor="tenant">TENANT</label>
                  <div>
                    <Building2 size={17} />
                    <select
                      id="tenant"
                      value={tenantId}
                      onChange={(event) => setTenantId(event.target.value)}
                      disabled={running || !tenants.length}
                    >
                      <option value="" disabled>
                        Select a tenant
                      </option>
                      {tenants.map((item) => (
                        <option value={item.id} key={item.id}>
                          {item.name}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={14} />
                  </div>
                </div>
                <span className="context-divider" />
                <div className="context-select workspace-select">
                  <label htmlFor="workspace">WORKSPACE</label>
                  <div>
                    <Layers3 size={17} />
                    <select
                      id="workspace"
                      value={workspaceId}
                      onChange={(event) => setWorkspaceId(event.target.value)}
                      disabled={
                        running || workspaceLoading || !workspaces.length
                      }
                    >
                      <option value="" disabled>
                        {workspaceLoading
                          ? 'Loading workspaces…'
                          : 'Select a workspace'}
                      </option>
                      {workspaces.map((item) => (
                        <option value={item.id} key={item.id}>
                          {item.name}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={14} />
                  </div>
                </div>
                <span
                  className={`workspace-ready ${workspace && health === 'online' ? '' : 'not-ready'}`}
                >
                  <span />
                  {workspace && health === 'online'
                    ? `${workspace.table_count} tables in scope`
                    : 'Awaiting connection'}
                </span>
              </div>
              <form className="composer card" onSubmit={submit}>
                <div className="composer-label">
                  <span>
                    <Sparkles size={16} />
                    Ask your data
                  </span>
                  <span className="composer-domain">
                    {workspace?.domain === 'smartwealth'
                      ? 'WEALTH ANALYTICS'
                      : 'PORTFOLIO INTELLIGENCE'}
                  </span>
                </div>
                <label className="sr-only" htmlFor="question">
                  Your analysis question
                </label>
                <textarea
                  id="question"
                  ref={textarea}
                  placeholder="What would you like to understand about your portfolio?"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  disabled={running}
                  maxLength={8000}
                  onKeyDown={(event) => {
                    if (
                      (event.metaKey || event.ctrlKey) &&
                      event.key === 'Enter'
                    ) {
                      event.preventDefault()
                      void submit()
                    }
                  }}
                />
                <div className="composer-bottom">
                  <div className="composer-hint">
                    <ShieldCheck size={14} />
                    <span>Grounded in your selected workspace</span>
                    <button
                      type="button"
                      className={`settings-button ${advanced ? 'active' : ''}`}
                      onClick={() => setAdvanced(!advanced)}
                      aria-label="Analysis settings"
                      aria-expanded={advanced}
                    >
                      <Settings2 size={15} />
                    </button>
                  </div>
                  <div className="run-actions">
                    <span className="keyboard-hint">
                      <kbd>⌘</kbd>
                      <kbd>↵</kbd>
                    </span>
                    {running && (
                      <button
                        className="stop-button"
                        type="button"
                        onClick={() => controller.current?.abort()}
                      >
                        <Square size={12} />
                        Stop
                      </button>
                    )}
                    <button
                      className="run-button"
                      disabled={
                        running ||
                        !question.trim() ||
                        !workspace ||
                        health !== 'online'
                      }
                      type="submit"
                    >
                      {running ? (
                        <>
                          <LoaderCircle size={16} className="spin" />
                          Analyzing
                        </>
                      ) : (
                        <>
                          Run analysis
                          <ArrowUpRight size={17} />
                        </>
                      )}
                    </button>
                  </div>
                </div>
                {advanced && (
                  <div className="advanced-settings">
                    <label htmlFor="budget">
                      Model output budget{' '}
                      <span>Maximum tokens per model response</span>
                    </label>
                    <input
                      id="budget"
                      type="number"
                      min="1"
                      max="32768"
                      value={maxTokens}
                      disabled={running}
                      onChange={(event) =>
                        setMaxTokens(
                          Math.min(
                            32768,
                            Math.max(1, Number(event.target.value) || 1),
                          ),
                        )
                      }
                    />
                  </div>
                )}
              </form>
              {!run && (
                <div className="suggestions-section">
                  <div className="suggestions-label">
                    <span>NEED A STARTING POINT?</span>
                    <span>Try a question</span>
                  </div>
                  <div className="suggestions">
                    {suggestions.map(({ icon: Icon, ...item }, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setQuestion(item.question)
                          textarea.current?.focus()
                        }}
                        disabled={running}
                      >
                        <span className={`suggestion-icon color-${i}`}>
                          <Icon size={18} />
                        </span>
                        <div>
                          <strong>{item.label}</strong>
                          <span>{item.text}</span>
                        </div>
                        <ArrowUpRight size={15} />
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {run?.error && (
                <div className="run-error" role="alert">
                  <TriangleAlert size={19} />
                  <div>
                    <strong>Analysis needs attention</strong>
                    <p>{run.error}</p>
                    <span>
                      Completed stage outcomes remain available below. Update
                      the question or run it again.
                    </span>
                  </div>
                </div>
              )}
              {run?.status === 'cancelled' && (
                <div className="stopped-note">
                  <Square size={14} />
                  Analysis stopped. Completed stage outcomes are still
                  available.
                </div>
              )}
              <Journey run={run} stages={stages} />
              {run?.status === 'completed' && run.answer && (
                <Answer run={run} />
              )}
              {!run && (
                <div className="trust-strip">
                  <span>
                    <Network size={15} />
                    Connected to your context
                  </span>
                  <span>
                    <Waypoints size={15} />
                    Every stage, visible
                  </span>
                  <span>
                    <ShieldCheck size={15} />
                    Evidence before answers
                  </span>
                </div>
              )}
            </>
          )}
          {page === 'history' && (
            <>
              <div className="page-heading">
                <div className="eyebrow">YOUR WORKSPACE ACTIVITY</div>
                <h1>Analysis history</h1>
                <p>
                  Revisit the questions, stage outcomes and answers from this
                  session.
                </p>
              </div>
              {runs.length === 0 ? (
                <div className="empty-page card">
                  <History size={36} />
                  <h2>A fresh perspective awaits.</h2>
                  <p>Run your first analysis to start building your history.</p>
                  <button className="run-button" onClick={reset}>
                    Start an analysis
                    <ArrowRight size={16} />
                  </button>
                </div>
              ) : (
                <div className="history-list">
                  {runs.map((item) => (
                    <button
                      className="history-card card"
                      key={item.id}
                      onClick={() => chooseRun(item.id)}
                    >
                      <span className="history-icon">
                        <MessageSquareText size={20} />
                      </span>
                      <div>
                        <strong>{item.question}</strong>
                        <p>
                          {item.workspace} <span>·</span>{' '}
                          {new Date(item.started).toLocaleString()}
                        </p>
                      </div>
                      <span className={`status-pill ${item.status}`}>
                        {item.status === 'running'
                          ? 'In progress'
                          : item.status === 'completed'
                            ? 'Completed'
                            : item.status === 'cancelled'
                              ? 'Stopped'
                              : 'Needs attention'}
                      </span>
                      <ChevronRight size={17} />
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
          {page === 'catalog' && (
            <>
              <div className="page-heading">
                <div className="eyebrow">CONNECTED KNOWLEDGE</div>
                <h1>Your data, in context.</h1>
                <p>
                  Workspaces available in the selected tenant. Change your
                  tenant in Analysis studio.
                </p>
              </div>
              <div className="catalog-grid">
                {workspaces.map((item) => (
                  <div className="catalog-card card" key={item.id}>
                    <span className="catalog-icon">
                      <Database size={24} />
                    </span>
                    <span className="quiet-badge">WORKSPACE {item.id}</span>
                    <h2>{item.name}</h2>
                    <p>{item.domain.replaceAll('_', ' ')}</p>
                    <div className="catalog-metrics">
                      <span>
                        <strong>{item.table_count}</strong> tables
                      </span>
                      <span>
                        <strong>{item.sources.length}</strong> data source
                        {item.sources.length === 1 ? '' : 's'}
                      </span>
                    </div>
                    <div className="table-chips">
                      {item.sources.map((source) => (
                        <span key={source}>{source.replaceAll('_', ' ')}</span>
                      ))}
                    </div>
                    <button
                      className="text-button"
                      onClick={() => {
                        setWorkspaceId(item.id)
                        setPage('studio')
                      }}
                      disabled={running}
                    >
                      Explore this workspace
                      <ArrowUpRight size={15} />
                    </button>
                  </div>
                ))}
                {!workspaces.length && (
                  <div className="empty-page card">
                    <Database size={30} />
                    <h2>No workspace selected</h2>
                    <p>
                      Connect to the API and select a tenant in Analysis studio.
                    </p>
                    <button
                      className="text-button"
                      onClick={() => setPage('studio')}
                    >
                      Go to Analysis studio
                      <ArrowRight size={15} />
                    </button>
                  </div>
                )}
              </div>
            </>
          )}
          <footer className="page-footer">
            <span>
              HDFC INSIGHT <i /> INTELLIGENCE WITH CONTEXT
            </span>
            <span>Powered by your workspace data</span>
          </footer>
        </main>
      </div>
      {help && (
        <div className="modal-backdrop" onClick={() => setHelp(false)}>
          <section
            className="help-modal card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="help-title"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              className="modal-close icon-button"
              onClick={() => setHelp(false)}
              aria-label="Close guide"
            >
              <X size={19} />
            </button>
            <span className="modal-symbol">
              <BookOpen size={26} />
            </span>
            <span className="eyebrow">A LITTLE CONTEXT GOES A LONG WAY</span>
            <h2 id="help-title">From question to insight.</h2>
            <p>
              Select a tenant and workspace, then ask a business question in
              plain language. Mention a period, metric or comparison when it
              matters.
            </p>
            <ol>
              <li>
                <strong>Ask a clear question.</strong>
                <span>
                  Use a suggested question or write your own. Press ⌘/Ctrl +
                  Enter to begin.
                </span>
              </li>
              <li>
                <strong>Follow the journey.</strong>
                <span>
                  Click any stage to inspect its outcome. The Activity tab shows
                  SQL sub-stages, tools and correction attempts.
                </span>
              </li>
              <li>
                <strong>Review the answer.</strong>
                <span>
                  The final answer appears only after the analysis finishes.
                  Copy or export it, and revisit its evidence in this session’s
                  history.
                </span>
              </li>
            </ol>
            <div className="help-note">
              <Clock3 size={17} />
              <p>
                Complex questions may take several minutes. The timer shows
                elapsed time, not an estimated completion time. Use Stop to
                cancel.
              </p>
            </div>
            <button className="run-button" onClick={() => setHelp(false)}>
              Let’s explore
              <ArrowRight size={16} />
            </button>
          </section>
        </div>
      )}
    </div>
  )
}

export interface Tenant {
  id: string
  name: string
}
export interface Workspace {
  id: string
  name: string
  domain: string
  sources: string[]
  table_count: number
}
export interface StageDefinition {
  id: string
  name: string
  description: string
}
export type StageStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'skipped'
  | 'failed'
  | 'cancelled'
  | 'timeout'
export interface StageState extends StageDefinition {
  status: StageStatus
  duration_seconds?: number
  outcome?: Record<string, unknown>
}
export interface Activity {
  timestamp?: string
  stage: string
  message: string
}
export interface Run {
  id: string
  question: string
  workspace: string
  tenant: string
  started: number
  finished?: number
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  stages: StageState[]
  activity: Activity[]
  answer?: string
  error?: string
}
export interface ProgressEvent {
  kind: 'ready' | 'stage' | 'activity' | 'result' | 'error'
  run_id?: string
  stages?: StageDefinition[]
  stage?: string
  status?: StageStatus
  timestamp?: string
  duration_seconds?: number
  outcome?: Record<string, unknown>
  message?: string
  answer?: string
  error?: { type: string; message: string }
}

export async function getJSON<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, { signal })
  if (!response.ok)
    throw new Error(
      `Could not load workspace information (${response.status}).`,
    )
  return response.json() as Promise<T>
}

/** Parse chunked SSE with UTF-8, CRLF, comments and multi-line data support. */
export async function consumeEvents(
  response: Response,
  receive: (event: ProgressEvent) => void,
) {
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    const detail = error?.detail
    throw new Error(
      typeof detail === 'string'
        ? detail
        : 'The analysis request could not be accepted. Check your selections and try again.',
    )
  }
  if (!response.body)
    throw new Error('Your browser did not receive an analysis stream.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = '',
    terminal = false
  const dispatch = (frame: string) => {
    const data = frame
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, ''))
      .join('\n')
    if (!data) return
    const event = JSON.parse(data) as ProgressEvent
    if (event.kind === 'error')
      throw new Error(
        event.error?.message || 'The analysis could not complete.',
      )
    if (event.kind === 'result') terminal = true
    receive(event)
  }
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer = (buffer + decoder.decode(value, { stream: !done })).replace(
        /\r\n/g,
        '\n',
      )
      let boundary: number
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        dispatch(buffer.slice(0, boundary))
        buffer = buffer.slice(boundary + 2)
      }
      if (done) break
    }
    if (buffer.trim()) dispatch(buffer)
    if (!terminal)
      throw new Error(
        'The connection ended before the analysis finished. Please run it again.',
      )
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

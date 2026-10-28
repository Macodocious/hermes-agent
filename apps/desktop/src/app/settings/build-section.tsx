/**
 * Build panel.
 *
 * Packaging runs on the box, where the checkout and the Node toolchain live;
 * this section only starts that build and saves the artifacts it produces. The
 * app is a client, so it cannot package itself — the gateway does the work and
 * the main process writes the file the user picks.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'

interface BuildArtifact {
  bytes: number
  name: string
}

interface BuildStatus {
  artifacts: BuildArtifact[]
  error: string | null
  finished_at: number | null
  log: string
  returncode: number | null
  started_at: number | null
  state: 'idle' | 'running' | 'done' | 'failed'
}

const POLL_MS = 2000

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return `${Math.max(1, Math.round(bytes / 1024))} KB`
}

export function BuildSection() {
  const [status, setStatus] = useState<BuildStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState<string | null>(null)
  const timer = useRef<number | null>(null)

  const bridge = window.hermesDesktop?.build

  const refresh = useCallback(async () => {
    if (!bridge) {
      return null
    }

    const next = await bridge.status()

    setStatus(next)

    return next
  }, [bridge])

  useEffect(() => {
    void refresh().catch((cause: unknown) => setError(String(cause)))

    return () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current)
      }
    }
  }, [refresh])

  useEffect(() => {
    if (status?.state !== 'running') {
      return undefined
    }

    timer.current = window.setTimeout(() => {
      void refresh()
        .then(next => {
          if (next?.state === 'running') {
            setStatus(next)
          }
        })
        .catch((cause: unknown) => setError(String(cause)))
    }, POLL_MS)

    return () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current)
      }
    }
  }, [status, refresh])

  if (!bridge) {
    return null
  }

  const start = async () => {
    setBusy(true)
    setError(null)
    setSaved(null)

    try {
      await bridge.start()
      await refresh()
    } catch (cause) {
      setError(String(cause))
    } finally {
      setBusy(false)
    }
  }

  const save = async (name: string) => {
    setError(null)

    try {
      const result = await bridge.save({ name })

      if (result.saved) {
        setSaved(result.saved)
      }
    } catch (cause) {
      setError(String(cause))
    }
  }

  const running = status?.state === 'running'

  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <h3 className="text-sm font-medium">Build installer</h3>
        <p className="text-xs text-(--ui-text-secondary)">
          Runs the build on the box and writes the portable .exe plus a .zip of the unpacked app. This machine only
          downloads the result — it has no Node toolchain.
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Button disabled={busy || running} onClick={() => void start()} size="sm" variant="secondary">
          {running ? 'Building…' : 'Build now'}
        </Button>
        <span className="text-xs text-(--ui-text-tertiary)">
          {status ? `state: ${status.state}` : 'state: unknown'}
          {status?.error ? ` — ${status.error}` : ''}
        </span>
      </div>
      {saved ? <p className="text-xs text-(--ui-text-secondary)">Saved to {saved}</p> : null}
      {error ? <p className="text-xs text-(--ui-text-danger)">{error}</p> : null}
      {status && status.artifacts.length > 0 ? (
        <ul className="flex flex-col gap-1">
          {status.artifacts.map(artifact => (
            <li className="flex items-center justify-between gap-2 text-xs" key={artifact.name}>
              <span className="truncate font-mono">{artifact.name}</span>
              <span className="shrink-0 text-(--ui-text-tertiary)">{formatBytes(artifact.bytes)}</span>
              <Button onClick={() => void save(artifact.name)} size="xs" variant="ghost">
                Save
              </Button>
            </li>
          ))}
        </ul>
      ) : null}
      {status?.log ? (
        <pre className="max-h-40 overflow-auto rounded border border-(--ui-stroke-tertiary) p-2 font-mono text-[0.66rem] leading-relaxed text-(--ui-text-secondary)">
          {status.log}
        </pre>
      ) : null}
    </section>
  )
}

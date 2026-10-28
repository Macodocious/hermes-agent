/**
 * Apply loop: how an edit made on the box reaches the running app.
 *
 * The renderer is served from the box's dev server, so a renderer or config
 * edit repaints the running window without a rebuild. The main process cannot
 * hot-reload: it is built on the box, the built output is pulled to the PC, and
 * this module reports that pending build so the user can relaunch it.
 *
 * The app never relaunches itself. That mirrors the gateway rule — the agent
 * says a restart is needed, the user performs it.
 */
import { app, BrowserWindow, ipcMain } from 'electron'
import fs from 'node:fs'
import path from 'node:path'

export const APPLY_LOOP_CHANNELS = {
  dismiss: 'hermes:apply-loop:dismiss',
  pending: 'hermes:apply-loop:pending',
  restart: 'hermes:apply-loop:restart',
  status: 'hermes:apply-loop:status'
} as const

/** Marker written next to the app's user data once a main-process build has
 *  been pulled from the box. Its presence IS the "a restart is pending" fact. */
const MARKER_FILENAME = 'pending-main-build.json'

/** Poll rather than watch: fs.watch is unreliable on Windows across the shares
 *  a pulled build arrives over, and a missed event leaves the pill stale. */
const PENDING_BUILD_POLL_MS = 2048

/** How long to wait before re-dialling a dev server that dropped. */
const DEV_SERVER_RETRY_MS = 4096

export interface PendingMainBuild {
  builtAt: string
  changed: readonly string[]
  failed: boolean
  error: string | null
}

function markerPath(): string {
  return path.join(app.getPath('userData'), MARKER_FILENAME)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function readPendingBuild(): PendingMainBuild | null {
  let raw: string

  try {
    raw = fs.readFileSync(markerPath(), 'utf8')
  } catch (error) {
    // A missing marker is the normal state, not a failure worth reporting.
    if ((error as NodeJS.ErrnoException)?.code !== 'ENOENT') {
      console.warn(`[apply-loop] pending build marker unreadable: ${String(error)}`)
    }

    return null
  }

  let parsed: unknown

  try {
    parsed = JSON.parse(raw)
  } catch (error) {
    console.warn(`[apply-loop] pending build marker is not valid JSON: ${String(error)}`)

    return null
  }

  if (!isRecord(parsed) || typeof parsed.builtAt !== 'string') {
    console.warn('[apply-loop] pending build marker is missing builtAt')

    return null
  }

  const changed = Array.isArray(parsed.changed)
    ? parsed.changed.filter((entry): entry is string => typeof entry === 'string')
    : []

  return {
    builtAt: parsed.builtAt,
    changed,
    failed: parsed.failed === true,
    error: typeof parsed.error === 'string' ? parsed.error : null
  }
}

function signatureOf(build: PendingMainBuild | null): string {
  return build ? `${build.builtAt}|${build.failed}|${build.changed.join(',')}` : ''
}

function broadcast(build: PendingMainBuild | null): void {
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) {
      win.webContents.send(APPLY_LOOP_CHANNELS.pending, build)
    }
  }
}

let lastSignature = ''

function pollPendingBuild(): void {
  const build = readPendingBuild()
  const signature = signatureOf(build)

  if (signature === lastSignature) {
    return
  }

  lastSignature = signature
  broadcast(build)
}

export function registerApplyLoopIpc(): void {
  ipcMain.handle(APPLY_LOOP_CHANNELS.status, () => readPendingBuild())

  ipcMain.handle(APPLY_LOOP_CHANNELS.dismiss, () => {
    try {
      fs.rmSync(markerPath(), { force: true })
    } catch (error) {
      console.warn(`[apply-loop] could not clear the pending build marker: ${String(error)}`)

      return readPendingBuild()
    }

    lastSignature = ''
    broadcast(null)

    return null
  })

  ipcMain.handle(APPLY_LOOP_CHANNELS.restart, () => {
    // Only an explicit click reaches here: the running process replaces itself
    // with the build that was pulled.
    app.relaunch({ args: process.argv.slice(1) })
    app.exit(0)

    return { restarting: true }
  })

  lastSignature = signatureOf(readPendingBuild())
  setInterval(pollPendingBuild, PENDING_BUILD_POLL_MS).unref()
}

/**
 * Keep the window usable when the dev server it was loaded from goes away.
 * A live window keeps painting from the renderer it already loaded; this only
 * governs the re-dial, so a restart during a build leaves a window that comes
 * back on its own instead of an error page.
 */
export function wireDevServerResilience(win: BrowserWindow, devServer: string | undefined): void {
  if (!devServer) {
    return
  }

  win.webContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedUrl, isMainFrame) => {
    if (!isMainFrame || !validatedUrl.startsWith(devServer)) {
      return
    }

    console.warn(`[apply-loop] dev server unreachable (${errorCode} ${errorDescription}); retrying`)

    setTimeout(() => {
      if (win.isDestroyed()) {
        return
      }

      win.loadURL(devServer).catch(error => {
        console.warn(`[apply-loop] dev server retry failed: ${String(error)}`)
      })
    }, DEV_SERVER_RETRY_MS).unref()
  })
}

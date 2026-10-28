/**
 * Packaging, driven from the desktop app.
 *
 * The app is a client; the checkout and the Node toolchain live on the box, so
 * the build itself runs there behind the gateway's authenticated plugin route.
 * This module only starts it, polls it, and writes the artifact the user asks
 * to save. No command, path or argument is accepted from the renderer beyond an
 * artifact name that must match the release naming pattern.
 */
import { dialog, ipcMain } from 'electron'
import fs from 'node:fs/promises'

export const BUILD_CHANNELS = {
  save: 'hermes:build:save',
  start: 'hermes:build:start',
  status: 'hermes:build:status'
} as const

const PLUGIN_BASE = '/api/plugins/atlas_builder'
const ARTIFACT_NAME = /^[A-Za-z0-9._-]+\.(exe|zip)$/

export interface BuildConnection {
  baseUrl: string
  token: string
}

export type BuildConnectionResolver = (profile?: string) => Promise<BuildConnection>

export interface BuildArtifact {
  bytes: number
  name: string
}

export interface BuildStatus {
  artifacts: BuildArtifact[]
  error: string | null
  finished_at: number | null
  log: string
  returncode: number | null
  started_at: number | null
  state: 'idle' | 'running' | 'done' | 'failed'
}

async function request(
  connection: BuildConnection,
  route: string,
  method: 'GET' | 'POST'
): Promise<Response> {
  const response = await fetch(`${connection.baseUrl}${PLUGIN_BASE}${route}`, {
    headers: { 'X-Hermes-Session-Token': connection.token },
    method
  })

  if (!response.ok) {
    const detail = await response.text()

    throw new Error(`build request failed: ${response.status} ${detail}`)
  }

  return response
}

export function registerBuildIpc(resolveConnection: BuildConnectionResolver): void {
  ipcMain.handle(BUILD_CHANNELS.start, async (_event, profile?: string) => {
    const connection = await resolveConnection(profile)
    const response = await request(connection, '/build', 'POST')

    return (await response.json()) as { started_at: number | null; state: string }
  })

  ipcMain.handle(BUILD_CHANNELS.status, async (_event, profile?: string) => {
    const connection = await resolveConnection(profile)
    const response = await request(connection, '/build/status', 'GET')

    return (await response.json()) as BuildStatus
  })

  ipcMain.handle(BUILD_CHANNELS.save, async (_event, payload: { name: string; profile?: string }) => {
    const name = String(payload?.name ?? '')

    if (!ARTIFACT_NAME.test(name)) {
      throw new Error('invalid artifact name')
    }

    const chosen = await dialog.showSaveDialog({ defaultPath: name })

    if (chosen.canceled || !chosen.filePath) {
      return { bytes: 0, saved: null }
    }

    const connection = await resolveConnection(payload.profile)
    const response = await request(connection, `/build/artifact/${encodeURIComponent(name)}`, 'GET')
    const bytes = Buffer.from(await response.arrayBuffer())

    await fs.writeFile(chosen.filePath, bytes)

    return { bytes: bytes.byteLength, saved: chosen.filePath }
  })
}

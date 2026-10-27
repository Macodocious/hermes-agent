/**
 * Workspace writes to the box.
 *
 * Saves, creates and deletes all leave the app here rather than from the
 * renderer, so the pinned checkout can be checked on the resolved real path
 * before any request is formed. The gateway's managed-files policy stays the
 * hard boundary behind this refusal; both must agree.
 *
 * Endpoints used, unchanged:
 *   POST   /api/files/upload   { path, data_url, overwrite }
 *   POST   /api/files/mkdir    { path }
 *   DELETE /api/files          { path }
 */
import { ipcMain } from 'electron'
import os from 'node:os'
import path from 'node:path'

import { assertWritablePath, resolveWorkspaceRoot } from './workspace-root'

export const WORKSPACE_CHANNELS = {
  mkdir: 'hermes:workspace:mkdir',
  remove: 'hermes:workspace:remove',
  write: 'hermes:workspace:write'
} as const

export interface WorkspaceConnection {
  baseUrl: string
  token: string
}

export type WorkspaceConnectionResolver = (profile?: string) => Promise<WorkspaceConnection>

interface WorkspaceWriteRequest {
  content: string
  path: string
  profile?: string
  root: string
}

interface WorkspacePathRequest {
  path: string
  profile?: string
  root: string
}

function requireStrings(value: unknown, keys: readonly string[]): Record<string, string> {
  if (typeof value !== 'object' || value === null) {
    throw new Error('workspace request must be an object')
  }

  const record = value as Record<string, unknown>
  const checked: Record<string, string> = {}

  for (const key of keys) {
    const entry = record[key]

    if (typeof entry !== 'string' || entry.trim() === '') {
      throw new Error(`workspace request is missing ${key}`)
    }

    checked[key] = entry
  }

  return checked
}

async function postJson(connection: WorkspaceConnection, route: string, body: unknown): Promise<unknown> {
  const response = await fetch(`${connection.baseUrl}${route}`, {
    body: JSON.stringify(body),
    headers: {
      'content-type': 'application/json',
      'X-Hermes-Session-Token': connection.token
    },
    method: 'POST'
  })

  if (!response.ok) {
    throw new Error(`workspace request failed: ${response.status} ${await response.text()}`)
  }

  return response.json()
}

function toDataUrl(content: string): string {
  return `data:text/plain;charset=utf-8;base64,${Buffer.from(content, 'utf8').toString('base64')}`
}

export function registerWorkspaceFilesIpc(resolveConnection: WorkspaceConnectionResolver): void {
  ipcMain.handle(WORKSPACE_CHANNELS.write, async (_event, request: WorkspaceWriteRequest) => {
    const fields = requireStrings(request, ['root', 'path'])
    const content = typeof request?.content === 'string' ? request.content : null

    if (content === null) {
      throw new Error('workspace request is missing content')
    }

    // Refused here, before a request exists to send.
    const pinnedRoot = resolveWorkspaceRoot(fields.root, os.homedir())
    const target = assertWritablePath(pinnedRoot, fields.path)
    const connection = await resolveConnection(request.profile)
    const result = (await postJson(connection, '/api/files/upload', {
      data_url: toDataUrl(content),
      overwrite: true,
      path: target
    })) as { entry?: unknown }

    return { entry: result?.entry ?? null, path: target }
  })

  ipcMain.handle(WORKSPACE_CHANNELS.mkdir, async (_event, request: WorkspacePathRequest) => {
    const fields = requireStrings(request, ['root', 'path'])
    const pinnedRoot = resolveWorkspaceRoot(fields.root, os.homedir())
    const target = assertWritablePath(pinnedRoot, fields.path)
    const connection = await resolveConnection(request.profile)
    const result = (await postJson(connection, '/api/files/mkdir', { path: target })) as { entry?: unknown }

    return { entry: result?.entry ?? null, path: target }
  })

  ipcMain.handle(WORKSPACE_CHANNELS.remove, async (_event, request: WorkspacePathRequest) => {
    const fields = requireStrings(request, ['root', 'path'])
    const pinnedRoot = resolveWorkspaceRoot(fields.root, os.homedir())
    const target = assertWritablePath(pinnedRoot, fields.path)
    const connection = await resolveConnection(request.profile)
    const response = await fetch(`${connection.baseUrl}/api/files`, {
      body: JSON.stringify({ path: target }),
      headers: {
        'content-type': 'application/json',
        'X-Hermes-Session-Token': connection.token
      },
      method: 'DELETE'
    })

    if (!response.ok) {
      throw new Error(`workspace delete failed: ${response.status} ${await response.text()}`)
    }

    return { path: target }
  })
}

/** Rename has no gateway route: copy through the upload endpoint, then delete. */
export async function renameThrough(
  write: (target: string, content: string) => Promise<unknown>,
  remove: (target: string) => Promise<unknown>,
  from: string,
  to: string,
  content: string
): Promise<{ from: string; to: string }> {
  await write(to, content)
  await remove(from)

  return { from, to }
}

export function joinInside(root: string, relative: string): string {
  return path.join(root, relative)
}

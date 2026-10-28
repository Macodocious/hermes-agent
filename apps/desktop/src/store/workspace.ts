/**
 * Workspace writes, as the renderer sees them.
 *
 * Every call lands in the main process, which checks the target against the
 * pinned checkout on its resolved real path before forming a request. The
 * renderer never posts a write itself — that is the whole point of the seam.
 */

import { atom } from 'nanostores'

import type { WorkspacePathRequest, WorkspaceWriteRequest } from '@/global'
import { persistString, storedString } from '@/lib/storage'
import { $connection } from '@/store/session'

/** Where the checkout lives on the box. Editable per connection in settings. */
export const DEFAULT_WORKSPACE_ROOT = '~/.hermes/repos/hermes-agent'

export const $workspaceRoot = atom<string>(DEFAULT_WORKSPACE_ROOT)

/** The checkout resolved in the box's own path space (an absolute POSIX path).
 *  The configured value may be `~`-relative, and only the main process can
 *  expand that against the box's home rather than this PC's, so the explorer
 *  roots at this value and never at the client-side string. */
export const $resolvedWorkspaceRoot = atom<string>('')

export const loadResolvedWorkspaceRoot = async (): Promise<string> => {
  try {
    const resolved = (await window.hermesDesktop?.workspace?.root?.()) ?? ''
    const value = typeof resolved === 'string' ? resolved : ''

    $resolvedWorkspaceRoot.set(value)

    return value
  } catch (error) {
    console.warn(`[workspace] could not resolve the box checkout: ${String(error)}`)
    $resolvedWorkspaceRoot.set('')

    return ''
  }
}

// Persisted per connection: the same app pointed at two boxes can hold two
// different checkouts, and neither may bleed into the other.
const storageKey = () => `hermes:workspace-root:${$connection.get()?.baseUrl ?? 'local'}`

export const loadWorkspaceRoot = (): string => {
  const stored = storedString(storageKey()) || DEFAULT_WORKSPACE_ROOT
  $workspaceRoot.set(stored)

  return stored
}

export const setWorkspaceRoot = (root: string): void => {
  const trimmed = root.trim() || DEFAULT_WORKSPACE_ROOT
  $workspaceRoot.set(trimmed)
  persistString(storageKey(), trimmed)
}

const workspaceApi = () => window.hermesDesktop?.workspace

function requireApi() {
  const api = workspaceApi()

  if (!api) {
    throw new Error('Workspace writes need the desktop app.')
  }

  return api
}

export const saveWorkspaceFile = async (path: string, content: string): Promise<string> => {
  const request: WorkspaceWriteRequest = { content, path, root: $workspaceRoot.get() }
  const result = await requireApi().write(request)

  return result.path
}

export const createWorkspaceDirectory = async (path: string): Promise<string> => {
  const request: WorkspacePathRequest = { path, root: $workspaceRoot.get() }
  const result = await requireApi().mkdir(request)

  return result.path
}

export const deleteWorkspacePath = async (path: string): Promise<string> => {
  const request: WorkspacePathRequest = { path, root: $workspaceRoot.get() }
  const result = await requireApi().remove(request)

  return result.path
}

/** Create an empty file in a directory on the box, then hand it to the inline
 *  rename editor so the name is chosen the same way a rename is. */
export const createWorkspaceFile = async (directory: string, name = 'untitled'): Promise<string> => {
  const target = `${directory.replace(/\/+$/, '')}/${name}`

  await saveWorkspaceFile(target, '')

  return target
}

function decodeBase64Utf8(encoded: string): string {
  const binary = atob(encoded)
  const bytes = Uint8Array.from(binary, char => char.charCodeAt(0))

  return new TextDecoder().decode(bytes)
}

/** Read over the gateway. Reads are governed by the server's own policy; the
 *  pin only has to guard writes. */
export const readWorkspaceFile = async (path: string): Promise<string> => {
  const connection = await window.hermesDesktop.getConnection()
  const token = connection.token ? `&token=${encodeURIComponent(connection.token)}` : ''
  const response = await fetch(`${connection.baseUrl}/api/files/read?path=${encodeURIComponent(path)}${token}`)

  if (!response.ok) {
    throw new Error(`Could not read ${path}: ${response.status}`)
  }

  const body = (await response.json()) as { data_url?: string }
  const dataUrl = body.data_url ?? ''
  const comma = dataUrl.indexOf(',')

  return comma >= 0 ? decodeBase64Utf8(dataUrl.slice(comma + 1)) : ''
}

/**
 * Rename on the box. The gateway has no move route, so this is a copy through
 * the upload endpoint followed by a delete — which is also why a directory
 * cannot be renamed here: reading one as a file is refused by the server.
 */
export const renameWorkspacePath = async (path: string, newName: string): Promise<string> => {
  const separator = path.lastIndexOf('/')
  const target = `${path.slice(0, separator)}/${newName}`
  const content = await readWorkspaceFile(path)

  await saveWorkspaceFile(target, content)
  await deleteWorkspacePath(path)

  return target
}

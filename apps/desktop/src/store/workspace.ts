/**
 * Workspace writes, as the renderer sees them.
 *
 * Every call lands in the main process, which checks the target against the
 * pinned checkout on its resolved real path before forming a request. The
 * renderer never posts a write itself — that is the whole point of the seam.
 */

import { atom } from 'nanostores'

import type { WorkspacePathRequest, WorkspaceWriteRequest } from '@/global'

/** Where the checkout lives on the box. Editable per connection in settings. */
export const DEFAULT_WORKSPACE_ROOT = '~/.hermes/repos/hermes-agent'

export const $workspaceRoot = atom<string>(DEFAULT_WORKSPACE_ROOT)

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

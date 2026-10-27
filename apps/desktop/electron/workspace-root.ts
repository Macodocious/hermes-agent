/**
 * The write boundary for the workspace tree.
 *
 * Exactly one root is writable: the pinned Atlas checkout on the box. The check
 * runs on the resolved real path, so a symlink, a `..` segment or a crafted
 * name cannot smuggle a write outside it. The gateway's managed-files policy
 * stays the hard boundary behind this refusal; both must agree.
 */
import fs from 'node:fs'
import path from 'node:path'

/** Where the checkout lives on the box unless a connection says otherwise. */
export const DEFAULT_WORKSPACE_ROOT = '~/.hermes/repos/hermes-agent'

export class WorkspaceEscapeError extends Error {
  readonly candidate: string
  readonly root: string

  constructor(candidate: string, root: string) {
    super(`Refusing to write ${candidate}: it resolves outside the workspace root ${root}`)
    this.name = 'WorkspaceEscapeError'
    this.candidate = candidate
    this.root = root
  }
}

export function expandHome(configured: string, home: string): string {
  const trimmed = configured.trim()

  if (!trimmed) {
    return expandHome(DEFAULT_WORKSPACE_ROOT, home)
  }

  if (trimmed === '~') {
    return home
  }

  if (trimmed.startsWith('~/')) {
    return path.join(home, trimmed.slice(2))
  }

  return trimmed
}

/**
 * Resolve to a real path, tolerating a target that does not exist yet: a file
 * about to be created resolves through its parent, so the check still sees the
 * directory chain a symlink could have redirected.
 */
function realpathOrParent(target: string): string {
  try {
    return fs.realpathSync.native(target)
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code !== 'ENOENT') {
      throw error
    }

    const parent = path.dirname(target)

    if (parent === target) {
      return path.resolve(target)
    }

    return path.join(realpathOrParent(parent), path.basename(target))
  }
}

export function resolveWorkspaceRoot(configured: string, home: string): string {
  return path.resolve(expandHome(configured, home))
}

export function isInsideRoot(root: string, candidate: string): boolean {
  const resolvedRoot = realpathOrParent(path.resolve(root))
  const resolvedCandidate = realpathOrParent(path.resolve(candidate))

  return resolvedCandidate === resolvedRoot || resolvedCandidate.startsWith(resolvedRoot + path.sep)
}

/** Resolve a write target, or refuse it before any request is formed. */
export function assertWritablePath(root: string, candidate: string): string {
  const resolvedCandidate = realpathOrParent(path.resolve(candidate))

  if (!isInsideRoot(root, resolvedCandidate)) {
    throw new WorkspaceEscapeError(resolvedCandidate, path.resolve(root))
  }

  return resolvedCandidate
}

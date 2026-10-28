import ignore from 'ignore'

import type { HermesReadDirEntry, HermesReadDirResult } from '@/global'
import { desktopFsCacheKey, desktopGitRoot, readDesktopDir, readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { ALWAYS_EXCLUDED } from '@/lib/excluded-paths'

export type ProjectTreeEntry = HermesReadDirEntry

interface GitignoreRule {
  base: string
  ig: ReturnType<typeof ignore>
}

const gitRootCache = new Map<string, Promise<string | null>>()
const gitignoreCache = new Map<string, Promise<GitignoreRule | null>>()

function decodeDataUrl(dataUrl: string) {
  const match = dataUrl.match(/^data:[^,]*,(.*)$/)
  const data = match?.[1] || ''
  const isBase64 = dataUrl.slice(0, dataUrl.indexOf(',')).includes(';base64')

  if (!isBase64) {
    return decodeURIComponent(data)
  }

  const bytes = Uint8Array.from(atob(data), ch => ch.charCodeAt(0))

  return new TextDecoder().decode(bytes)
}

function clean(path: string) {
  return path.replace(/\\/g, '/').replace(/\/+$/, '') || '/'
}

/** Strict POSIX-style relative path; null if `child` is not inside `root`. */
function relativeTo(root: string, child: string) {
  const r = clean(root)
  const c = clean(child)

  if (c === r) {
    return ''
  }

  return c.startsWith(`${r}/`) ? c.slice(r.length + 1) : null
}

/** Repo-root → repo-root/a → repo-root/a/b → … for every dir between root and `dir`. */
function ancestorDirs(root: string, dir: string) {
  const r = clean(root)
  const rel = relativeTo(r, dir)

  if (rel === null || rel === '') {
    return [r]
  }

  const dirs = [r]
  let current = r

  for (const part of rel.split('/').filter(Boolean)) {
    current = `${current}/${part}`
    dirs.push(current)
  }

  return dirs
}

async function gitRootFor(start: string) {
  const key = `${desktopFsCacheKey()}:${clean(start)}`
  let cached = gitRootCache.get(key)

  if (!cached) {
    cached = desktopGitRoot(clean(start))
    gitRootCache.set(key, cached)
  }

  return cached
}

/** Read .gitignore at `dir` if it actually exists — never probe missing files. */
async function readGitignore(dir: string): Promise<GitignoreRule | null> {
  try {
    const listing = await readDesktopDir(dir)

    if (!listing.entries.some(e => e.name === '.gitignore' && !e.isDirectory)) {
      return null
    }

    const text = decodeDataUrl(await readDesktopFileDataUrl(`${dir}/.gitignore`))

    return { base: dir, ig: ignore().add(text) }
  } catch {
    return null
  }
}

async function gitignoreFor(dir: string) {
  const key = `${desktopFsCacheKey()}:${clean(dir)}`
  let cached = gitignoreCache.get(key)

  if (!cached) {
    cached = readGitignore(clean(dir))
    gitignoreCache.set(key, cached)
  }

  return cached
}

function ignoredBy(rules: GitignoreRule[], entry: HermesReadDirEntry) {
  return rules.some(rule => {
    const rel = relativeTo(rule.base, entry.path)

    if (rel === null || rel === '') {
      return false
    }

    return rule.ig.ignores(entry.isDirectory ? `${rel}/` : rel)
  })
}

async function filterIgnored(entries: HermesReadDirEntry[], rootPath: string, dirPath: string) {
  const root = await gitRootFor(rootPath)

  if (!root) {
    return entries
  }

  const rules = (await Promise.all(ancestorDirs(root, dirPath).map(gitignoreFor))).filter((r): r is GitignoreRule =>
    Boolean(r)
  )

  return rules.length > 0 ? entries.filter(entry => !ignoredBy(rules, entry)) : entries
}

/**
 * The box checkout, once resolved by the main process. Until it is set the tree
 * is purely local, which is the app's behaviour today.
 */
let boxRoot: string | null = null
let boxRootInitialized = false

export function setBoxRoot(root: string | null) {
  boxRoot = root ? clean(root) : null
  boxRootInitialized = true
}

function isUnderBox(dirPath: string): boolean {
  if (!boxRoot) {
    return false
  }

  const target = clean(dirPath)

  return target === boxRoot || target.startsWith(`${boxRoot}/`)
}

/** True for a path that lives on the box, which decides which reader and which
 *  write path a tree action uses. */
export function isBoxPath(target: string): boolean {
  return isUnderBox(target)
}

export async function readProjectDir(dirPath: string, rootPath = dirPath): Promise<HermesReadDirResult> {
  if (!window.hermesDesktop) {
    return { entries: [], error: 'no-bridge' }
  }

  // Files on the box are not on this PC: read them over the gateway. Everything
  // else is local, and stays read-only at the write boundary.
  if (!boxRootInitialized) {
    try {
      const resolved = await window.hermesDesktop.workspace?.root?.()
      setBoxRoot(typeof resolved === 'string' ? resolved : null)
    } catch (error) {
      console.warn(`[workspace] could not resolve the box root: ${String(error)}`)
      boxRootInitialized = true
    }
  }

  const result = isUnderBox(dirPath) ? await readGatewayDir(dirPath) : await readDesktopDir(dirPath)
  const entries = (result?.entries ?? []).filter(entry => !ALWAYS_EXCLUDED.has(entry.name))

  return { ...result, entries: await filterIgnored(entries, rootPath, dirPath) }
}

export function clearProjectDirCache(rootPath?: string) {
  if (!rootPath) {
    gitRootCache.clear()
    gitignoreCache.clear()

    return
  }

  const key = `${desktopFsCacheKey()}:${clean(rootPath)}`
  gitRootCache.delete(key)
  gitignoreCache.delete(key)
}

/**
 * The box's checkout is listed through the gateway, not the local bridge: those
 * files live on the server, so nothing on this PC can read the directory.
 * Shapes map 1:1 onto the local reader so the tree does not care which it got.
 */
export async function readGatewayDir(dirPath: string): Promise<HermesReadDirResult> {
  const bridge = window.hermesDesktop

  if (!bridge) {
    return { entries: [], error: 'no-bridge' }
  }

  const connection = await bridge.getConnection()

  if (!connection?.baseUrl) {
    return { entries: [], error: 'no-connection' }
  }

  const token = connection.token ? `&token=${encodeURIComponent(connection.token)}` : ''
  const response = await fetch(`${connection.baseUrl}/api/files?path=${encodeURIComponent(dirPath)}${token}`)

  if (!response.ok) {
    return { entries: [], error: `http-${response.status}` }
  }

  const body = (await response.json()) as {
    entries?: { is_directory?: boolean; name?: string; path?: string }[]
  }

  const entries = (body.entries ?? []).flatMap(entry =>
    typeof entry.path === 'string' && typeof entry.name === 'string'
      ? [{ isDirectory: entry.is_directory === true, name: entry.name, path: entry.path }]
      : []
  )

  return { entries }
}

/**
 * Open files as tabs.
 *
 * One tab per path. Content is held here so switching tabs never re-reads the
 * box, and so a tab can be dirty across switches. Saving goes through
 * `saveWorkspaceFile`, which refuses any path outside the pinned checkout.
 */
import { atom } from 'nanostores'

import { saveWorkspaceFile } from '@/store/workspace'

export interface EditorTab {
  content: string
  /** Content as last read or saved; the tab is dirty while they differ. */
  savedContent: string
  name: string
  path: string
}

export interface EditorTabsState {
  activePath: string | null
  error: string | null
  loading: boolean
  tabs: EditorTab[]
}

const emptyState: EditorTabsState = { activePath: null, error: null, loading: false, tabs: [] }

export const $editorTabs = atom<EditorTabsState>(emptyState)

export function fileNameOf(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean)

  return parts[parts.length - 1] ?? path
}

const update = (updater: (current: EditorTabsState) => EditorTabsState) => {
  $editorTabs.set(updater($editorTabs.get()))
}

export function openEditorTab(path: string, content: string) {
  update(current => {
    const existing = current.tabs.find(tab => tab.path === path)

    if (existing) {
      return { ...current, activePath: path }
    }

    return {
      ...current,
      activePath: path,
      error: null,
      tabs: [...current.tabs, { content, name: fileNameOf(path), path, savedContent: content }]
    }
  })
}

export function activateEditorTab(path: string) {
  update(current => ({ ...current, activePath: path }))
}

export function updateEditorTab(path: string, content: string) {
  update(current => ({
    ...current,
    tabs: current.tabs.map(tab => (tab.path === path ? { ...tab, content } : tab))
  }))
}

export function closeEditorTab(path: string) {
  update(current => {
    const index = current.tabs.findIndex(tab => tab.path === path)
    const tabs = current.tabs.filter(tab => tab.path !== path)

    if (tabs.length === 0) {
      return { ...current, activePath: null, tabs }
    }

    if (current.activePath !== path) {
      return { ...current, tabs }
    }

    const next = tabs[Math.min(index, tabs.length - 1)]

    return { ...current, activePath: next?.path ?? null, tabs }
  })
}

export function isEditorTabDirty(tab: EditorTab): boolean {
  return tab.content !== tab.savedContent
}

export function activeEditorTab(state: EditorTabsState): EditorTab | null {
  return state.tabs.find(tab => tab.path === state.activePath) ?? null
}

/** Marks the tab saved only after the write came back, so a failed save leaves
 *  the dot on the tab and the buffer untouched. */
export async function saveEditorTab(path: string): Promise<void> {
  const tab = $editorTabs.get().tabs.find(candidate => candidate.path === path)

  if (!tab) {
    return
  }

  update(current => ({ ...current, error: null, loading: true }))

  try {
    await saveWorkspaceFile(path, tab.content)
    update(current => ({
      ...current,
      loading: false,
      tabs: current.tabs.map(candidate => (candidate.path === path ? { ...candidate, savedContent: tab.content } : candidate))
    }))
  } catch (error) {
    console.error(`[editor] save failed for ${path}: ${String(error)}`)
    update(current => ({ ...current, error: String(error), loading: false }))
  }
}

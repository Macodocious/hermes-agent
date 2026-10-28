/**
 * The editor surface: one tab per open file, the active one rendered on the
 * existing CodeMirror component. Remounting by `key` is how the editor loads a
 * different document (its contract), which is also why content lives in the
 * tab store rather than in the editor.
 */
import { useStore } from '@nanostores/react'
import { useRef } from 'react'

import { type CodeEditorApi, CodeEditor } from '@/components/chat/code-editor'
import { PaneTab, PaneTabLabel } from '@/components/ui/pane-tab'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $editorTabs,
  activateEditorTab,
  activeEditorTab,
  closeEditorTab,
  isEditorTabDirty,
  saveEditorTab,
  updateEditorTab
} from '@/store/editor-tabs'

interface WorkspaceEditorProps {
  className?: string
}

export function WorkspaceEditor({ className }: WorkspaceEditorProps) {
  const { t } = useI18n()
  const state = useStore($editorTabs)
  const tab = activeEditorTab(state)
  const apiRef = useRef<CodeEditorApi | null>(null)

  if (!tab) {
    return null
  }

  return (
    <div className={cn('flex h-full min-h-0 w-full flex-col overflow-hidden bg-(--ui-editor-surface-background)', className)}>
      <div className="flex h-(--titlebar-height) shrink-0 overflow-x-auto bg-(--ui-sidebar-surface-background)">
        {state.tabs.map(candidate => (
          <PaneTab
            active={candidate.path === tab.path}
            dirty={isEditorTabDirty(candidate)}
            key={candidate.path}
            onClose={() => closeEditorTab(candidate.path)}
          >
            <PaneTabLabel
              aria-selected={candidate.path === tab.path}
              as="button"
              className="normal-case tracking-normal"
              onClick={() => activateEditorTab(candidate.path)}
              role="tab"
              title={candidate.path}
              type="button"
            >
              {candidate.name}
            </PaneTabLabel>
          </PaneTab>
        ))}
      </div>
      <div className="min-h-0 flex-1">
        <CodeEditor
          apiRef={apiRef}
          className="h-full"
          disabled={state.loading}
          filePath={tab.path}
          initialValue={tab.content}
          key={tab.path}
          onChange={value => updateEditorTab(tab.path, value)}
          onSave={() => void saveEditorTab(tab.path)}
        />
      </div>
      {state.error ? (
        <div className="shrink-0 border-t border-(--ui-stroke-tertiary) px-3 py-2 text-xs text-(--ui-text-danger)">
          {t.common.error}: {state.error}
        </div>
      ) : null}
    </div>
  )
}

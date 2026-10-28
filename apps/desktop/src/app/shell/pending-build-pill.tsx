import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $pendingMainBuild, dismissPendingBuild, restartToApplyBuild } from '@/store/apply-loop'

const PILL_CLASS = 'inline-flex h-full items-center gap-1.5 px-1.5 text-[0.6875rem]'
const ACTION_CLASS =
  'rounded px-1 py-0.5 text-[0.6875rem] text-(--ui-text-secondary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground'

/**
 * The apply loop's control. A main-process build has been pulled from the box
 * and is on disk; only a click replaces the running process, so this states
 * what is waiting and offers the relaunch — it never fires on its own. A
 * failed build reports at the same control and leaves the running build alone.
 */
export function PendingBuildPill() {
  const { t } = useI18n()
  const copy = t.pendingBuild
  const build = useStore($pendingMainBuild)

  if (!build) {
    return null
  }

  if (build.failed) {
    return (
      <div className={cn(PILL_CLASS, 'text-(--ui-text-tertiary)')} title={build.error ?? undefined}>
        <span aria-hidden className="size-1.5 rounded-full bg-current" />
        <span className="text-foreground">{copy.failed}</span>
        <span>{copy.failedHint}</span>
        <button className={ACTION_CLASS} onClick={() => void dismissPendingBuild()} type="button">
          {copy.dismiss}
        </button>
      </div>
    )
  }

  return (
    <div className={cn(PILL_CLASS, 'text-(--ui-text-tertiary)')}>
      <span aria-hidden className="size-1.5 rounded-full bg-current" />
      <span className="text-foreground">{copy.ready}</span>
      {build.changed.length > 0 ? <span>{copy.changed(build.changed.length)}</span> : null}
      <button className={ACTION_CLASS} onClick={() => void restartToApplyBuild()} type="button">
        {copy.restart}
      </button>
      <button className={ACTION_CLASS} onClick={() => void dismissPendingBuild()} type="button">
        {copy.dismiss}
      </button>
    </div>
  )
}

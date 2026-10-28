/**
 * Apply-loop store: the renderer's view of a main-process build that has been
 * pulled from the box and is waiting to be applied.
 *
 * The window never applies a build itself — it shows the control and the user
 * clicks it. The main process owns the marker file; this is a cache of it.
 */

import { atom } from 'nanostores'

import type { PendingMainBuild } from '@/global'

export const $pendingMainBuild = atom<PendingMainBuild | null>(null)

const applyLoopApi = () => window.hermesDesktop?.applyLoop

export const refreshPendingMainBuild = async (): Promise<void> => {
  const api = applyLoopApi()

  if (!api) {
    return
  }

  $pendingMainBuild.set(await api.status())
}

export const restartToApplyBuild = async (): Promise<void> => {
  await applyLoopApi()?.restart()
}

export const dismissPendingBuild = async (): Promise<void> => {
  await applyLoopApi()?.dismiss()
  $pendingMainBuild.set(null)
}

/** Load the current marker and follow later changes. Returns the unsubscribe. */
export const subscribeToPendingBuild = (): (() => void) => {
  const api = applyLoopApi()

  if (!api) {
    return () => {}
  }

  void refreshPendingMainBuild()

  return api.onPending(build => $pendingMainBuild.set(build))
}

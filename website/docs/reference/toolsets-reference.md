---
sidebar_position: 4
title: "Toolsets Reference"
description: "Reference for Hermes core, composite, platform, and dynamic toolsets"
---

# Toolsets Reference

Toolsets are named bundles of tools that control what the agent can do. They're the primary mechanism for configuring tool availability per platform, per session, or per task.

## How Toolsets Work

Every tool belongs to exactly one toolset. When you enable a toolset, all tools in that bundle become available to the agent. Toolsets come in three kinds:

- **Core** — A single logical group of related tools (e.g., `file` bundles `read_file`, `write_file`, `patch`, `search_files`)
- **Composite** — Combines multiple core toolsets for a common scenario (e.g., `debugging` bundles file, terminal, and web tools)
- **Platform** — A complete tool configuration for a specific deployment context (e.g., `hermes-cli` is the default for interactive CLI sessions)

## Configuring Toolsets

### Per-session (CLI)

```bash
hermes chat --toolsets web,file,terminal
hermes chat --toolsets debugging        # composite — expands to file + terminal + web
hermes chat --toolsets all              # everything
```

### Per-platform (config.yaml)

```yaml
toolsets:
  - hermes-cli          # default for CLI
  # - hermes-telegram   # override for Telegram gateway
```

### Interactive management

```bash
hermes tools                            # curses UI to enable/disable per platform
```

Or in-session:

```
/tools list
/tools disable browser
/tools enable homeassistant
```

## Core Toolsets

| Toolset | Tools | Purpose |
|---------|-------|---------|
| `browser` | `back_browser`, `browser_cdp`, `click_browser`, `browser_console`, `browser_dialog`, `get_browser_images`, `navigate_browser`, `press_browser`, `scroll_browser`, `snapshot_browser`, `type_browser`, `browser_vision`, `search_web` | Core browser automation. Includes `search_web` as a fallback for quick lookups. `browser_cdp` and `browser_dialog` are gated at runtime — registered only when a CDP endpoint is reachable at session start (via `/browser connect`, `browser.cdp_url` config, Browserbase, or Camofox). `browser_dialog` works together with the `pending_dialogs` and `frame_tree` fields that `snapshot_browser` adds when a CDP supervisor is attached. |
| `clarify` | `clarify` | Ask the user a question when the agent needs clarification. |
| `code_execution` | `execute_code` | Run Python scripts that call Hermes tools programmatically. |
| `coding` | composite (`file` + `terminal` + `search` + `web` + `skills` + `browser` + `todo` + `memory` + `search_session` + `clarify` + `code_execution` + `delegation` + `vision`) | Coding-focused bundle for software work: file editing, terminal, search, web docs, skills, browser, delegate, and code execution. |
| `cronjob` | `cronjob` | Schedule and manage recurring tasks. |
| `debugging` | composite (`file` + `terminal` + `web`) | Debug bundle — file, process/terminal, web extract/search. |
| `delegation` | `delegate_task` | Spawn isolated subagent instances for parallel work. |
| `discord` | `discord` | Core Discord text/embed/DM actions (gateway-only). Active on the `hermes-discord` toolset. |
| `discord_admin` | `discord_admin` | Discord moderation (bans, role changes, channel management). Active on the `hermes-discord` toolset; requires the bot to hold the relevant Discord permissions. |
| `feishu_doc` | `read_feishu_doc` | Read Feishu/Lark document content. Used by the Feishu document-comment intelligent-reply handler. |
| `feishu_drive` | `add_feishu_drive_comment`, `list_feishu_drive_comments`, `list_feishu_drive_comment_replies`, `reply_feishu_drive_comment` | Feishu/Lark drive comment operations. Scoped to the comment agent; not exposed on `hermes-cli` or other messaging toolsets. |
| `file` | `patch`, `read_file`, `search_files`, `write_file` | File reading, writing, searching, and editing. |
| `homeassistant` | `call_ha_service`, `get_ha_state`, `list_ha_entities`, `list_ha_services` | Smart home control via Home Assistant. Only available when `HASS_TOKEN` is set. |
| `use_computer` | `use_computer` | Background desktop control via cua-driver — does not steal cursor/focus. Works with any tool-capable model. macOS, Windows, and Linux; requires `cua-driver` on `$PATH`. |
| `context_engine` | (varies) | Runtime tools exposed by the active context-engine plugin (empty until a plugin populates it). |
| `image_gen` | `generate_image` | Text-to-image generation via FAL.ai (with opt-in OpenAI / xAI backends). |
| `video_gen` | `generate_video` | Text-to-video and image-to-video via plugin-registered backends (xAI Grok-Imagine, FAL.ai Veo 3.1 / Pixverse v6 / Kling O3). Pass `image_url` to animate an image; omit it for text-to-video. |
| `kanban` | `block_kanban`, `comment_kanban`, `complete_kanban`, `create_kanban`, `heartbeat_kanban`, `link_kanban`, `list_kanban`, `show_kanban`, `unblock_kanban` | Multi-agent coordination tools. Registered for dispatcher-spawned task workers (`HERMES_KANBAN_TASK`) and for profiles that explicitly list the `kanban` toolset by name (the `all`/`*` wildcard does **not** enable it). Workers mark tasks done, block, heartbeat, comment, and create/link follow-up tasks; orchestrator profiles additionally get board-routing tools like list/unblock. |
| `memory` | `memory` | Persistent cross-session memory management. |
| `project` | `create_project`, `list_project`, `switch_project` | Create and switch desktop [Projects](../user-guide/cli.md) (named, multi-folder workspaces). GUI / desktop sessions only. |
| `safe` | `generate_image`, `analyze_vision`, `extract_web`, `search_web` (via `includes`) | Read-only research + media generation. No file writes, no terminal, no code execution. |
| `search` | `search_web` | Web search only (without extract). |
| `search_session` | `search_session` | Search past conversation sessions. |
| `skills` | `manage_skill`, `view_skill`, `list_skills` | Skill CRUD and browsing. |
| `spotify` | `spotify_albums`, `spotify_devices`, `spotify_library`, `spotify_playback`, `spotify_playlists`, `spotify_queue`, `spotify_search` | Native Spotify control (playback, queue, search, playlists, albums, library). Registered by the bundled `spotify` plugin. |
| `terminal` | `process`, `terminal` | Shell command execution and background process management. |
| `todo` | `todo` | Task list management within a session. |
| `tts` | `text_to_speech` | Text-to-speech audio generation. |
| `vision` | `analyze_vision` | Image analysis via vision-capable models. |
| `video` | `analyze_video` | Video analysis and understanding tools (opt-in, not in the default toolset — add explicitly via `--toolsets`). |
| `web` | `extract_web`, `search_web` | Web search and page content extraction. |
| `search_x` | `search_x` | Search X (Twitter) posts and threads via xAI's built-in `search_x` Responses tool. Off by default; opt in via `hermes tools`. Schema only registered when xAI credentials (SuperGrok OAuth or `XAI_API_KEY`) are configured. |
| `yuanbao` | `query_yb_group_info`, `query_yb_group_members`, `search_yb_sticker`, `send_yb_dm`, `send_yb_sticker` | Yuanbao DM/group actions and sticker search. Registered only on `hermes-yuanbao`. |

## Platform Toolsets

Platform toolsets define the complete tool configuration for a deployment target. Most messaging platforms use the same set as `hermes-cli`:

| Toolset | Differences from `hermes-cli` |
|---------|-------------------------------|
| `hermes-cli` | Full toolset — the default for interactive CLI sessions. Includes file, terminal, web, browser, memory, skills, vision, image_gen, todo, tts, delegation, code_execution, cronjob, search_session, and clarify, plus the `safe` (read-only) bundle. |
| `hermes-acp` | Drops `clarify`, `cronjob`, `generate_image`, `text_to_speech`, and all four Home Assistant tools. Focused on coding tasks in IDE context. |
| `hermes-api-server` | Drops `clarify` and `text_to_speech`. Keeps everything else — suitable for programmatic access where user interaction isn't possible. |
| `hermes-cron` | Same as `hermes-cli`. |
| `hermes-telegram` | Same as `hermes-cli`. |
| `hermes-discord` | Adds `discord` and `discord_admin` on top of `hermes-cli`. |
| `hermes-slack` | Same as `hermes-cli`. |
| `hermes-whatsapp` | Same as `hermes-cli`. |
| `hermes-signal` | Same as `hermes-cli`. |
| `hermes-matrix` | Same as `hermes-cli`. |
| `hermes-mattermost` | Same as `hermes-cli`. |
| `hermes-email` | Same as `hermes-cli`. |
| `hermes-sms` | Same as `hermes-cli`. |
| `hermes-bluebubbles` | Same as `hermes-cli`. |
| `hermes-dingtalk` | Same as `hermes-cli`. |
| `hermes-feishu` | Adds the five `feishu_doc_*` / `feishu_drive_*` tools (only used by the document-comment handler, not the regular chat adapter). |
| `hermes-qqbot` | Same as `hermes-cli`. |
| `hermes-wecom` | Same as `hermes-cli`. |
| `hermes-wecom-callback` | Same as `hermes-cli`. |
| `hermes-weixin` | Same as `hermes-cli`. |
| `hermes-yuanbao` | Adds the five `yb_*` tools (DM/group/sticker) on top of `hermes-cli`. |
| `hermes-homeassistant` | Same as `hermes-cli` (the Home Assistant tools are already present by default and activate when `HASS_TOKEN` is set). |
| `hermes-webhook` | Same as `hermes-cli`. |
| `hermes-gateway` | Internal gateway orchestrator toolset — union of every `hermes-<platform>` toolset; used when the gateway needs to accept any message source. |

## Dynamic Toolsets

### MCP server toolsets

Each configured MCP server generates a `mcp-<server>` toolset at runtime. For example, if you configure a `github` MCP server, a `mcp-github` toolset is created containing all tools that server exposes.

```yaml
# config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
```

This creates a `mcp-github` toolset you can reference in `--toolsets` or platform configs.

### Plugin toolsets

Plugins can register their own toolsets via `ctx.register_tool()` during plugin initialization. These appear alongside built-in toolsets and can be enabled/disabled the same way.

### Custom toolsets

Define custom toolsets in `config.yaml` to create project-specific bundles:

```yaml
toolsets:
  - hermes-cli
custom_toolsets:
  data-science:
    - file
    - terminal
    - code_execution
    - web
    - vision
```

### Wildcards

- `all` or `*` — expands to every registered toolset (built-in + dynamic + plugin)

A handful of tools have an additional availability check on top of toolset membership and are **not** turned on by `all`/`*` alone:

- **Capability-gated** tools (browser, `use_computer`, `code_execution`, Feishu, Home Assistant, cronjob) appear only when their backend/credential prerequisite is configured.
- **Workflow-gated** tools — the `kanban` toolset — are deliberately opt-in. `all`/`*` does **not** enable kanban; you must list `kanban` explicitly (or be a dispatcher-spawned worker with `HERMES_KANBAN_TASK` set). Kanban tools mutate shared board state, so they stay off by default even under `all`.

## Relationship to `hermes tools`

The `hermes tools` command provides a curses-based UI for toggling individual tools on or off per platform. This operates at the tool level (finer than toolsets) and persists to `config.yaml`. Disabled tools are filtered out even if their toolset is enabled.

See also: [Tools Reference](./tools-reference.md) for the complete list of individual tools and their parameters.

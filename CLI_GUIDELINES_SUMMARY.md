# Command Line Interface Guidelines — Summary

Summary of the **[Command Line Interface Guidelines](https://clig.dev/)** (clig.dev)
by Aanand Prasad, Ben Firshman, Carl Tashian, and Eva Parish — an open-source
guide to designing modern CLI programs. It blends traditional UNIX philosophy
with contemporary UX.

## Philosophy

- **Human-first design.** Design for people first; machine-readability is
  secondary and additive, not the default assumption.
- **Simple parts that work together.** Respect UNIX composability — stdin/stdout/stderr,
  exit codes, plain text, JSON. Your program will become part of larger systems.
- **Consistency across programs.** Lean on established terminal conventions so
  users' existing knowledge transfers. Break them only for a compelling reason.
- **Say just enough.** Not so little that users are lost, not so much that the
  signal drowns.
- **Ease of discovery.** Combine type-and-remember efficiency with
  see-and-point discoverability via help, examples, and error suggestions.
- **Conversation as the norm.** A CLI session is an iterative conversation —
  explore, confirm, correct, proceed.
- **Robustness** — both objective (graceful errors, idempotency) and subjective
  (feels solid and responsive).
- **Empathy & Chaos.** Show users you want them to succeed; break rules only
  with intention and clarity of purpose.

## The Basics

- Use a proven argument-parsing library (docopt, Cobra, Click, clap, …).
- Return `0` on success, non-zero on failure.
- Send primary output to **stdout**; send messages and errors to **stderr**.

## Help

- Show help with both `-h` and `--help`, for the program and every subcommand.
- Show concise help by default when run with no arguments; ignore other flags
  when help is requested.
- Lead with examples; show common flags/commands first.
- Provide a support path (website / repo link) and link to web docs.
- Suggest corrections for likely mistakes (ask before acting).
- If input is expected and stdin is interactive, prompt; if not interactive, don't hang.

## Documentation

- Provide **web docs** (searchable, linkable) *and* **terminal docs** (fast,
  offline, version-synced).
- Consider man pages (e.g. via ronn), accessible through a `help` subcommand.

## Output

- Human-readable by default; detect whether stdout is a TTY.
- Support machine-readable output: `--plain` (line/tabular) and `--json`.
- Print something on success but keep it brief; offer `-q`/`--quiet`.
- Explain state changes and make current state visible (status commands).
- Suggest follow-up commands; signal boundary-crossing actions (network, file writes).
- Use color intentionally; disable it when not a TTY, when `NO_COLOR` is set,
  when `TERM=dumb`, or with `--no-color`.
- Disable animations when stdout isn't interactive (avoid CI-log spam).
- Keep debug output / stack traces behind a verbose flag; don't treat stderr as a log.
- Use a pager (e.g. `less -FIRX`) for long output on interactive terminals.

## Errors

- Catch errors and rewrite them for humans, with actionable guidance.
- Keep signal-to-noise high — group similar errors under a header.
- Put the most important information last; use red sparingly.
- For unexpected errors, provide debug info and a bug-report path (pre-fill an
  issue URL where possible).

## Arguments and Flags

- Prefer flags to positional arguments.
- Always offer full-length flags; reserve single-letter flags for common options.
- Allow multiple arguments for simple multi-file cases (`rm a b c`), but avoid
  positional args with *different* meanings — use flags.
- Follow standard flag names: `-a/--all`, `-d/--debug`, `-f/--force`, `--json`,
  `-h/--help`, `-n/--dry-run`, `--no-input`, `-o/--output`, `-p/--port`,
  `-q/--quiet`, `-u/--user`, `--version`.
- Provide sensible defaults; prompt only when stdin is interactive, never require it.
- Confirm dangerous actions (interactive `y/yes`, or `-f/--force` in scripts).
- Support `-` for stdin/stdout; make arg/flag order flexible.
- **Never pass secrets via flags or env vars** (they leak via `ps`, history,
  logs) — use a file or stdin.

## Interactivity

- Prompt only when stdin is a TTY; support `--no-input` to disable prompts.
- Hide password input (disable echo).
- Ensure Ctrl-C always works; document escape sequences.

## Subcommands

- Be consistent across subcommands (flags, output, help).
- Use consistent `noun verb` naming; avoid ambiguous pairs like update/upgrade.

## Robustness

- Validate input early with clear messages.
- Be responsive over fast — print something within ~100 ms; signal network calls.
- Show progress for long operations (spinners, bars, estimates).
- Add timeouts to network operations; make actions recoverable.
- Design "crash-only" where possible — defer/handle cleanup on next run.
- Anticipate misuse: scripts, bad connections, concurrent instances.

## Future-Proofing

- Keep changes additive; warn before breaking changes and suggest migrations.
- Treat human-facing output as mutable; steer scripts toward `--plain`/`--json`.
- Avoid catch-all subcommands and arbitrary abbreviations.
- Avoid "time bombs" — don't depend on services that may vanish.

## Signals and Control Characters

- Handle Ctrl-C (INT) immediately; print before cleanup.
- Add timeouts to cleanup; allow repeated Ctrl-C to skip it.
- Expect that cleanup may not have run last time (crash-only).

## Configuration

Precedence (high → low): **flags → env vars → project config (`.env`) →
user config → system config.**

- Per-invocation variation → flags.
- Per-machine/per-project → env vars and `.env`.
- Shared, version-controlled workflow settings → config files.
- Follow the **XDG Base Directory Spec** (`~/.config`), not scattered dotfiles.
- Ask consent before modifying system config; prefer new files.

## Environment Variables

- Use for context that varies with the terminal session.
- Names: uppercase, digits, underscores; not starting with a digit; single-line.
- Don't commandeer POSIX-standard names.
- Honor general ones: `NO_COLOR`, `FORCE_COLOR`, `DEBUG`, `EDITOR`,
  `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`, `SHELL`, `TERM`, `TMPDIR`, `HOME`,
  `PAGER`, `LINES`/`COLUMNS`.
- Read `.env` for project config, but don't use it as a real config file or for secrets.
- **Never store secrets in env vars** — use credential files, pipes, or a secret service.

## Naming

- Simple, memorable, single word; lowercase and dashes only.
- Short and easy to type; reserve ultra-short names for hourly-use utilities.

## Distribution

- Ship a single binary where possible; otherwise use native package installers.
- Make uninstallation easy and documented.

## Analytics

- Never phone home without explicit consent.
- If collecting data, be transparent (what, why, retention, anonymization).
- Prefer opt-in; document prominently if opt-out.
- Consider alternatives: doc instrumentation, download metrics, direct feedback.

---

*Source: <https://clig.dev/> — an open-source, community-maintained guide.*

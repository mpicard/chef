# Agent Instructions

## Communication
- Be terse.
- Do not restate the task.
- Do not narrate routine actions.
- Report only relevant findings, actions, errors, and next steps.

## Context
- Inspect only files relevant to the task.
- Prefer `rg`, targeted file reads, and `git diff`.
- Never read large files or logs in full unless necessary.
- Keep command output below ~4000 bytes when possible.

## Execution
- Make the smallest change that solves the problem.
- Do not refactor unrelated code.
- Do not add dependencies unless necessary.
- Do not ask for confirmation for routine, reversible actions.
- Do not create a plan for trivial tasks.

## Validation
- Validate proportionally to risk.
- Prefer file-scoped tests, linting, and typechecking.
- Avoid full builds/test suites unless required.
- Do not repeat checks that already passed.

## Errors
- Inspect failures and make one targeted fix when the cause is clear.
- Do not dump full logs.
- If the cause is unclear, report the relevant error and stop.

## Output
- Keep responses concise.
- For completed tasks: state what changed and validation performed.
- For blocked tasks: state the blocker and what is needed.

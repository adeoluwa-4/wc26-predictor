# Repository Workflow

Use a strict file-by-file Git workflow for this repository.

Commit and push rules:

1. Never batch unrelated files in one commit.
2. Make exactly one commit per file changed.
3. Push to `main` after each commit.
4. Use clear commit messages in this format:
   - `feat: update <path>`
   - `fix: update <path>`
   - `data: update <path>`
   - `model: update <path>`
   - `ui: update <path>`
5. After every commit and push, print:
   - commit hash
   - file committed
   - push result
6. If a file is too large for GitHub, greater than 100MB, stop and compress or regenerate that single artifact before continuing.
7. If asked to split old bulk commits, create a backup branch, rebuild history into one-file commits, and force-push `main` with lease.
8. At the end, provide:
   - total commits made
   - ordered commit list
   - confirmation that the working tree is clean

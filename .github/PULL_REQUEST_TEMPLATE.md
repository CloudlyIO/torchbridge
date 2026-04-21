## What does this PR do?

<!-- One sentence. If you can't write one sentence, the PR is too large. -->

## Contributor journey stage

<!-- Where are you in the cycle? Check one. -->

- [ ] Day 1 — found a bug while testing the install
- [ ] Day 2 — hit this while using TorchBridge on my own models
- [ ] Planned contribution — working from an open issue
- [ ] Maintainer — internal fix/refactor

## Linked issue

<!-- Every PR must close or reference an issue. No exceptions. -->

Closes #

## Changes

<!-- Bullet list of what changed and why. Be specific — "fixed bug" is not enough. -->

-

## Test evidence

<!-- Paste the pytest summary line. Do not submit without this. -->

```
# pytest tests/ -q -m "not gpu and not slow"
... N passed, M skipped in Xs
```

## Ruff

<!-- Paste the output. Must be clean. -->

```
# ruff check src tests
# (no output = clean)
```

## Checklist

- [ ] Linked issue filled in above
- [ ] Tests added or updated for every behaviour change
- [ ] `pytest` summary pasted above — all passing
- [ ] `ruff check src tests` clean — output pasted above
- [ ] No new hardcoded version strings (version lives in `pyproject.toml` only)
- [ ] `CHANGELOG.md` entry added if this changes user-visible behaviour

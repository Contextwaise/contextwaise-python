# Contributing

Thanks for helping. Bug reports, fixes and docs improvements are all welcome.

## Before you start

- **Bugs:** open an issue with a minimal reproduction first, unless the fix is small and obvious.
- **Features and behavior changes:** open an issue to discuss it before writing code.
  Changes to what gets cut affect every user's answers, so they need agreement up front.
- **Security issues:** see [SECURITY.md](SECURITY.md), never a public issue.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Tests never call the network: the Contextwaise API is stubbed.

## Pull requests

- Keep each PR to one change, with tests for it.
- Update the README when the public API or an option changes.
- The [TypeScript](https://github.com/Contextwaise/contextwaise-typescript) and
  [Python](https://github.com/Contextwaise/contextwaise-python) SDKs make the same decisions.
  A change to cleanup, relevance or the SDK core should land in both; say in the PR
  if the other side is still to do.
- CI must pass, and a maintainer reviews every PR.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).

# Contributing to Assay

Thanks for your interest. Assay is a measurement project, so contributions are judged by whether they keep results **correct, reproducible and honestly reported**.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md) (add one, for example the Contributor Covenant, before accepting outside contributions).

## Ways to help

- **New metrics.** Metrics plug in behind one interface: `score(document, context) -> MetricResult`, returning a value in [0, 1], an optional critical-failure flag and an explanation. Include a written definition and hand-built test fixtures.
- **Human rating.** Additional raters strengthen the calibration study. Open an issue to coordinate; raters are recorded under pseudonymous ids.
- **Corpus additions.** Propose repositories with a permissive license (for example MIT or Apache-2.0).
- **Bug reports and docs fixes.**

## Ground rules

- **No confidential material.** Never submit proprietary, employer or otherwise non-public code, prompts, documents or data.
- **Licensing.** Contributions are licensed under the project's MIT license. Corpus sources must have a license that permits use and redistribution of generated documentation.
- **No secrets.** Do not commit API keys, tokens or `.env` files. CI fails on detected secrets.
- **Pre-registered evaluation.** Do not tune weights, thresholds or rubrics against the held-out test split. Changes that affect scoring must state which split informed them.

## Development setup

```bash
git clone https://github.com/salgue441/assay.git
cd assay
uv sync
cp .env.example .env     # add provider keys locally; never commit
uv run pre-commit install
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Before opening a pull request

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

- Keep pull requests focused: one logical change each.
- Add or update tests; every metric needs fixtures with known expected values.
- Update `docs/SAD.md` when a change alters architecture, a decision record or a target.
- Note any change to model identifiers, prompt hashes, rubric version or weights in the PR description, since these affect reproducibility.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.

## Reporting results

Report intervals, not just point estimates. Negative and null results are welcome and get the same prominence as positive ones.

## Security

Do not open a public issue for a vulnerability or an exposed credential. Report it privately to the maintainer through the contact listed on the GitHub profile.

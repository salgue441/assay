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
./scripts/setup-env.sh   # or scripts\setup-env.ps1 on Windows
```

Requires [uv](https://docs.astral.sh/uv/); the project Python (3.12) is pinned in `.python-version`. The setup script creates a dedicated conda env named `assay` when conda is installed (otherwise a uv-managed `.venv`), installs the locked dependencies and creates `.env` from `.env.example`. Add provider keys to `.env` locally; never commit it.

## Before opening a pull request

```bash
./scripts/check.sh        # or scripts\check.ps1; add --fix / -Fix to apply formatting first
```

This runs the same steps as CI: formatting, lint, type-check, tests (excluding real LLM calls) and the GitHub Actions audit.

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

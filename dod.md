## Facts

- [x] Git リポジトリのルートは `ortify/` である。
- [x] `ortify/uv.lock` が存在し、このプロジェクトはすでに `uv` を利用できる状態にある。
- [x] `ortify/README.md` の開発手順は現在 `pip install -e ".[dev]"` を前提としている。
- [x] `ortify/pyproject.toml` には `dev` 依存として `pre-commit`、`pytest`、`ruff` が定義されている。
- [x] Git リポジトリ内には `.pre-commit-config.yaml` と GitHub Actions の Ruff CI がまだ存在しない。

## Required changes based on facts

- [x] README の開発手順を `uv` 前提に変更する。
- [x] Git リポジトリ内に `uv` 前提の pre-commit 設定を追加する。
- [x] Git リポジトリ内に `uv` 前提で Ruff を実行する CI を追加する。
- [x] 必要なら `uv.lock` を更新して依存定義と整合させる。

## Constraints

- [x] 変更対象は `ortify/` リポジトリ内に限定する。
- [x] Lint 設定そのものは既存の `pyproject.toml` の Ruff 設定を流用する。
- [x] 追加するローカル開発手順と CI は `uv` を前提にして統一する。

## Verification

- [x] `uv run --extra dev ruff check .` が通ることを確認する。
- [x] `uv run pre-commit validate-config` で pre-commit 設定が妥当であることを確認する。
- [x] CI 設定が `uv` セットアップと Ruff 実行を行うことを確認する。

## Open Questions

- None.

## Deferred

- テスト実行やビルド公開フローまで `uv` ベースに広げること。

## データ生成と Mallows パラメータ

### Mallows モデル（順位生成）
順位は Mallows モデルを Repeated Insertion Method (RIM) で生成する。

**具体的な生成式（RIM の確率）**
- 参照順序 `reference` を先頭から 1 要素ずつ挿入する。
- そのときの挿入位置 `idx` は以下の確率で選ばれる（コードの通り）。
  - `size = i + 1`（現在の順位長）
  - `probs[j] ∝ phi^(size - j - 1)`（`j` は挿入位置 0..size-1）
  - `np.random.choice(size, p=probs)` で `idx` を選ぶ
  - `phi = 1.0` なら `probs` は一様（完全ランダムに近い挿入）
  - `phi < 1.0` ほど先頭寄り（低い `j`）に挿入されやすい

**パラメータの意味と実際の値**
- `phi`（`priority_phi` / `preference_phi` に渡される）
  参照順序に対する集中度。
  - 1.0 に近いほど参照順序に近い順位が出やすい
  - 小さいほどランダム性が高い
- `priority_phi`
  各カテゴリの「エージェント優先順位」を生成するための `phi`。デフォルトは `0.8`。
- `preference_phi`
  各エージェントの「カテゴリ選好順位」を生成するための `phi`。デフォルトは `0.7`。

### データセット生成パラメータ（`BatchDatasetConfig`）
実際に使っている数値は以下。
- `num_agents`
  by_size 実験では `--sizes` の値を使う。デフォルトは `[50, 100, 1000]`。
- `num_categories`
  デフォルトは `10`。
- `capacity_ratio`
  デフォルトは `1.0`。総定員 = `num_agents * 1.0`。
- `capacity_std`
  デフォルトは `0.0`。ばらつきを使わず一律の定員にする。
- `eligibility_prob`
  デフォルトは `0.55`。各カテゴリが許容される確率が 55%。
- `max_preference_length`
  デフォルトは `5`。許容カテゴリは最大 5 個。
- `priority_phi`
  デフォルトは `0.8`。カテゴリの優先順位生成に使う。
- `preference_phi`
  デフォルトは `0.7`。エージェントの選好順位生成に使う。
- `seed`
  by_size 実験では `0..99`（`BATCH_DATASET_SEEDS`）から `--count` 個を使用。

### by_size 実験のコマンド引数（デフォルト）
- `--sizes`
  デフォルトは `[50, 100, 1000]`。
- `--count`
  デフォルトは `100`（`len(BATCH_DATASET_SEEDS)`）。
- `--algorithms`
  デフォルトは全て（`mma`, `rev`, `rev_bipartite`, `scu`, `scucomb`）。

### 生成の詳細
- `capacity_std <= 0` の場合、カテゴリ定員は全カテゴリで均等（端数は先頭カテゴリに +1 で配分）。
- `capacity_std > 0` の場合、カテゴリ定員は正規分布で生成し、総定員に合うよう調整する。
- カテゴリ優先順位とエージェント選好順位は Mallows（各 `phi`）で生成する。
- 許容カテゴリは `eligibility_prob` に基づいて選び、`max_preference_length` で打ち切る。

### 参照コード
- `codes/data_generation/mallows.py`
- `codes/data_generation/dataset.py`
- `codes/batch_shared.py`
- `codes/results/run_all_batches_by_size.py`

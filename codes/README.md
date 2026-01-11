## researchリポジトリ

このリポジトリは研究用のコードをまとめたディレクトリである。


### compare_six_algo.py
MMA,DA,SCUなど6つのアルゴリズムを実装したものである。

### testディレクトリ
実装したアルゴリズムが適切に動くか確認するディレクトリ

### resutlsディレクトリ
実際のマッチングの中身を出力するプログラムのディレクトリ

resultsディレクトリに入って
```bash
find results -name "*.py" -exec python {} \;
```

で全部実行

### makefileによる実行例
```bash
make batch-compare SIZES="50 100 1000" COUNT=100
```
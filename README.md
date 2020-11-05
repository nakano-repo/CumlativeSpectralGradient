# CumlativeSpectralGradient

下記論文（Accepted by CVPR2019）の再現実装．

https://arxiv.org/abs/1905.07299

入力データのカテゴリ同士の近さ（学習のしやすさ）を数値化する．

## Requirement
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- umap

## Usage
```
model = CumulativeSpectralGradient()
model.fit(dataset.data, dataset.target, montecarlo_sampling_size, k_for_nearest_neighbor, embedding_mode='umap')
model.show_result(only_graph=False)
```
- dataset.data: データセットの入力変数
- dataset.target: データセットのラベル
- montecarlo_sampling_size: 類似度行列計算の際のサンプリングサイズ
- k_for_nearest_neighbor: 類似度行列計算の際に使われているk近傍法で用いる近傍サンプルの数（自分自身を1とする）
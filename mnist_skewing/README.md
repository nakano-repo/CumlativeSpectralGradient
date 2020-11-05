# MNISTのラベルを歪める実験

第1段階ではMNISTの0と1のラベルをランダムに入れ替える，第2段階では0〜2のラベルをランダムに入れ替える，…，第9段階で全ラベルをランダムに入れ替えることにより，学習がしにくい状態を作る．それがラプラシアン行列の固有値やCSGにどう現れるかを見る．

## Umapの場合

最後の方で単調増加しなくなる

### 固有値

![umap_eigenvalues](umap/umap_eigenvalues.png)

### CSG

![umap_csgs](umap/umap_csgs.png)

## t-SNEの場合

論文ほどきれいにはならない

### 固有値

![tsne_eigenvalues](tsne/tsne_eigenvalues.png)

### CSG

![tsne_csgs](tsne/tsne_csgs.png)

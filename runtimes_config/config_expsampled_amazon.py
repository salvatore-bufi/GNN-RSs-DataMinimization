TEMPLATE_BPR = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.BPRMF:
      meta:
        verbose: True
        save_weights: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      lr: 0.0007756486208064678
      epochs: 1000
      factors: [ 64 ]
      batch_size: 1024
      l_w: 0.004730315267298346
      normalize: True
      seed: 123
      early_stopping:
        patience: 10
        mode: auto
        monitor: Recall@20
        verbose: True
"""



TEMPLATE_DIRECTAU = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.DirectAU:
      meta:
        verbose: True
        save_weights: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      lr: 0.004214945628333456
      epochs: 1000
      factors: [ 64 ]
      batch_size: 1024
      l_w: 4.869521027891824e-06
      gamma: 1.3710835880821897
      seed: 123
      early_stopping:
        patience: 10
        mode: auto
        monitor: Recall@20
        verbose: True
"""


TEMPLATE_ITEMKNN = """experiment:
  backend: tensorflow
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    ItemKNN:
      meta:
        verbose: True
        save_weights: False
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      neighbors: 200
      similarity: cosine
      seed: 123"""

TEMPLATE_LIGHTGCN = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.LightGCN:
      meta:
        verbose: True
        save_weights: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      lr: 0.0011804184702263693
      epochs: 1000
      factors: [ 64 ]
      batch_size: 1024
      l_w: 0.00035562834787914794
      n_layers: 2
      normalize: True
      seed: 123
      early_stopping:
        patience: 10
        mode: auto
        monitor: Recall@20
        verbose: True
"""

TEMPLATE_SIMPLEX = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.SimpleX:
      meta:
        verbose: True
        save_weights: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      lr: 0.0015707032958272276
      epochs: 1000
      factors: [ 64 ]
      batch_size: 1024
      lw: 4.869521027891824e-06
      m: 0.9
      nw: 200
      g: 0.2
      dprob: 0.1 
      n_n: 500
      seed: 123
      early_stopping:
        patience: 10
        mode: auto
        monitor: Recall@20
        verbose: True
"""


TEMPLATE_SVDGCN = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.SVDGCN:
      meta:
        verbose: True
        save_weights: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      factors: 64
      epochs: 1000
      batch_size: 1024
      l_w: 0.0003542794098758525
      lr:  5.541551797327792
      req_vec: 90
      beta: 3.0
      alpha: 3.0
      coef_u: 0.5
      coef_i: 0.7
      seed: 123
      early_stopping:
        patience: 10
        mode: auto
        monitor: Recall@20
        verbose: True"""

TEMPLATE_ULTRAGCN = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.UltraGCN:
      meta:
        verbose: True
        save_weights: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      lr: 0.00026474549244554077
      epochs: 1000
      factors: [ 64 ]
      batch_size: 1024
      g: 3.0390787274500798e-05
      l: 0.10002170802377214
      w1: 2.1211369071454087e-05
      w2: 0.0940189567127606
      w3: 0.004026906602401245
      w4: 0.4403762515721559
      ii_n_n: 10
      n_n: 200
      n_w: 500
      s_s_p: False
      i_w: 1e-4
      seed: 123
      early_stopping:
        patience: 10
        mode: auto
        monitor: Recall@20
        verbose: True
"""


TEMPLATE_USERKNN = """experiment:
  backend: tensorflow
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    UserKNN:
      meta:
        verbose: True
        save_weights: False
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      neighbors: 200
      similarity: cosine
      seed: 123
"""

TEMPLATE_GFCF = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [nDCGRendle2020, Recall]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.GFCF:
      meta:
        verbose: True
        save_weights: False
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      svd_factors: 256
      alpha: 0.27926506534762485
      seed: 123
"""
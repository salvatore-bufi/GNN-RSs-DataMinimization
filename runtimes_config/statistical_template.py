ml_1m_template = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../dataset/ml-1m/full/3.tsv
    validation_path: ../dataset/ml-1m/val.tsv
    test_path: ../dataset/ml-1m/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [10]
    paired_ttest: True
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  models:
    RecommendationFolder:  
        folder: ./result_statistic/{dataset_name}/recs
"""

ambar_template = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../dataset/ambar/full/1.tsv
    validation_path: ../dataset/ambar/val.tsv
    test_path: ../dataset/ambar/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [10]
    paired_ttest: True
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  models:
    RecommendationFolder:  
        folder: result_statistic/{dataset_name}/recs
"""
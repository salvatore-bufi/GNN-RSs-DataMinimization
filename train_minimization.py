from elliot.run import run_experiment
import os

# Import template modules for each dataset
from runtimes_config import config_exp_template_amazon as amz_templates
from runtimes_config import config_exp_template_yelp as yelp_templates
from runtimes_config import config_exp_template_amazonsw_minimized as amzsw_templates

CONFIG_DIR = './config_files_data_min'
log_file_path = os.path.abspath('./log_error_experiments_yelp.txt')


def create_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created : {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    return None


# Datasets to train on
DATASETS = ['amazon-book', 'yelp', 'amazon-software']

INTERACTIONS_NUM = [1, 3, 6, 9, 12, 15, 18, 50, 100]
# INTERACTIONS_NUM = [1]
STRATEGIES = [
    'highest_variance',
    'least_favorite',
    'most_characteristic',
    'most_favorite',
    'most_rated',
    'most_recent',
    'random'
]


def get_templates_for_dataset(dataset: str):
    """
    Return the correct TEMPLATE dict depending on the dataset.
    This preserves the original behavior of:
    - using config_exp_template_amazon for amazon-book
    - using config_exp_template_yelp for yelp
    """
    if dataset == 'amazon-book':
        return {
            'BPR': amz_templates.TEMPLATE_BPR,
            'DIRECTAU': amz_templates.TEMPLATE_DIRECTAU,
            'ITEM_KNN': amz_templates.TEMPLATE_ITEMKNN,
            'LIGHT_GCN': amz_templates.TEMPLATE_LIGHTGCN,
            'SIMPLEX': amz_templates.TEMPLATE_SIMPLEX,
            'ULTRAGCN': amz_templates.TEMPLATE_ULTRAGCN,
            'USER_KNN': amz_templates.TEMPLATE_USERKNN,
            'GFCF': amz_templates.TEMPLATE_GFCF,
            'DGCF': amz_templates.TEMPLATE_DGCF,  # if you decide to use it
        }
    elif dataset == 'yelp':
        return {
            'BPR': yelp_templates.TEMPLATE_BPR,
            'DIRECTAU': yelp_templates.TEMPLATE_DIRECTAU,
            'ITEM_KNN': yelp_templates.TEMPLATE_ITEMKNN,
            'LIGHT_GCN': yelp_templates.TEMPLATE_LIGHTGCN,
            'SIMPLEX': yelp_templates.TEMPLATE_SIMPLEX,
            'ULTRAGCN': yelp_templates.TEMPLATE_ULTRAGCN,
            'USER_KNN': yelp_templates.TEMPLATE_USERKNN,
            'GFCF': yelp_templates.TEMPLATE_GFCF
        }
    elif dataset == 'amazon-software':
        return {
            'BPR': amzsw_templates.TEMPLATE_BPR,
            'DIRECTAU': amzsw_templates.TEMPLATE_DIRECTAU,
            'ITEM_KNN': amzsw_templates.TEMPLATE_ITEMKNN,
            'LIGHT_GCN': amzsw_templates.TEMPLATE_LIGHTGCN,
            'SIMPLEX': amzsw_templates.TEMPLATE_SIMPLEX,
            'ULTRAGCN': amzsw_templates.TEMPLATE_ULTRAGCN,
            'USER_KNN': amzsw_templates.TEMPLATE_USERKNN,
            'GFCF': amzsw_templates.TEMPLATE_GFCF,
            'DGCF': amzsw_templates.TEMPLATE_DGCF,  # if you decide to use it
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train_models():
    create_directory(CONFIG_DIR)

    for dataset in DATASETS:
        templates = get_templates_for_dataset(dataset)

        for model, TEMPLATE in templates.items():
            for strategy in STRATEGIES:
                for int_n in INTERACTIONS_NUM:
                    dataset_name = f"{dataset}_{strategy}_{int_n}"

                    config = TEMPLATE.format(
                        dataset=dataset,
                        strategy=strategy,
                        interactions_numb=int_n,
                        dataset_name=dataset_name
                    )

                    config_path = os.path.abspath(
                        os.path.join(CONFIG_DIR, 'runtime_metrics_conf.yml')
                    )

                    with open(config_path, 'w') as file:
                        file.write(config)

                    try:
                        run_experiment(config_path)
                    except Exception as e:
                        with open(log_file_path, 'a') as log_file:
                            error_msg = (
                                f'Error Processing {TEMPLATE} \n'
                                f'{strategy} \n'
                                f'{int_n} \n'
                                f'{model} \n'
                                f'{str(e)} \n\n'
                            )
                            log_file.write(error_msg)


if __name__ == "__main__":
    train_models()
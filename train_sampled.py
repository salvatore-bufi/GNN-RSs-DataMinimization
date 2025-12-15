import os
import argparse
from elliot.run import run_experiment
import runtimes_config.config_expsampled_amazon as amz_template
import runtimes_config.config_expsampled_yelp as yelp_template

### TEMPLATES
TEMPLATES_AMAZON = {
    'BPRMF': amz_template.TEMPLATE_BPR,
    'DirectAU': amz_template.TEMPLATE_DIRECTAU,
    'UltraGCN': amz_template.TEMPLATE_ULTRAGCN,
    'LightGCN': amz_template.TEMPLATE_LIGHTGCN,
    'SimpleX': amz_template.TEMPLATE_SIMPLEX,
    'SVDGCN': amz_template.TEMPLATE_SVDGCN,
    'UserKNN': amz_template.TEMPLATE_USERKNN,
    'ItemKNN': amz_template.TEMPLATE_ITEMKNN,
    'GFCF': amz_template.TEMPLATE_GFCF
}

TEMPLATES_YELP = {
    'BPRMF': yelp_template.TEMPLATE_BPR,
    'DirectAU': yelp_template.TEMPLATE_DIRECTAU,
    'UltraGCN': yelp_template.TEMPLATE_ULTRAGCN,
    'LightGCN': yelp_template.TEMPLATE_LIGHTGCN,
    'SimpleX': yelp_template.TEMPLATE_SIMPLEX,
    'SVDGCN': yelp_template.TEMPLATE_SVDGCN,
    'UserKNN': yelp_template.TEMPLATE_USERKNN,
    'ItemKNN': yelp_template.TEMPLATE_ITEMKNN,
    'GFCF': yelp_template.TEMPLATE_GFCF
}

TEMPLATES = {'amazon-book': TEMPLATES_AMAZON, 'yelp': TEMPLATES_YELP}

# Default lists
DEFAULT_DATASETS = ['yelp', 'amazon-book']
DEFAULT_STRATEGIES = ['edge-dropout', 'node-dropout']
DEFAULT_RANGE = (0, 900)
DEFAULT_MODELS = ['UltraGCN', 'LightGCN', 'GFCF']

CONFIG_DIR = './config_files'


def build_log_file_path(datasets, models, interactions_range):
    models_used = models or DEFAULT_MODELS
    range_used = interactions_range or DEFAULT_RANGE

    max_r = range_used[1]
    datasets_part = "_".join(datasets)
    models_part = "_".join(models_used)

    filename = f"log_error_sampled_train_{models_part}_{datasets_part}_{max_r}.txt"
    return os.path.abspath(filename)


def train_on_sampled(datasets, strategies=None, interactions_range=None, models=None, log_file_path=None):
    strategies = strategies or DEFAULT_STRATEGIES
    min_int, max_int = interactions_range or DEFAULT_RANGE
    interactions_list = list(range(min_int, max_int + 1))
    allowed_models = set(models or DEFAULT_MODELS)

    # log file path
    log_file_path = log_file_path or build_log_file_path(datasets, list(allowed_models), (min_int, max_int))

    os.makedirs(CONFIG_DIR, exist_ok=True)

    for dataset in datasets:
        actual_configuration = dict()
        for int_n in interactions_list:
            for strategy in strategies:
                dataset_dir = f"./data/{dataset}/{strategy}"
                candidate_file = os.path.join(dataset_dir, f"{int_n}.tsv")
                if not os.path.exists(candidate_file):
                    continue

                dataset_name = f"{dataset}_{strategy}_{int_n}"
                print(f" Training on {dataset_name}\n")

                for model in allowed_models:
                    current_template = TEMPLATES[dataset][model]
                    print(f"\t\t Model: {model}")
                    actual_configuration['dataset'] = dataset
                    actual_configuration['strategy'] = strategy
                    actual_configuration['interactions_numb'] = str(int_n)
                    actual_configuration['dataset_name'] = dataset_name
                    current_config = current_template.format(**actual_configuration)

                    unique_config_filename = f"{model}_{dataset}_{strategy}_{int_n}.yml"
                    unique_config_path = os.path.abspath(os.path.join(CONFIG_DIR, unique_config_filename))

                    with open(unique_config_path, 'w') as f:
                        f.write(current_config)
                    try:
                        run_experiment(unique_config_path)
                        os.remove(unique_config_path)
                    except Exception as e:
                        with open(log_file_path, 'a') as log_file:
                            error_msg = (
                                f'Error Processing {dataset_name} ({model})\n'
                                f'CONFIG={unique_config_filename}\n'
                                f'STRATEGY={strategy}, INTERACTIONS={int_n}\n{str(e)}\n\n'
                            )
                            log_file.write(error_msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute recommendations for given datasets/strategies/interactions/models."
    )
    parser.add_argument(
        '--dataset', '-d', nargs='+', required=True,
        help="One or more datasets (space-separated). Example: --dataset yelp amazon-book"
    )
    parser.add_argument(
        '--strategies', '-s', nargs='+', default=None,
        help=f"List of sampling strategies (default: {DEFAULT_STRATEGIES})"
    )
    parser.add_argument(
        '--range', '-r', nargs=2, type=int, default=None,
        help=f"Range of interactions: min max  (default: {DEFAULT_RANGE[0]} {DEFAULT_RANGE[1]})"
    )
    parser.add_argument(
        '--models', '-m', nargs='+', default=None,
        help=f"List of models to run (default: {DEFAULT_MODELS})"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    interactions_range = tuple(args.range) if args.range else None
    log_path = build_log_file_path(
        datasets=args.dataset,
        models=args.models,
        interactions_range=interactions_range
    )

    train_on_sampled(
        datasets=args.dataset,
        strategies=args.strategies,
        interactions_range=interactions_range,
        models=args.models,
        log_file_path=log_path
    )

# example of usage:
# python train_on_sampled.py --dataset yelp amazon-book --range 0 2 --models BPRMF LightGCN
# -> log: log_error_sampled_train_BPRMF_LightGCN_yelp_amazon-book_2.txt
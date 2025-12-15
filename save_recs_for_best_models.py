import os
import argparse
from elliot.run import run_experiment
from runtimes_config.compute_save_recs_template import (
    extract_best_model_files_absolute_paths,
    extract_models_parameter,
    fulfill_template,
    TEMPLATES
)

# Default lists
DEFAULT_STRATEGIES_ALL = [
    'highest_variance', 'least_favorite', 'most_characteristic', 'most_favorite',
    'most_rated', 'most_recent', 'random', 'full'
]
# default da usare SE l'utente non passa --strategies (esclude 'full')
DEFAULT_STRATEGIES_NO_FULL = [
    'highest_variance', 'least_favorite', 'most_characteristic', 'most_favorite',
    'most_rated', 'most_recent', 'random'
]
DEFAULT_INTERACTIONS_NUM = ['1', '3', '6', '9', '12', '15', '18', '50', '100']
DEFAULT_MODELS = ['BPRMF', 'DirectAU', 'UltraGCN', 'LightGCN', 'SimpleX', 'SVDGCN', 'UserKNN', 'ItemKNN', 'GFCF']

BEST_MODELS_PARAMS_DIR = 'JSON_best_models'
CONFIG_DIR = './config_files'
LOG_FILE_PATH = os.path.abspath('log_error_recs.txt')

def save_recs(datasets, strategies=None, interactions_num=None, models=None):
    # Se l'utente non specifica le strategie -> usa il default SENZA 'full'
    strategies = strategies or DEFAULT_STRATEGIES_NO_FULL
    default_interactions = interactions_num or DEFAULT_INTERACTIONS_NUM
    allowed_models = set(models or DEFAULT_MODELS)

    # Assicuro che la cartella per i config esista
    os.makedirs(CONFIG_DIR, exist_ok=True)

    for dataset in datasets:
        for strategy in strategies:
            # se è full, forzo ['1'] indipendentemente dall'input
            if strategy == 'full':
                interactions_list = ['1']
            else:
                interactions_list = default_interactions

            for int_n in interactions_list:
                dataset_name = f"{dataset}_{strategy}_{int_n}"
                print(f" Computing recs for {dataset_name}\n")

                best_models_config_files_path_list = extract_best_model_files_absolute_paths(
                    parent_directory=BEST_MODELS_PARAMS_DIR,
                    dataset_name=dataset_name
                )

                for best_model_path in best_models_config_files_path_list:
                    actual_configuration = extract_models_parameter(best_model_path)
                    current_model = actual_configuration['model']

                    # Check modello
                    if current_model not in allowed_models:
                        print(f"  Skipping model {current_model} (not in allowed list).")
                        continue

                    print(f"\t\t Model: {current_model}")

                    current_template = TEMPLATES[dataset][current_model]

                    # arricchisco la config
                    actual_configuration['dataset'] = dataset
                    actual_configuration['dataset_name'] = dataset_name
                    actual_configuration['strategy'] = strategy
                    actual_configuration['interactions_numb'] = int_n

                    current_config = fulfill_template(current_template, actual_configuration)

                    # Nome UNIVOCO del file di config, riflette il ciclo interno
                    unique_config_filename = f"{current_model}_{dataset}_{strategy}_{int_n}.yml"
                    unique_config_path = os.path.abspath(os.path.join(CONFIG_DIR, unique_config_filename))

                    with open(unique_config_path, 'w') as f:
                        f.write(current_config)

                    try:
                        # esegue l'esperimento usando il file univoco
                        run_experiment(unique_config_path)
                    except Exception as e:
                        with open(LOG_FILE_PATH, 'a') as log_file:
                            error_msg = (
                                f'Error Processing {dataset_name} ({current_model})\n'
                                f'CONFIG={unique_config_filename}\n'
                                f'STRATEGY={strategy}, INTERACTIONS={int_n}\n{str(e)}\n\n'
                            )
                            log_file.write(error_msg)

def parse_args():
    # Esempio:
    # python compute_recs.py --dataset yelp --strategies most_recent full --models BPRMF LightGCN --interactions 1 3 6 9
    parser = argparse.ArgumentParser(
        description="Compute recommendations for given datasets/strategies/interactions/models."
    )
    parser.add_argument(
        '--dataset', '-d', nargs='+', required=True,
        help="Uno o più dataset (spazio-separati). Esempio: --dataset yelp amazon-book"
    )
    parser.add_argument(
        '--strategies', '-s', nargs='+', default=None,
        help=f"Lista di strategie (se omessa usa il default senza 'full'). Default: {DEFAULT_STRATEGIES_NO_FULL}"
    )
    parser.add_argument(
        '--interactions', '-n', nargs='+', default=None,
        help=f"Lista di numeri interazioni (se omessa usa il default). Default: {DEFAULT_INTERACTIONS_NUM}"
    )
    parser.add_argument(
        '--models', '-m', nargs='+', default=None,
        help=f"Lista di modelli (se omessa usa il default). Default: {DEFAULT_MODELS}"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    save_recs(
        datasets=args.dataset,
        strategies=args.strategies,      # se None -> DEFAULT_STRATEGIES_NO_FULL
        interactions_num=args.interactions,
        models=args.models
    )
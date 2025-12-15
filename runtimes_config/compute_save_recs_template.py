import json
from typing import Dict, List

import runtimes_config.save_recs_template_amazon as amz_template
import runtimes_config.save_recs_template_yelp as yelp_template
import os

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

def extract_models_parameter(model_parameter_json_path: str) -> Dict[str, str]:
    configuration = dict()
    with open(model_parameter_json_path) as file:
        config_data = json.load(file)

    for entry in config_data:
        if 'recommender' in entry:
            configuration['model'] = entry['recommender'].split('_')[0]
        if 'configuration' in entry:
            configuration.update(entry['configuration'])
    str_dict = {str(key): str(value) for key, value in configuration.items()}
    return str_dict


def extract_best_model_files_names(parent_directory: str, dataset_name: str) -> List[str]:
    performance_directory = os.path.abspath(os.path.join(parent_directory, dataset_name, 'performance'))
    best_models_files_names_list = [file for file in os.listdir(performance_directory) if file.startswith("bestmodel")]
    return best_models_files_names_list


def extract_best_model_files_absolute_paths(parent_directory: str, dataset_name: str) -> List[str]:
    performance_directory = os.path.abspath(os.path.join(parent_directory, dataset_name, 'performance'))
    best_models_files_absolute_paths_list = [os.path.join(performance_directory, file) for file in
                                             os.listdir(performance_directory) if file.startswith("bestmodel")]
    return best_models_files_absolute_paths_list


def fulfill_template(template: str, str_dict: Dict[str, str]) -> str:
    # Handle best_iteration overrides for validation_rate and epochs
    # str_dict contains the best model hyperparameter in the format: parameter_name : value
    if 'best_iteration' in str_dict:
        str_dict['validation_rate'] = str_dict['best_iteration']
        str_dict['epochs'] = str_dict['best_iteration']

    # Fill the TEMPLATE using the formatted dictionary
    try:
        fulfilled_template = template.format(**str_dict)
    except KeyError as e:
        missing_key = str(e).strip("'")
        raise ValueError(f"Missing required parameter in the dictionary: {missing_key}")

    return fulfilled_template

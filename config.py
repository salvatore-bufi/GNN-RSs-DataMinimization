import os

DATA_FOLDER = os.path.abspath('./data')
OUTPUT_FOLDER = os.path.abspath('./data/')
RESULT_FOLDER = os.path.abspath('./results/')
ACCEPTED_DATASETS = ['amazon-book', 'yelp']
ACCEPTED_SPLITTINGS = ['edge-dropout', 'node-dropout']
ACCEPTED_METRICS = ['Recall', 'nDCG']


ACCEPTED_CHARACTERISTICS = ['space_size_log', 'shape_log', 'density_log', 'gini_user',
                            'gini_item', 'average_degree_users_log', 'average_degree_items_log',
                            'average_clustering_coefficient_dot_users_log',
                            'average_clustering_coefficient_dot_items_log', 'degree_assortativity_users',
                            'degree_assortativity_items']

ACCEPTED_CHARACTERISTICS_MIN = ['space_size_log','density_log', 'gini_user',
                            'gini_item', 'degree_assortativity_items']

ACCEPTED_CHARACTERISTICS_SUB = ['space_size_log', 'shape_log', 'gini_user',
                                'degree_assortativity_users',
                                'degree_assortativity_items']

ALL_ACCEPTED_CHARACTERISTICS_SUB = ['space_size_log', 'shape_log', 'gini_user',
                                'degree_assortativity_users',
                                'degree_assortativity_items',
                                    'most_least_favorite',
                                    #'highest_variance',
                                    #'most_characteristic',
                                    #'most_rated',
                                    'most_recent_1'
                                    ]

ALL_ACCEPTED_CHARACTERISTICS_MIN = ['space_size_log',
                            'density_log',
                            'gini_user',
                            'gini_item',
                            'degree_assortativity_items',
                            'most_least_favorite',
                            # 'highest_variance',
                            #'most_characteristic',
                            'most_rated',
                            'most_recent_1'
                            ]
'''
YELP MIN DEF
'space_size_log',
                            'density_log',
                            'gini_user',
                            'gini_item',
                            'degree_assortativity_items',
                            'most_least_favorite',
                            # 'highest_variance',
                            # 'most_characteristic',
                            'most_rated',
                            'most_recent_1'
'''


ALL_ACCEPTED_CHARACTERISTICS = ['space_size_log', 'shape_log', 'density_log',
                            'gini_user',
                            'gini_item',
                            'average_degree_users_log', 'average_degree_items_log',
                            'average_clustering_coefficient_dot_users_log',
                            'average_clustering_coefficient_dot_items_log',
                            'degree_assortativity_users',
                            'degree_assortativity_items',
                            'most_least_favorite',
                            'highest_variance',
                            'most_characteristic',
                            'most_rated',
                            'most_recent_1']


# ACCEPTED_CHARACTERISTICS = ['transactions', 'space_size', 'space_size_log', 'shape', 'shape_log', 'density', 'density_log',
#                         'gini_item', 'gini_user',
#                         # 'average_degree', 'average_degree_users', 'average_degree_items',
#                         'average_degree_log', 'average_degree_users_log', 'average_degree_items_log',
#                         # 'average_clustering_coefficient_dot',
#                         # 'average_clustering_coefficient_min',
#                         # 'average_clustering_coefficient_max',
#                         'average_clustering_coefficient_dot_log',
#                         'average_clustering_coefficient_min_log',
#                         'average_clustering_coefficient_max_log',
#                         # 'average_clustering_coefficient_dot_users', 'average_clustering_coefficient_dot_items',
#                         # 'average_clustering_coefficient_min_users', 'average_clustering_coefficient_min_items',
#                         # 'average_clustering_coefficient_max_users', 'average_clustering_coefficient_max_items',
#                         'average_clustering_coefficient_dot_users_log', 'average_clustering_coefficient_dot_items_log',
#                         'average_clustering_coefficient_min_users_log', 'average_clustering_coefficient_min_items_log',
#                         'average_clustering_coefficient_max_users_log', 'average_clustering_coefficient_max_items_log',
#                         'degree_assortativity_users', 'degree_assortativity_items', 'most_least_favorite', 'most_rated',
#                         # 'highest_variance',
#                         # 'most_recent_1', 'most_recent_2', 'most_recent_3',
#                         'most_characteristic']

TO_LOG =[
          'most_least_favorite',
          'highest_variance',
          'most_characteristic',
          'most_rated',
          'most_recent_1']
# TO_LOG =[]

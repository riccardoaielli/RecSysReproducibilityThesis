#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/02/2023

@author: Anonymized for blind review
"""

from SIGIR2022.KGCL_our_interface.KGCLDataReader import KGCLDataReader, KGCL_Add_KB_to_existing_split_DataReader
from SIGIR2022.LightGCN_our_interface.LightGCNDataReader import LightGCNReader
from SIGIR2022.KGCL_our_interface.KGCL_RecommenderWrapper import KGCL_RecommenderWrapper
from SIGIR2022.HAKG_our_interface.HAKGDataReader import HAKGDataReader

from optimize_all_baselines import _baseline_tune, _run_algorithm_hyperopt, ExperimentConfiguration, copy_reproduced_metadata_in_baseline_folder
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
from data_statistics import save_data_statistics
from Utils.ResultFolderLoader import ResultFolderLoader

import numpy as np
import os, traceback, argparse
import pandas as pd

from Evaluation.Evaluator import EvaluatorHoldout
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

from skopt.space import Real, Integer, Categorical

from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


def _run_algorithm_fixed_hyperparameters(experiment_configuration,
                                         knowledge_base_df,
                                         recommender_class,
                                         hyperparameters_dictionary,
                                         max_epochs_for_earlystopping,
                                         min_epochs_for_earlystopping,
                                         output_folder_path, use_gpu):
    """
    Train algorithm with the original hyperparameters
    1 - The model is trained with the maximum number of epochs
    2 - The model is trained by selecting the optimal number of epochs with earlystopping on the validation data

    :param experiment_configuration:
    :param hyperparameters_dictionary:
    :param output_folder_path:
    :param use_gpu:
    :return:
    """

    hyperparameters_dictionary = hyperparameters_dictionary.copy()

    # Specify the maximum number of epochs and the earlystopping hyperparameters
    hyperparameters_dictionary["epochs"] = max_epochs_for_earlystopping

    # Specify the minimum number of epochs and the earlystopping hyperparameters
    earlystopping_hyperparameters = {"validation_every_n": 5,
                                     "stop_on_validation": True,
                                     "lower_validations_allowed": 5,
                                     "evaluator_object": experiment_configuration.evaluator_validation_earlystopping,
                                     "validation_metric": experiment_configuration.metric_to_optimize,
                                     "epochs_min": min_epochs_for_earlystopping
                                     }

    hyperparameterSearch = SearchSingleCase(recommender_class,
                                            evaluator_validation = experiment_configuration.evaluator_validation,
                                            evaluator_test = experiment_configuration.evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [experiment_configuration.URM_train, knowledge_base_df],
        CONSTRUCTOR_KEYWORD_ARGS = {"use_gpu": use_gpu, "use_cython_sampler": True, "verbose": False},
        FIT_KEYWORD_ARGS = {},
        EARLYSTOPPING_KEYWORD_ARGS = earlystopping_hyperparameters)

    recommender_input_args_last_test = recommender_input_args.copy()
    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = experiment_configuration.URM_train_last_test

    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test = recommender_input_args_last_test,
                                fit_hyperparameters_values = hyperparameters_dictionary,
                                metric_to_optimize = experiment_configuration.metric_to_optimize,
                                cutoff_to_optimize = experiment_configuration.cutoff_to_optimize,
                                output_folder_path = output_folder_path,
                                output_file_name_root = recommender_class.RECOMMENDER_NAME,
                                resume_from_saved = experiment_configuration.resume_from_saved,
                                save_model = experiment_configuration.save_model,
                                evaluate_on_test = experiment_configuration.evaluate_on_test,
                                )





def run_this_algorithm_experiment(dataset_name,
                                  flag_baselines_tune = False,
                                  flag_article_default = False,
                                  flag_article_tune = False,
                                  flag_print_results = False):

    dataset_name, split_type = dataset_name

    result_folder_path = "result_experiments/{}/{}_{}/".format(ALGORITHM_NAME, dataset_name, split_type)
    data_folder_path = result_folder_path + "data/"
    baseline_folder_path = result_folder_path + "baselines/"
    this_model_folder_path = result_folder_path + "this_model/"

    if dataset_name in ["yelp2018", "amazon-book"] and split_type == "ours":
        assert False, "KB mapping may be inconsistent"
        LightGCN_result_folder_path = "result_experiments/{}/{}_{}/".format("LightGCN", dataset_name, split_type)
        LightGCN_data_folder_path = LightGCN_result_folder_path + "data/"
        baseline_folder_path = LightGCN_result_folder_path + "baselines/"

        # Take the collaborative data split already used for LightGCN and add the knowledge base
        LightGCN_dataset = LightGCNReader(dataset_name, LightGCN_data_folder_path, split_type)
        dataset = KGCL_Add_KB_to_existing_split_DataReader(dataset_name, LightGCN_dataset, data_folder_path)

    elif dataset_name == "yelp2018" and split_type == "original":
        HAKG_result_folder_path = "result_experiments/{}/{}_{}/".format("HAKG", dataset_name, split_type)
        HAKG_data_folder_path = HAKG_result_folder_path + "data/"
        baseline_folder_path = HAKG_result_folder_path + "baselines/"

        # Take the collaborative data split already used for HAKG and add the knowledge base
        # The collaborative data is the same but the KB is not.
        HAKG_dataset = HAKGDataReader(dataset_name, HAKG_data_folder_path)
        dataset = KGCL_Add_KB_to_existing_split_DataReader(dataset_name, HAKG_dataset, data_folder_path, freeze_split=False)

    else:
        dataset = KGCLDataReader(dataset_name, split_type, data_folder_path)


    print('Current dataset is: {}'.format(dataset_name))

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    knowledge_base_df = dataset.knowledge_base_df.copy()

    URM_train_last_test = URM_train + URM_validation

    # Ensure IMPLICIT data and disjoint test-train split
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    save_data_statistics(URM_train + URM_validation + URM_test,
                         dataset_name,
                         data_folder_path + "data_statistics")

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["Training data", "Test data"],
                         data_folder_path + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["Training data", "Test data"],
                               data_folder_path + "popularity_statistics")


    metric_to_optimize = 'RECALL'
    cutoff_to_optimize = 20
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]
    max_total_time = 14*24*60*60  # 14 days
    n_cases = 50
    n_processes = 3

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)


    experiment_configuration = ExperimentConfiguration(
             URM_train = URM_train,
             URM_train_last_test = URM_train_last_test,
             ICM_DICT = dataset.ICM_DICT,
             UCM_DICT = dataset.UCM_DICT,
             n_cases = n_cases,
             n_random_starts = int(n_cases/3),
             resume_from_saved = True,
             save_model = "best",
             evaluate_on_test = "best",
             evaluator_validation = evaluator_validation,
             KNN_similarity_to_report_list = KNN_similarity_to_report_list,
             evaluator_test = evaluator_test,
             max_total_time = max_total_time,
             evaluator_validation_earlystopping = evaluator_validation_earlystopping,
             metric_to_optimize = metric_to_optimize,
             cutoff_to_optimize = cutoff_to_optimize,
             model_folder_path = baseline_folder_path,
             n_processes = n_processes,
             )

    # recommender_instance = KGCL_RecommenderWrapper(URM_train, knowledge_base_df, use_gpu=False, verbose=True)
    # recommender_instance.fit(
    #     epochs=3,
    #     GNN_layers_K=3,
    #     batch_size=1024,
    #     embedding_size=64,
    #     GNN_dropout_rate=0.3,
    #     entities_per_head = 10,
    #     contrastive_loss_temperature_tau = 0.2,
    #     knowledge_graph_dropout_rate=0.5,
    #     user_interaction_dropout_rate=0.001,
    #     mix_ratio=0.75,
    #     uicontrast="WEIGHTED",
    #     l2_reg=1e-3,
    #     self_supervised_loss_weight= 0.1,
    #     sgd_mode="adam",
    #     learning_rate=1e-2,
    #     learning_rate_milestones=[1500, 2500],
    # )
    #
    # result_df, result_string = evaluator_test.evaluateRecommender(recommender_instance)
    # print(result_string)
    #
    # exit()

    ################################################################################################
    ######
    ######      REPRODUCED ALGORITHM
    ######
    
    use_gpu = True

    if dataset_name == 'amazon-book':
        dataset_hyperparameters = {
            "self_supervised_loss_weight": 0.1,
            "GNN_dropout_rate": 0.2,
            "entities_per_head": 10,
            "user_interaction_dropout_rate": 0.05,
            "mix_ratio": None,
            "uicontrast": "WEIGHTED",
            "l2_reg": 1e-4,
            "learning_rate_milestones": [1500, 2500],
        }

    elif dataset_name == 'MIND':
        dataset_hyperparameters = {
            "self_supervised_loss_weight": 0.06,
            "GNN_dropout_rate": 0.4,
            "entities_per_head": 6,
            "user_interaction_dropout_rate": 0.4,
            "mix_ratio": 0.6,
            "uicontrast": "WEIGHTED-MIX",
            "l2_reg": 1e-3,
            "learning_rate_milestones": [5, 10],
        }

    elif dataset_name == 'yelp2018':
        dataset_hyperparameters = {
            "self_supervised_loss_weight": 0.1,
            "GNN_dropout_rate": 0.2,
            "entities_per_head": 10,
            "user_interaction_dropout_rate": 0.1,
            "mix_ratio": None,
            "uicontrast": "WEIGHTED",
            "l2_reg": 1e-4,
            "learning_rate_milestones": [1500, 2500],
        }


    common_hyperparameters = {
        'embedding_size': 64,
        'learning_rate': 1e-3,
        'batch_size': 2048,
        "contrastive_loss_temperature_tau": 0.2,
        "sgd_mode": "Adam",
        'epochs': 1000,
        "GNN_layers_K": 3,
        "knowledge_graph_dropout_rate": 0.5,
    }

    assert len(dataset_hyperparameters.keys() & common_hyperparameters.keys()) == 0

    all_hyperparameters = common_hyperparameters.copy()
    all_hyperparameters.update(dataset_hyperparameters.copy())

    max_epochs_for_earlystopping = 1000
    min_epochs_for_earlystopping = 50

    if flag_article_default:

        for fold_index in range(1,2):

            try:
                fold_folder = this_model_folder_path + "{}/".format(fold_index)

                _run_algorithm_fixed_hyperparameters(experiment_configuration,
                                                     knowledge_base_df,
                                                     KGCL_RecommenderWrapper,
                                                     all_hyperparameters,
                                                     max_epochs_for_earlystopping,
                                                     min_epochs_for_earlystopping,
                                                     fold_folder,
                                                     use_gpu)

            except Exception as e:
                print("On recommender {} Exception {}".format(KGCL_RecommenderWrapper, str(e)))
                traceback.print_exc()


    # if flag_article_tune:
    #
    #     n_users, n_items = URM_train.shape
    #
    #     hyperparameters_range_dictionary = {
    #         "epochs": Categorical([max_epochs_for_earlystopping]),
    #         "batch_size": Categorical([64, 128, 256, 512, 1024, 2048, 4096]),
    #         "learning_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),
    #         "beta": Real(low=1e-1, high=1e2, prior="log-uniform"),
    #         "feature_type": Categorical(["smoothed", "both"]),
    #         "drop_out": Real(low=0.1, high=0.9, prior="log-uniform"),
    #         "latent_size": Integer(1, 200),
    #         "reg": Real(low=1e-6, high=1e-1, prior="log-uniform"),
    #
    #         # For both ratios:
    #         #   floor(ratio * user_size) and floor(smooth_ratio * item_size) must be >=1
    #         #   ratio must be <= 0.33 due to the lobpcg function
    #         "smooth_ratio": Real(low=1/(min(n_users, n_items) -1), high=0.32, prior="log-uniform"),
    #         "rough_ratio": Real(low=1/(min(n_users, n_items) -1), high=0.32, prior="log-uniform"),
    #         "loss_type": Categorical(["adaptive", "bpr"]),
    #
    #         "sgd_mode": Categorical(["sgd", "adagrad", "adam", "rmsprop"]),
    #     }
    #
    #     earlystopping_hyperparameters = {"validation_every_n": 5,
    #                                      "stop_on_validation": True,
    #                                      "lower_validations_allowed": 5,
    #                                      "evaluator_object": evaluator_validation_earlystopping,
    #                                      "validation_metric": metric_to_optimize,
    #                                      "epochs_min": min_epochs_for_earlystopping
    #                                      }
    #
    #     _run_algorithm_hyperopt(experiment_configuration,
    #                             KGCL_RecommenderWrapper,
    #                             hyperparameters_range_dictionary,
    #                             earlystopping_hyperparameters,
    #                             this_model_folder_path + "hyperopt/",
    #                             use_gpu)

    ################################################################################################
    ######
    ######      BASELINE ALGORITHMS
    ######

    if flag_baselines_tune and dataset_name != "yelp2018":
        _baseline_tune(experiment_configuration, baseline_folder_path)

    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:
        paper_results = pd.DataFrame(index=[cutoff_to_optimize], columns=['RECALL', 'NDCG'])

        if dataset_name == 'amazon-book':
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0793
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.1496

        elif dataset_name == 'MIND':
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0551
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.1073

        elif dataset_name == 'yelp2018':
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0493
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.0756
        else:
            paper_results = None


        reproduced_algorithm_list, base_algorithm_list = copy_reproduced_metadata_in_baseline_folder(KGCL_RecommenderWrapper,
                                                                                                     this_model_folder_path,
                                                                                                     baseline_folder_path,
                                                                                                     paper_results = paper_results if split_type == "original" else None,
                                                                                                     baselines_to_print = ResultFolderLoader._DEFAULT_BASE_ALGORITHM_LIST)

        result_loader = ResultFolderLoader(baseline_folder_path,
                                           base_algorithm_list = base_algorithm_list,
                                           other_algorithm_list = reproduced_algorithm_list,
                                           KNN_similarity_list=KNN_similarity_to_report_list,
                                           ICM_names_list=dataset.ICM_DICT.keys(),
                                           UCM_names_list=dataset.UCM_DICT.keys(),
                                           )

        result_loader.generate_latex_results(result_folder_path + "{}_{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, split_type, "article_metrics"),
                                             metrics_list=['RECALL', 'NDCG'],
                                             cutoffs_list=[cutoff_to_optimize],
                                             table_title=None,
                                             highlight_best=True)

        result_loader.generate_latex_results(
            result_folder_path + "{}_{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, split_type, "beyond_accuracy_metrics"),
            metrics_list=["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
            cutoffs_list=cutoff_list,
            table_title=None,
            highlight_best=True)

        result_loader.generate_latex_time_statistics(result_folder_path + "{}_{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, split_type, "time"),
                                                     n_evaluation_users = np.sum(np.ediff1d(URM_test.indptr) >= 1),
                                                     table_title=None)


if __name__ == '__main__':

    ALGORITHM_NAME = "KGCL"
    CONFERENCE_NAME = "SIGIR22"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type=bool, default=False)
    parser.add_argument('-a', '--article_default',      help="Train the reproduced model with article hyperparameters", type=bool, default=False)
    parser.add_argument('-t', '--article_tune',         help="Reproduced model hyperparameters search", type=bool, default=False)
    parser.add_argument('-p', '--print_results',        help="Print results", type=bool, default=False)

    input_flags = parser.parse_args()
    print(input_flags)

    # Reporting only the cosine similarity is enough
    KNN_similarity_to_report_list = ["cosine"]  # , "dice", "jaccard", "asymmetric", "tversky"]

    dataset_list = [("amazon-book", "original"),
                    #("amazon-book", "ours"),
                    ("yelp2018", "original"),
                    #("yelp2018", "ours"),
                    ("MIND", "original"),
                    ]

    for dataset_name in dataset_list:
        print ("Running dataset: {} {}".format(*dataset_name))
        run_this_algorithm_experiment(dataset_name,
                                   flag_baselines_tune = input_flags.baseline_tune,
                                   flag_article_default = input_flags.article_default,
                                   flag_article_tune = input_flags.article_tune,
                                   flag_print_results = input_flags.print_results,
                                   )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/06/2023

@author: Anonymized for blind review
"""

import torch
from optimize_all_baselines import _baseline_tune, _run_algorithm_hyperopt, ExperimentConfiguration, copy_reproduced_metadata_in_baseline_folder
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
from data_statistics import save_data_statistics
from Utils.ResultFolderLoader import ResultFolderLoader

import numpy as np
import os
import traceback
import argparse
import pandas as pd

from Evaluation.Evaluator import EvaluatorHoldout
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

from skopt.space import Real, Integer, Categorical

from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from WWW2023.BM3_our_interface.BM3DataReader import BM3DataReader
from WWW2023.BM3_our_interface.BM3_RecommenderWrapper import BM3_RecommenderWrapper


def _run_algorithm_fixed_hyperparameters(experiment_configuration,
                                         train_data,
                                         test_data,
                                         valid_data,
                                         config,
                                         recommender_class,
                                         hyperparameters_dictionary,
                                         max_epochs_for_earlystopping,
                                         min_epochs_for_earlystopping,
                                         output_folder_path,
                                         use_gpu):
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

    # The SearchSingleCase object will train the method and evaluate it, saving all data in a zip folder
    hyperparameterSearch = SearchSingleCase(recommender_class,  # Addestra il meteodo sul training lo valuta, lo riaddestra su training e validation e infine valuta sul test
                                            evaluator_validation=experiment_configuration.evaluator_validation,
                                            evaluator_test=experiment_configuration.evaluator_test)

    # This data structure contains the attributes needed to create an instance of the recommender, additional
    # attributes needed by the fit function but that are not hyperparameters to tune and earlystopping hyperparameters
    recommender_input_args = SearchInputRecommenderArgs(
        # Mettere qua i parametri ossia le strutture dati che mi servono (grafo, edgeloaders) da passare al wrapper e in modo simile anche per our_interface
        CONSTRUCTOR_POSITIONAL_ARGS=[
            experiment_configuration.URM_train, config, train_data, test_data, valid_data],
        CONSTRUCTOR_KEYWORD_ARGS={"use_gpu": use_gpu, "verbose": False},
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={})

    # A copy of the SearchInputRecommenderArgs is needed to specify that the "last" instance will be trained
    # on the union of training and validation data. If earlystopping is used, the "last" instance will be trained for the
    # selected number of epochs. The "last" model will be evaluated on the test data and the results saved in a zip file.
    # Copio l'oggetto e sostituisco il primo argomento alla riga successiva
    recommender_input_args_last_test = recommender_input_args.copy()
    # Unione di train e validation
    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        0] = experiment_configuration.URM_train_last_test

    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test=recommender_input_args_last_test,
                                fit_hyperparameters_values=hyperparameters_dictionary,
                                metric_to_optimize=experiment_configuration.metric_to_optimize,
                                cutoff_to_optimize=experiment_configuration.cutoff_to_optimize,
                                output_folder_path=output_folder_path,
                                output_file_name_root=recommender_class.RECOMMENDER_NAME + "_no_earlystopping",
                                resume_from_saved=experiment_configuration.resume_from_saved,
                                save_model=experiment_configuration.save_model,
                                evaluate_on_test=experiment_configuration.evaluate_on_test,
                                )

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
                                            evaluator_validation=experiment_configuration.evaluator_validation,
                                            evaluator_test=experiment_configuration.evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(  # Stessa cosa di riga 58 ma con early stopping
        CONSTRUCTOR_POSITIONAL_ARGS=[
            experiment_configuration.URM_train, config, train_data, test_data, valid_data],
        CONSTRUCTOR_KEYWORD_ARGS={"use_gpu": use_gpu, "verbose": False},
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=earlystopping_hyperparameters)  # Dizionario di iperparametri con cui fa early stopping

    recommender_input_args_last_test = recommender_input_args.copy()
    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        0] = experiment_configuration.URM_train_last_test

    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test=recommender_input_args_last_test,
                                fit_hyperparameters_values=hyperparameters_dictionary,
                                metric_to_optimize=experiment_configuration.metric_to_optimize,
                                cutoff_to_optimize=experiment_configuration.cutoff_to_optimize,
                                output_folder_path=output_folder_path,
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,
                                resume_from_saved=experiment_configuration.resume_from_saved,
                                save_model=experiment_configuration.save_model,
                                evaluate_on_test=experiment_configuration.evaluate_on_test,
                                )


def run_this_algorithm_experiment(dataset_name,
                                  flag_baselines_tune=False,
                                  flag_article_default=False,
                                  flag_article_tune=False,
                                  flag_print_results=False):

    # Definisco le cartelle in cui mettere quello che mi serve
    result_folder_path = "result_experiments/{}/{}/".format(
        ALGORITHM_NAME, dataset_name)
    data_folder_path = result_folder_path + "data/"
    baseline_folder_path = result_folder_path + "baselines/"
    this_model_folder_path = result_folder_path + "this_model/"

    use_gpu = False

    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            print("GPU is not available, using cpu")
            device = torch.device("cpu:0")
    else:
        print("GPU is not available, using cpu")
        device = torch.device("cpu:0")

    print('device: ', device)

    dataset = BM3DataReader(dataset_name, data_folder_path, device)

    print('Current dataset is: {}'.format(dataset_name))

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    train_data = dataset.train_data
    test_data = dataset.test_data
    valid_data = dataset.valid_data
    config = dataset.config

    config['device'] = device

    URM_train_last_test = URM_train + URM_validation

    # Ensure IMPLICIT data and disjoint test-train split
    # Verifica che i dati siano impliciti
    assert_implicit_data([URM_train, URM_validation, URM_test])
    # Verifica che i dati siano disgiunti
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    save_data_statistics(URM_train + URM_validation + URM_test,  # Consentono di capire le distribuzioni degli item
                         dataset_name,
                         data_folder_path + "data_statistics")

    plot_popularity_bias([URM_train + URM_validation, URM_test],  # Consentono di capire le distribuzioni degli item
                         ["Training data", "Test data"],
                         data_folder_path + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["Training data", "Test data"],
                               data_folder_path + "popularity_statistics")

    # Nell'articolo anno usato RECALL per fare l'hyperparameter sensitivity study
    metric_to_optimize = 'RECALL'
    # Dato originale dell'articolo
    cutoff_to_optimize = 20
    # Liste su cui il modello viene valutato (questa resta sempre uguale)
    cutoff_list = [1, 5, 10, 20, 30, 40, 50, 100]
    max_total_time = 14*24*60*60  # 14 days # Tempo massimo di training
    n_cases = 50  # Numero di iperparametri che vengono valutati
    n_processes = 3

    # Usato per fare evaluation sul validation set
    evaluator_validation = EvaluatorHoldout(
        URM_validation, cutoff_list=cutoff_list)
    # Uguale a quello sopra ma con un solo cutoff in modo da fare early stopping
    evaluator_validation_earlystopping = EvaluatorHoldout(
        URM_validation, cutoff_list=[cutoff_to_optimize])
    # Usato per fare evaluation sul test set
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # Oggetto che contiene tutta la configurazione sperimentale in un unico oggetto che viene spostato in diverse parti
    experiment_configuration = ExperimentConfiguration(
        URM_train=URM_train,
        URM_train_last_test=URM_train_last_test,
        ICM_DICT=dataset.ICM_DICT,
        UCM_DICT=dataset.UCM_DICT,
        n_cases=n_cases,
        n_random_starts=int(n_cases/3),
        resume_from_saved=True,
        save_model="best",
        evaluate_on_test="best",
        evaluator_validation=evaluator_validation,
        KNN_similarity_to_report_list=KNN_similarity_to_report_list,
        evaluator_test=evaluator_test,
        max_total_time=max_total_time,
        evaluator_validation_earlystopping=evaluator_validation_earlystopping,
        metric_to_optimize=metric_to_optimize,
        cutoff_to_optimize=cutoff_to_optimize,
        model_folder_path=baseline_folder_path,
        n_processes=n_processes,
    )

    ################################################################################################
    ######
    # REPRODUCED ALGORITHM
    # Sezione che continene i valori degli iperparametri usati nell'articolo per ciascun dataset

    # Potrei non usarla come struttura in quanto non dobbiamo fare tuning degli iperparametri
    all_hyperparameters = {
        'lr': config['learning_rate'],
        'dropout': config['dropout'][0],
        'epochs': config['epochs'],
        'embedding_size': config['embedding_size'],  # hidden size
        'reg': config['reg_weight'][0],  # lambda

    }

    # Max uguale a quanto detto nel paper e min a zero epochs
    max_epochs_for_earlystopping = config['epochs']
    min_epochs_for_earlystopping = 0

    if flag_article_default:

        try:

            # Funzione che esegue il modello con gli iperparametri settati
            _run_algorithm_fixed_hyperparameters(experiment_configuration,
                                                 train_data,
                                                 test_data,
                                                 valid_data,
                                                 config,
                                                 BM3_RecommenderWrapper,  # Classe del metodo che deve eseguire
                                                 all_hyperparameters,
                                                 max_epochs_for_earlystopping,
                                                 min_epochs_for_earlystopping,
                                                 this_model_folder_path,
                                                 use_gpu)

        except Exception as e:
            print("On recommender {} Exception {}".format(
                BM3_RecommenderWrapper, str(e)))
            traceback.print_exc()

    ## NON DA FARE ##
    if flag_article_tune:

        n_users, n_items = URM_train.shape

        # Reducing the maximum percentage of "features" for the eigenvalue-eigenvector decomposition if the dataset
        # is large to avoid memory errors. The available memory on the GPU is 24GB
        ratio_high = 0.32

        hyperparameters_range_dictionary = {
            "epochs": Categorical([max_epochs_for_earlystopping]),
            "batch_size": Categorical([64, 128, 256, 512, 1024, 2048, 4096]),
            "learning_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),
            "beta": Real(low=1e-1, high=1e2, prior="log-uniform"),
            "feature_type": Categorical(["smoothed", "both"]),
            "drop_out": Real(low=0.1, high=0.9, prior="log-uniform"),
            "embedding_size": Integer(1, 200),
            "reg": Real(low=1e-6, high=1e-1, prior="log-uniform"),

            # For both ratios:
            #   floor(ratio * user_size) and floor(smooth_ratio * item_size) must be >=1
            #   ratio must be <= 0.33 due to the lobpcg function
            "smooth_ratio": Real(low=1/(min(n_users, n_items) - 1), high=ratio_high, prior="log-uniform"),
            "rough_ratio": Real(low=1/(min(n_users, n_items) - 1), high=ratio_high, prior="log-uniform"),
            "loss_type": Categorical(["adaptive", "bpr"]),

            "sgd_mode": Categorical(["sgd", "adagrad", "adam", "rmsprop"]),
        }

        earlystopping_hyperparameters = {"validation_every_n": 5,
                                         "stop_on_validation": True,
                                         "lower_validations_allowed": 5,
                                         "evaluator_object": evaluator_validation_earlystopping,
                                         "validation_metric": metric_to_optimize,
                                         "epochs_min": min_epochs_for_earlystopping
                                         }

        _run_algorithm_hyperopt(experiment_configuration,
                                BM3_RecommenderWrapper,
                                hyperparameters_range_dictionary,
                                earlystopping_hyperparameters,
                                this_model_folder_path + "hyperopt/",
                                use_gpu)

    ################################################################################################
    ######
    # BASELINE ALGORITHMS
    ######

    if flag_baselines_tune:
        # Ignora come è scritta. Continene la lista dei modelli che si vogliono usare come baseline del modello, deve essere ridotta a un Top popular, item knn, pure svd e matrix factorization
        _baseline_tune(experiment_configuration, baseline_folder_path)

    ################################################################################################
    ######
    # PRINT RESULTS
    ######

    if flag_print_results:
        paper_results = pd.DataFrame(index=[cutoff_to_optimize], columns=[
                                     'RECALL', 'NDCG'])

        if dataset_name == 'baby':
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.0883
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0383
        elif dataset_name == 'elec':
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.0648
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0302
        elif dataset_name == 'sports':
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.0980
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0438
        else:
            paper_results = None

        reproduced_algorithm_list, base_algorithm_list = copy_reproduced_metadata_in_baseline_folder(BM3_RecommenderWrapper,
                                                                                                     this_model_folder_path,
                                                                                                     baseline_folder_path,
                                                                                                     paper_results=paper_results,
                                                                                                     baselines_to_print=ResultFolderLoader._DEFAULT_BASE_ALGORITHM_LIST)

        result_loader = ResultFolderLoader(baseline_folder_path,
                                           base_algorithm_list=base_algorithm_list,
                                           other_algorithm_list=reproduced_algorithm_list,
                                           KNN_similarity_list=KNN_similarity_to_report_list,
                                           ICM_names_list=dataset.ICM_DICT.keys(),
                                           UCM_names_list=dataset.UCM_DICT.keys(),
                                           )

        result_loader.generate_latex_results(result_folder_path + "{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, "article_metrics"),
                                             metrics_list=[
                                                 'RECALL', 'NDCG'],
                                             cutoffs_list=[cutoff_to_optimize],
                                             table_title=None,
                                             highlight_best=True)

        result_loader.generate_latex_results(
            result_folder_path + "{}_{}_{}_latex_results.txt".format(
                ALGORITHM_NAME, dataset_name, "beyond_accuracy_metrics"),
            metrics_list=["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "COVERAGE_ITEM_HIT",
                          "DIVERSITY_GINI", "SHANNON_ENTROPY"],  # TODO sistema beyond_accuracy_metrics
            cutoffs_list=[10, 20],
            table_title=None,
            highlight_best=True)

        result_loader.generate_latex_time_statistics(result_folder_path + "{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, "time"),
                                                     n_evaluation_users=np.sum(
                                                         np.ediff1d(URM_test.indptr) >= 1),
                                                     table_title=None)


if __name__ == '__main__':

    ALGORITHM_NAME = "BM3"
    CONFERENCE_NAME = "WWW2023"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',
                        help="Baseline hyperparameter search", action='store_true')
    parser.add_argument('-a', '--article_default',
                        help="Train the reproduced model with article hyperparameters", action='store_true')
    parser.add_argument('-t', '--article_tune',
                        help="Reproduced model hyperparameters search", action='store_true')
    parser.add_argument('-p', '--print_results',
                        help="Print results", action='store_true')

    input_flags = parser.parse_args()
    print(input_flags)

    # Reporting only the cosine similarity is enough
    # , "dice", "jaccard", "asymmetric", "tversky"]
    KNN_similarity_to_report_list = ["cosine"]

    # , 'elec', 'sports'] # TODO includi tutti i dataset, elec è un dataset troppo grande e con le risorse computazionali a disposizione non rieco a fare il training quindi lo escludo
    dataset_list = ['baby', 'sports']

    for dataset_name in dataset_list:
        print("Running dataset: {}".format(dataset_name))
        run_this_algorithm_experiment(dataset_name,
                                      flag_baselines_tune=input_flags.baseline_tune,
                                      flag_article_default=input_flags.article_default,
                                      flag_article_tune=input_flags.article_tune,
                                      flag_print_results=input_flags.print_results,
                                      )

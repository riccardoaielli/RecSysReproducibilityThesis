#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/06/2023

@author: Anonymized for blind review
"""

from optimize_all_baselines import _baseline_tune, _run_algorithm_hyperopt, ExperimentConfiguration, copy_reproduced_metadata_in_baseline_folder
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
from data_statistics import save_data_statistics
from Utils.ResultFolderLoader import ResultFolderLoader

import numpy as np
import os
import traceback
import argparse
import pandas as pd
import torch

from Evaluation.Evaluator import EvaluatorHoldout
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

from skopt.space import Real, Integer, Categorical

from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from WWW2023.MMSSL_our_interface.MMSSLDataReader import MMSSLDataReader
from WWW2023.MMSSL_our_interface.MMSSL_RecommenderWrapper import MMSSL_RecommenderWrapper


def _run_algorithm_fixed_hyperparameters(experiment_configuration,
                                         image_feats,
                                         text_feats,
                                         image_feat_dim,
                                         text_feat_dim,
                                         ui_graph,
                                         exist_users,
                                         train_set,
                                         config,
                                         n_train,
                                         ui_graph_last_test,
                                         n_train_last_test,
                                         train_set_last_test,
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
            experiment_configuration.URM_train, image_feats, text_feats, image_feat_dim, text_feat_dim, ui_graph, exist_users, train_set, config, n_train],
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

    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        5] = ui_graph_last_test

    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        7] = train_set_last_test

    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        9] = n_train_last_test

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
        CONSTRUCTOR_POSITIONAL_ARGS=[experiment_configuration.URM_train, image_feats,
                                     text_feats, image_feat_dim, text_feat_dim, ui_graph, exist_users, train_set, config, n_train],
        CONSTRUCTOR_KEYWORD_ARGS={"use_gpu": use_gpu, "verbose": False},
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=earlystopping_hyperparameters)  # Dizionario di iperparametri con cui fa early stopping

    recommender_input_args_last_test = recommender_input_args.copy()
    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        0] = experiment_configuration.URM_train_last_test

    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        5] = ui_graph_last_test

    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        7] = train_set_last_test

    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[
        9] = n_train_last_test

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

    dataset = MMSSLDataReader(dataset_name, data_folder_path)

    print('Current dataset is: {}'.format(dataset_name))

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    image_feats = dataset.image_feats
    text_feats = dataset.text_feats
    image_feat_dim = dataset.image_feat_dim
    text_feat_dim = dataset.text_feat_dim
    ui_graph = dataset.ui_graph
    n_users = dataset.n_users
    n_items = dataset.n_items
    n_train = dataset.n_train
    n_val = dataset.n_val
    exist_users = dataset.exist_users
    train_set = dataset.train_set
    train_set_last_test = dataset.train_set_last_test

    URM_train_last_test = URM_train + URM_validation

    ui_graph_last_test = URM_train_last_test

    n_train_last_test = n_train + n_val

    # Verifica che i dati siano impliciti
    assert_implicit_data([URM_train, URM_validation, URM_test])
    # Verifica che i dati siano disgiunti
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    # If directory does not exist, create
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    save_data_statistics(URM_train + URM_validation + URM_test,  # Consentono di capire le distribuzioni degli item
                         dataset_name,
                         data_folder_path + "data_statistics")

    plot_popularity_bias([URM_train + URM_validation, URM_test],  # Consentono di capire le distribuzioni degli item
                         ["Training data", "Test data"],
                         data_folder_path + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["Training data", "Test data"],
                               data_folder_path + "popularity_statistics")

    # Non specificato metto NDCG@20
    metric_to_optimize = 'NDCG'
    # Dato originale dell'articolo, non è specificato che cutoff to optimize hanno usato, metto 20
    cutoff_to_optimize = 20
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

    use_gpu = True  # TODO gpu da mettere a True

    config = dict()
    config['n_users'] = n_users
    config['n_items'] = n_items

    # useless
    config['verbose'] = True
    config['lambda_coeff'] = 0.9
    config['early_stopping_patience'] = 7
    # default 1, best 2 o 3, dalle analisi 2 sembra migliore
    config['layers'] = 2
    config['mess_dropout'] = [0.1, 0.1]
    config['sparse'] = 1
    config['test_flag'] = 'part'
    config['metapath_threshold'] = 2
    config['sc'] = 1.0
    config['ssl_c_rate'] = 1.3
    config['ssl_s_rate'] = 0.8
    config['g_rate'] = 0.000029
    config['sample_num'] = 1
    config['sample_num_neg'] = 1
    config['sample_num_ii'] = 8
    config['sample_num_co'] = 2
    config['mask_rate'] = 0.75
    config['gss_rate'] = 0.85
    config['anchor_rate'] = 0.75
    config['feat_reg_decay'] = 1e-5
    config['ad1_rate'] = 0.2
    config['ad2_rate'] = 0.2
    config['ad_sampNum'] = 1
    config['ad_topk_multi_num'] = 100
    config['fake_gene_rate'] = 0.0001
    config['ID_layers'] = 1
    config['reward_rate'] = 1
    config['G_embed_size'] = 64
    config['model_num'] = 2
    config['negrate'] = 0.01
    config['cis'] = 25
    config['confidence'] = 0.5
    config['ii_it'] = 15
    config['isload'] = False
    config['isJustTest'] = False

    # train
    config['seed'] = 2022
    config['epoch'] = 400  # TODO default epochs 1000
    config['embed_size'] = 64
    config['batch_size'] = 1024
    config['D_lr'] = 3e-4
    config['topk'] = 10
    config['cf_model'] = 'slmrec'
    config['cl_rate'] = 0.03
    config['norm_type'] = 'sym'
    config['Ks'] = [10, 20, 50]
    config['regs'] = [1e-5, 1e-5, 1e-2]
    config['lr'] = 0.00055
    config['emm'] = 1e-3
    config['L2_alpha'] = 1e-3
    config['weight_decay'] = 1e-4

    # GNN
    config['drop_rate'] = 0.2
    config['model_cat_rate'] = 0.55
    config['gnn_cat_rate'] = 0.55
    config['id_cat_rate'] = 0.36
    config['id_cat_rate1'] = 0.36
    config['head_num'] = 4
    config['dgl_nei_num'] = 8

    # GAN
    config['weight_size'] = [64, 64]
    config['G_rate'] = 0.0001
    config['G_drop1'] = 0.31
    config['G_drop2'] = 0.5
    config['gp_rate'] = 1
    config['real_data_tau'] = 0.005
    config['ui_pre_scale'] = 100

    # cl
    config['T'] = 1
    config['tau'] = 0.085  # default 0.5, best secondo il paper 0.085
    config['geneGraph_rate'] = 0.1
    config['geneGraph_rate_pos'] = 2
    config['geneGraph_rate_neg'] = -1
    config['m_topk_rate'] = 0.0001
    config['log_log_scale'] = 0.00001

    all_hyperparameters = {
        'epochs': config['epoch'],
    }

    max_epochs_for_earlystopping = config['epoch']
    min_epochs_for_earlystopping = 0

    if flag_article_default:

        try:

            # Funzione che esegue il modello con gli iperparametri settati
            _run_algorithm_fixed_hyperparameters(experiment_configuration,
                                                 image_feats,
                                                 text_feats,
                                                 image_feat_dim,
                                                 text_feat_dim,
                                                 ui_graph,
                                                 exist_users,
                                                 train_set,
                                                 config,
                                                 n_train,
                                                 ui_graph_last_test,
                                                 n_train_last_test,
                                                 train_set_last_test,
                                                 MMSSL_RecommenderWrapper,  # Classe del metodo che deve eseguire
                                                 all_hyperparameters,
                                                 max_epochs_for_earlystopping,
                                                 min_epochs_for_earlystopping,
                                                 this_model_folder_path,
                                                 use_gpu)

        except Exception as e:
            print("On recommender {} Exception {}".format(
                MMSSL_RecommenderWrapper, str(e)))
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
                                MMSSL_RecommenderWrapper,
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
                                     'RECALL', 'PRECISION', 'NDCG'])

        if dataset_name == "baby":
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.0962
            paper_results.loc[cutoff_to_optimize, 'PRECISION'] = 0.0051
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0422

        elif dataset_name == "sports":
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.0998
            paper_results.loc[cutoff_to_optimize, 'PRECISION'] = 0.0052
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0470

        elif dataset_name == "allrecipes":
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.1277
            paper_results.loc[cutoff_to_optimize, 'PRECISION'] = 0.1277
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0879

        elif dataset_name == "tiktok":
            paper_results.loc[cutoff_to_optimize, 'RECALL'] = 0.0367
            paper_results.loc[cutoff_to_optimize, 'PRECISION'] = 0.0018
            paper_results.loc[cutoff_to_optimize, 'NDCG'] = 0.0135

        else:
            paper_results = None

        reproduced_algorithm_list, base_algorithm_list = copy_reproduced_metadata_in_baseline_folder(MMSSL_RecommenderWrapper,
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
                                                 'RECALL', 'PRECISION', 'NDCG'],
                                             cutoffs_list=[cutoff_to_optimize],
                                             table_title=None,
                                             highlight_best=True)

        result_loader.generate_latex_results(
            result_folder_path + "{}_{}_{}_latex_results.txt".format(
                ALGORITHM_NAME, dataset_name, "beyond_accuracy_metrics"),
            metrics_list=["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "COVERAGE_ITEM_HIT",
                          "DIVERSITY_GINI", "SHANNON_ENTROPY"],
            cutoffs_list=[20],
            table_title=None,
            highlight_best=True)

        result_loader.generate_latex_time_statistics(result_folder_path + "{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, "time"),
                                                     n_evaluation_users=np.sum(
                                                         np.ediff1d(URM_test.indptr) >= 1),
                                                     table_title=None)


if __name__ == '__main__':

    ALGORITHM_NAME = "MMSSL"
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

    # Forse tiktok ha biosgno di caricare anche audio_feat, non è impementato nel modello originale
    # quindi ignorerei le audio feautures
    # , "sports", "baby", "allrecipes"]  # TODO lista datasets
    dataset_list = ["tiktok"]  # , "baby", "sports", "allrecipes"]

    for dataset_name in dataset_list:
        print("Running dataset: {}".format(dataset_name))
        run_this_algorithm_experiment(dataset_name,
                                      flag_baselines_tune=input_flags.baseline_tune,
                                      flag_article_default=input_flags.article_default,
                                      flag_article_tune=input_flags.article_tune,
                                      flag_print_results=input_flags.print_results,
                                      )

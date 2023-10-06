#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/12/2022

@author: Anonymized for blind review
"""


import pandas as pd
import shutil
from collections import namedtuple
import traceback
# import mkl
from Recommenders.Recommender_import_list import *
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from functools import partial
import numpy as np
import os
import traceback
import argparse
import multiprocessing
from Recommenders.DataIO import DataIO


def _get_model_list(recommender_class_list, KNN_similarity_list, ICM_dict, UCM_dict):

    recommender_class_list = recommender_class_list.copy()

    # Model list format: recommender class, KNN heuristic, ICM/UCM name, ICM/UCM matrix
    model_list = []

    for recommender_class in recommender_class_list:

        if issubclass(recommender_class, BaseItemCBFRecommender):
            for ICM_name, ICM_object in ICM_dict.items():
                if recommender_class in [ItemKNNCBFRecommender, ItemKNN_CFCBF_Hybrid_Recommender]:
                    for KNN_similarity in KNN_similarity_list:
                        model_list.append(
                            (recommender_class, KNN_similarity, ICM_name, ICM_object))
                else:
                    model_list.append(
                        (recommender_class, None, ICM_name, ICM_object))

        elif issubclass(recommender_class, BaseUserCBFRecommender):
            for UCM_name, UCM_object in UCM_dict.items():
                if recommender_class in [UserKNNCBFRecommender, UserKNN_CFCBF_Hybrid_Recommender]:
                    for KNN_similarity in KNN_similarity_list:
                        model_list.append(
                            (recommender_class, KNN_similarity, UCM_name, UCM_object))
                else:
                    model_list.append(
                        (recommender_class, None, UCM_name, UCM_object))

        else:
            if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:
                for KNN_similarity in KNN_similarity_list:
                    model_list.append(
                        (recommender_class, KNN_similarity, None, None))

            else:
                model_list.append((recommender_class, None, None, None))

    return model_list


def _optimize_single_model(model_tuple, URM_train, URM_train_last_test=None,
                           n_cases=None, n_random_starts=None, resume_from_saved=False,
                           save_model="best", evaluate_on_test="best", max_total_time=None,
                           evaluator_validation=None, evaluator_test=None, evaluator_validation_earlystopping=None,
                           metric_to_optimize=None, cutoff_to_optimize=None,
                           model_folder_path="result_experiments/"):

    try:

        recommender_class, KNN_similarity, ICM_UCM_name, ICM_UCM_object = model_tuple

        if recommender_class in [ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender,
                                 LightFMUserHybridRecommender, LightFMItemHybridRecommender]:
            runHyperparameterSearch_Hybrid(recommender_class,
                                           URM_train=URM_train,
                                           URM_train_last_test=URM_train_last_test,
                                           metric_to_optimize=metric_to_optimize,
                                           cutoff_to_optimize=cutoff_to_optimize,
                                           evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                           evaluator_validation=evaluator_validation,
                                           similarity_type_list=[
                                               KNN_similarity],
                                           evaluator_test=evaluator_test,
                                           max_total_time=max_total_time,
                                           output_folder_path=model_folder_path,
                                           parallelizeKNN=False,
                                           allow_weighting=True,
                                           save_model=save_model,
                                           evaluate_on_test=evaluate_on_test,
                                           resume_from_saved=resume_from_saved,
                                           ICM_name=ICM_UCM_name,
                                           ICM_object=ICM_UCM_object.copy(),
                                           n_cases=n_cases,
                                           n_random_starts=n_random_starts)

        elif issubclass(recommender_class, BaseItemCBFRecommender) or issubclass(recommender_class, BaseUserCBFRecommender):
            runHyperparameterSearch_Content(recommender_class,
                                            URM_train=URM_train,
                                            URM_train_last_test=URM_train_last_test,
                                            metric_to_optimize=metric_to_optimize,
                                            cutoff_to_optimize=cutoff_to_optimize,
                                            evaluator_validation=evaluator_validation,
                                            similarity_type_list=[
                                                KNN_similarity],
                                            evaluator_test=evaluator_test,
                                            output_folder_path=model_folder_path,
                                            parallelizeKNN=False,
                                            allow_weighting=True,
                                            save_model=save_model,
                                            evaluate_on_test=evaluate_on_test,
                                            max_total_time=max_total_time,
                                            resume_from_saved=resume_from_saved,
                                            ICM_name=ICM_UCM_name,
                                            ICM_object=ICM_UCM_object.copy(),
                                            n_cases=n_cases,
                                            n_random_starts=n_random_starts)

        else:

            runHyperparameterSearch_Collaborative(recommender_class, URM_train=URM_train,
                                                  URM_train_last_test=URM_train_last_test,
                                                  metric_to_optimize=metric_to_optimize,
                                                  cutoff_to_optimize=cutoff_to_optimize,
                                                  evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                  evaluator_validation=evaluator_validation,
                                                  similarity_type_list=[
                                                      KNN_similarity],
                                                  evaluator_test=evaluator_test,
                                                  max_total_time=max_total_time,
                                                  output_folder_path=model_folder_path,
                                                  resume_from_saved=resume_from_saved,
                                                  parallelizeKNN=False,
                                                  allow_weighting=True,
                                                  save_model=save_model,
                                                  evaluate_on_test=evaluate_on_test,
                                                  n_cases=n_cases,
                                                  n_random_starts=n_random_starts)

    except Exception as e:
        print("On CBF recommender {} Exception {}".format(
            model_tuple[0], str(e)))
        traceback.print_exc()


def copy_reproduced_metadata_in_baseline_folder(recommender_class, this_model_folder_path, baseline_folder_path, paper_results=None, baselines_to_print=None):

    # Use named tuple because the object used to read results expects the field RECOMMENDER_NAME to be available
    recommender_tuple = namedtuple(
        recommender_class.RECOMMENDER_NAME, "RECOMMENDER_NAME")

    other_algorithm_list = []

    # Copy results of reproduced paper to allow the creation of the result table
    for label in [recommender_class.RECOMMENDER_NAME,
                  "{}_no_earlystopping".format(
                      recommender_class.RECOMMENDER_NAME),
                  "{}_original_earlystopping".format(
                      recommender_class.RECOMMENDER_NAME),
                  "{}_ours_earlystopping".format(recommender_class.RECOMMENDER_NAME)]:
        try:
            shutil.copyfile("{}/{}_metadata.zip".format(this_model_folder_path, label),
                            "{}/{}_metadata.zip".format(baseline_folder_path, label))
            other_algorithm_list.append(recommender_tuple(label))
        except FileNotFoundError:
            pass

    try:
        shutil.copyfile("{}/{}/{}_metadata.zip".format(this_model_folder_path, "hyperopt", recommender_class.RECOMMENDER_NAME),
                        "{}/{}_metadata.zip".format(baseline_folder_path, recommender_class.RECOMMENDER_NAME + "_hyperopt"))
        other_algorithm_list.append(recommender_tuple(
            recommender_class.RECOMMENDER_NAME + "_hyperopt"))
    except FileNotFoundError:
        pass

    if paper_results is not None:
        dataIO = DataIO(baseline_folder_path)

        data_dict_to_save = {
            "result_on_last": paper_results,
            "time_df": pd.DataFrame(columns=["train", "validation", "test"], index=np.arange(1)),
            "hyperparameters_best_index": None,
        }

        label = "{}_paper".format(recommender_class.RECOMMENDER_NAME)
        dataIO.save_data(label + "_metadata.zip",
                         data_dict_to_save=data_dict_to_save)

        baselines_to_print = baselines_to_print.copy()
        baselines_to_print.extend([None, recommender_tuple(label)])

    return other_algorithm_list, baselines_to_print


class ExperimentConfiguration(object):

    def __init__(self,
                 URM_train=None,
                 URM_train_last_test=None,
                 ICM_DICT=None,
                 UCM_DICT=None,
                 n_cases=None,
                 n_random_starts=None,
                 resume_from_saved=None,
                 save_model=None,
                 evaluate_on_test=None,
                 evaluator_validation=None,
                 KNN_similarity_to_report_list=None,
                 evaluator_test=None,
                 max_total_time=None,
                 evaluator_validation_earlystopping=None,
                 metric_to_optimize=None,
                 cutoff_to_optimize=None,
                 model_folder_path=None,
                 n_processes=None,
                 ):
        super(ExperimentConfiguration, self).__init__()

        self.URM_train = URM_train
        self.URM_train_last_test = URM_train_last_test
        self.ICM_DICT = ICM_DICT
        self.UCM_DICT = UCM_DICT
        self.n_cases = n_cases
        self.n_random_starts = n_random_starts
        self.resume_from_saved = resume_from_saved
        self.save_model = save_model
        self.evaluate_on_test = evaluate_on_test
        self.evaluator_validation = evaluator_validation
        self.KNN_similarity_to_report_list = KNN_similarity_to_report_list
        self.evaluator_test = evaluator_test
        self.max_total_time = max_total_time
        self.evaluator_validation_earlystopping = evaluator_validation_earlystopping
        self.metric_to_optimize = metric_to_optimize
        self.cutoff_to_optimize = cutoff_to_optimize
        self.model_folder_path = model_folder_path
        self.n_processes = n_processes


def _baseline_tune(experiment_configuration, output_folder_path):

    # Lsciare decommentate solo le baseline che ho deciso di utilizzare ossia un Top popular, item knn, pure svd e matrix factorization
    recommender_class_list = [
        # Random,
        TopPop,
        # GlobalEffects,
        # SLIMElasticNetRecommender,
        # UserKNNCFRecommender,
        MatrixFactorization_BPR_Cython,
        # MatrixFactorization_WARP_Cython,
        # IALSRecommender,
        # MatrixFactorization_SVDpp_Cython,
        # # MatrixFactorization_AsySVD_Cython,
        # EASE_R_Recommender,
        ItemKNNCFRecommender,
        # P3alphaRecommender,
        # SLIM_BPR_Cython,
        RP3betaRecommender,
        PureSVDRecommender,
        # NMFRecommender,
        # UserKNNCBFRecommender,
        # ItemKNNCBFRecommender,
        # UserKNN_CFCBF_Hybrid_Recommender,
        # ItemKNN_CFCBF_Hybrid_Recommender,
        # LightFMCFRecommender,
        # LightFMUserHybridRecommender,
        # LightFMItemHybridRecommender,
        # NegHOSLIMRecommender,
        # NegHOSLIMElasticNetRecommender,
        # MultVAERecommender,
        GraphFilterCFRecommender,
        # ItemRankRecommender,
        # ItemRankSVDRecommender,
    ]

    model_cases_list = _get_model_list(recommender_class_list,
                                       experiment_configuration.KNN_similarity_to_report_list,
                                       experiment_configuration.ICM_DICT,
                                       experiment_configuration.UCM_DICT)

    _optimize_single_model_partial = partial(_optimize_single_model,
                                             URM_train=experiment_configuration.URM_train,
                                             URM_train_last_test=experiment_configuration.URM_train_last_test,
                                             n_cases=experiment_configuration.n_cases,
                                             n_random_starts=experiment_configuration.n_random_starts,
                                             resume_from_saved=experiment_configuration.resume_from_saved,
                                             save_model=experiment_configuration.save_model,
                                             evaluate_on_test=experiment_configuration.evaluate_on_test,
                                             evaluator_validation=experiment_configuration.evaluator_validation,
                                             evaluator_test=experiment_configuration.evaluator_test,
                                             max_total_time=experiment_configuration.max_total_time,
                                             evaluator_validation_earlystopping=experiment_configuration.evaluator_validation_earlystopping,
                                             metric_to_optimize=experiment_configuration.metric_to_optimize,
                                             cutoff_to_optimize=experiment_configuration.cutoff_to_optimize,
                                             model_folder_path=output_folder_path)

    # mkl.set_num_threads(4)

    pool = multiprocessing.Pool(
        processes=experiment_configuration.n_processes, maxtasksperchild=1)
    resultList = pool.map(_optimize_single_model_partial,
                          model_cases_list, chunksize=1)

    pool.close()
    pool.join()


def _run_algorithm_hyperopt(experiment_configuration, recommender_class, hyperparameters_range_dictionary, earlystopping_hyperparameters, output_folder_path, use_gpu):

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=experiment_configuration.evaluator_validation,
                                               evaluator_test=experiment_configuration.evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[experiment_configuration.URM_train],
        CONSTRUCTOR_KEYWORD_ARGS={"use_gpu": use_gpu},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=earlystopping_hyperparameters,
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[
            experiment_configuration.URM_train_last_test],
        CONSTRUCTOR_KEYWORD_ARGS={"use_gpu": use_gpu},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=earlystopping_hyperparameters,
    )

    # Final step, after the hyperparameter range has been defined for each type of algorithm
    hyperparameterSearch.search(recommender_input_args,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=experiment_configuration.n_cases,
                                n_random_starts=experiment_configuration.n_random_starts,
                                resume_from_saved=experiment_configuration.resume_from_saved,
                                save_model=experiment_configuration.save_model,
                                evaluate_on_test=experiment_configuration.evaluate_on_test,
                                max_total_time=experiment_configuration.max_total_time,
                                output_folder_path=output_folder_path,
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,
                                metric_to_optimize=experiment_configuration.metric_to_optimize,
                                cutoff_to_optimize=experiment_configuration.cutoff_to_optimize,
                                recommender_input_args_last_test=recommender_input_args_last_test)

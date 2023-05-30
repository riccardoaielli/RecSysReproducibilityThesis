#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/01/23

@author: Anonymized for blind review
"""


import pandas as pd
import numpy as np
import time, os
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import itertools

from HyperparameterTuning.SearchAbstractClass import SearchAbstractClass
import traceback


class TimeoutError(Exception):
    def __init__(self, max_total_time_seconds, current_total_time):
        max_total_time_seconds_value, max_total_time_seconds_unit = seconds_to_biggest_unit(max_total_time_seconds)
        current_total_time_seconds_value, current_total_time_seconds_unit = seconds_to_biggest_unit(current_total_time)

        message = "Total training and evaluation time is {:.2f} {}, exceeding the maximum threshold of {:.2f} {}".format(
            current_total_time_seconds_value, current_total_time_seconds_unit, max_total_time_seconds_value, max_total_time_seconds_unit)

        super().__init__(message)



class SearchGrid(SearchAbstractClass):

    ALGORITHM_NAME = "SearchGrid"

    def __init__(self, recommender_class, evaluator_validation = None, evaluator_test = None, verbose = True):

        assert evaluator_validation is not None, "{}: evaluator_validation must be provided".format(self.ALGORITHM_NAME)

        super(SearchGrid, self).__init__(recommender_class,
                                         evaluator_validation = evaluator_validation,
                                         evaluator_test = evaluator_test,
                                         verbose = verbose)


    def _resume_from_saved(self):

        try:
            self.metadata_dict = self.dataIO.load_data(file_name = self.output_file_name_root + "_metadata")

        except (KeyboardInterrupt, SystemExit) as e:
            # If getting a interrupt, terminate without saving the exception
            raise e

        except FileNotFoundError:
            self._write_log("{}: Resuming '{}' Failed, no such file exists.\n".format(self.ALGORITHM_NAME, self.output_file_name_root))
            self.resume_from_saved = False
            return None, None

        except Exception as e:
            self._write_log("{}: Resuming '{}' Failed, generic exception: {}.\n".format(self.ALGORITHM_NAME, self.output_file_name_root, str(e)))
            raise e

        # Get hyperparameter list and corresponding result
        # Make sure that the hyperparameters only contain those given as input and not others like the number of epochs selected by earlystopping
        # Add only those having a search space, in the correct ordering
        hyperparameters_df = self.metadata_dict['hyperparameters_df'][self.hyperparameter_names]

        assert self.hyperparameter_names == self.metadata_dict['hyperparameters_df'].columns.to_list(), "The ordering of the hyperparameters was changed"

        # Check if search was only done partially.
        # Some hyperparameters may be nans, but at least one should have a definite value.
        # All valid hyperprameter cases should be at the beginning
        self.model_counter = hyperparameters_df.notna().any(axis=1).sum()

        # If the data structure exists but is empty, return None
        if self.model_counter == 0:
            self.resume_from_saved = False
            return None, None

        assert hyperparameters_df[:self.model_counter].notna().any(axis=1).all(),\
                   "{}: Resuming '{}' Failed due to inconsistent data, valid hyperparameter configurations are not contiguous at the beginning of the dataframe.".format(self.ALGORITHM_NAME, self.output_file_name_root)

        # hyperparameters_df = hyperparameters_df[:self.model_counter]
        #
        # # Check if single value categorical. It is aimed at intercepting
        # # Hyperparameters that are chosen via early stopping and set them as the
        # # maximum value as per hyperparameter search space. If not, the gp_minimize will return an error
        # # as some values will be outside (lower) than the search space
        # for hyperparameter_index, hyperparameter_name in enumerate(self.hyperparams_names):
        #     if isinstance(self.hyperparams_values[hyperparameter_index], Categorical) and len(self.hyperparams_values[hyperparameter_index].categories) == 1:
        #         hyperparameters_df[hyperparameter_name] = self.hyperparams_values[hyperparameter_index].bounds[0]
        #
        # hyperparameters_list_input = hyperparameters_df.values.tolist()

        result_on_validation_df = self.metadata_dict['result_on_validation_df']

        # All valid hyperparameters must have either a valid result or an exception
        for index in range(self.model_counter):
            is_exception = self.metadata_dict["exception_list"][index] is None
            is_validation_valid = result_on_validation_df is not None and result_on_validation_df[self.metric_to_optimize].notna()[index].any()
            assert is_exception == is_validation_valid,\
                   "{}: Resuming '{}' Failed due to inconsistent data. There cannot be both a valid result and an exception for the same case.".format(self.ALGORITHM_NAME, self.output_file_name_root)

        self._print("{}: Resuming '{}'... Loaded {} configurations.".format(self.ALGORITHM_NAME, self.output_file_name_root, self.model_counter))


    def _was_already_evaluated_check(self, current_fit_hyperparameters_dict):
        return False, None


    def search(self, recommender_input_args,
               hyperparameter_search_space,
               metric_to_optimize = None,
               cutoff_to_optimize = None,
               output_folder_path = None,
               output_file_name_root = None,
               save_model = "best",
               save_metadata = True,
               resume_from_saved = False,
               recommender_input_args_last_test = None,
               evaluate_on_test = "best",
               max_total_time = None,
               terminate_on_memory_error = True,
               ):
        """

        :param recommender_input_args:
        :param hyperparameter_search_space:
        :param metric_to_optimize:
        :param cutoff_to_optimize:
        :param output_folder_path:
        :param output_file_name_root:
        :param save_model:          "no"    don't save anything
                                    "all"   save every model
                                    "best"  save the best model trained on train data alone and on last, if present
                                    "last"  save only last, if present
        :param save_metadata:
        :param recommender_input_args_last_test:
        :return:
        """



        self.n_loaded_counter = 0

        self.max_total_time = max_total_time

        if self.max_total_time is not None:
            total_time_value, total_time_unit = seconds_to_biggest_unit(self.max_total_time)
            self._print("{}: The search has a maximum allotted time of {:.2f} {}".format(self.ALGORITHM_NAME, total_time_value, total_time_unit))

        hyperparameter_lists = []
        self.hyperparameter_names = []
        n_cases = 1

        for name, value_list in hyperparameter_search_space.items():
            assert isinstance(value_list, list), "Hyperparameter values must be specified in a list even if only one value exists."
            n_cases *= len(value_list)
            hyperparameter_lists.append(value_list)
            self.hyperparameter_names.append(name)

        # Sort the hyperparameters according to the decreasing number of hyperparameter values
        sort_hyperparameter_size = np.argsort([-len(value_list) for value_list in hyperparameter_lists])
        hyperparameter_lists = [hyperparameter_lists[i] for i in sort_hyperparameter_size]
        self.hyperparameter_names = [self.hyperparameter_names[i] for i in sort_hyperparameter_size]

        self._print("{}: Number of hyperparameter configurations to search is {:.2E}".format(self.ALGORITHM_NAME, n_cases))


        self._set_search_attributes(recommender_input_args,
                                    recommender_input_args_last_test,
                                    self.hyperparameter_names,
                                    metric_to_optimize,
                                    cutoff_to_optimize,
                                    output_folder_path,
                                    output_file_name_root,
                                    resume_from_saved,
                                    save_metadata,
                                    save_model,
                                    evaluate_on_test,
                                    n_cases,
                                    terminate_on_memory_error)


        try:
            if self.resume_from_saved:
                self._resume_from_saved()

            for index,current_hyperparameter_config in enumerate(itertools.product(*hyperparameter_lists)):
                if index >= self.model_counter:
                    self._objective_function_list_input(current_hyperparameter_config)
                else:
                    print("Skipping config: {}".format(current_hyperparameter_config))

        except TimeoutError as e:
            # When in TimeoutError, stop search but continue to train the _last model, if requested
            self._write_log("{}: Search interrupted. {}\n".format(self.ALGORITHM_NAME, e))


        if self.n_loaded_counter < self.model_counter:
            self._write_log("{}: Search complete. Best config is {}: {}\n".format(self.ALGORITHM_NAME,
                                                                           self.metadata_dict["hyperparameters_best_index"],
                                                                           self.metadata_dict["hyperparameters_best"]))

        if self.recommender_input_args_last_test is not None:
            self._evaluate_on_test_with_data_last()







    def _objective_function_list_input(self, current_fit_hyperparameters_list_of_values):
        """
        This function parses the hyperparameter list provided by the iterator function into a dictionary that
        can be used for the fitting of the model and provided to the objective function defined in the abstract class

        This function also checks if the search should be interrupted if the time has expired

        :param current_fit_hyperparameters_list_of_values:
        :return:
        """

        # The search can only progress if the total training + validation time is lower than max threshold
        # The time necessary for the last case is estimated based on the time the corresponding case took
        total_current_time = self.metadata_dict["time_on_train_total"] + self.metadata_dict["time_on_validation_total"]
        estimated_last_time = self.metadata_dict["time_df"].loc[self.metadata_dict['hyperparameters_best_index']][["train", "validation"]].sum() if \
                              self.metadata_dict['hyperparameters_best_index'] is not None else 0


        if self.max_total_time is not None:
            # If there is no "last" use the current total time, otherwise estimate its required time form the average
            if self.recommender_input_args_last_test is None and total_current_time > self.max_total_time:
                raise TimeoutError(self.max_total_time, total_current_time)
            elif self.recommender_input_args_last_test is not None and total_current_time + estimated_last_time> self.max_total_time:
                raise TimeoutError(self.max_total_time, total_current_time + estimated_last_time)


        current_fit_hyperparameters_dict = dict(zip(self.hyperparameter_names, current_fit_hyperparameters_list_of_values))
        result = self._objective_function(current_fit_hyperparameters_dict)

        return result


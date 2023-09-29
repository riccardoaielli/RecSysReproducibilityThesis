# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):

    # Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append(
            (hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(
            dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
                                                        hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                        hyper_ret[best_test_idx][0],
                                                                        dict2str(
                                                                            hyper_ret[best_test_idx][1]),
                                                                        dict2str(hyper_ret[best_test_idx][2])))

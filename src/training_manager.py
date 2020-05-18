#!/usr/bin/env python

import time
import os
import json
import sys
import shutil
from copy import deepcopy

import numpy as np
import torch


# IO
ver = sys.version_info
if ver > (3, 0):
    # import pickle as pk
    opts_write = {'encoding': 'utf-8', 'newline': ''}
    opts_read = {'encoding': 'utf-8'}
else:
    # import cPickle as pk
    opts_write = {}
    opts_read = {}


def get_current_time():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
        return data


def pretty_print_dict(dict_object):
    print(json.dumps(dict_object, indent=4, sort_keys=True))


def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def write_line(file_path, data):
    with open(file_path, 'w', **opts_write) as opdwf:
        opdwf.write(data)


def write_lines(file_path, data_list):
    with open(file_path, 'w', **opts_write) as opdwf:
        opdwf.writelines([str(term)+'\n' for term in data_list])


def read_lines(file_path):
    with open(file_path, 'r', **opts_read) as opdrf:
        data = [term.strip() for term in opdrf.readlines()]
        return data


def append_line(file_path, data):
    with open(file_path, 'a', **opts_write) as opdwf:
        opdwf.write(data)


# Save best
def save_best_models(metric_name, epoch, best_value,
                     output_dir, networks, optimizers, names):
    """
    multiple networks
    """
    if output_dir is not None:
        model_dir = os.path.join(output_dir, 'model')

        for network, optimizer, name in zip(networks, optimizers, names):
            params_best_fp = os.path.join(model_dir, 'params.{}.best_{}.torch'.format(name, metric_name))
            save_best_params(
                params_best_fp,
                network, optimizer, epoch, best_value,
                metric_name
            )

        epoch_best_fp = os.path.join(model_dir, 'epoch.best_{}.txt'.format(metric_name))
        write_line(epoch_best_fp, str(epoch))


def save_best_models_so_far(metric_name, epoch, output_dir, names):
    """
    multiple networks
    """
    if output_dir is not None:
        model_dir = os.path.join(output_dir, 'model')

        for name in names:
            params_best_fp = os.path.join(model_dir, 'params.{}.best_{}.torch'.format(name, metric_name))
            out_fp = os.path.join(model_dir, 'params.{}.best@{}_{}.torch'.format(name, epoch, metric_name))
            shutil.copyfile(params_best_fp, out_fp)


# check best value to save best model
def check_best_value(best_value, current_value, higher_or_lower):
    '''
    higher_or_better: str
        'higher_better', 'lower_better'
    '''
    if higher_or_lower == 'higher_better':
        def comp_func(x, y):
            return x >= y
    elif higher_or_lower == 'lower_better':
        def comp_func(x, y):
            return x <= y

    if comp_func(current_value, best_value):
        best_value = current_value
        is_best_value_updated = True
    else:
        is_best_value_updated = False
    return best_value, is_best_value_updated


# Save/load
def save_best_params(fp, network, optimizer, epoch, value, metric):
    net_state_dict = network.state_dict()

    if optimizer is None:
        opt_state_dict = None
    else:
        opt_state_dict = optimizer.state_dict()

    out = {
        'state_dict.model': net_state_dict,
        'state_dict.optimizer': opt_state_dict,
        'epoch': epoch,
        'value.{}'.format(metric): value
    }
    torch.save(out, fp)


def save_params(fp, network, optimizer):
    net_state_dict = network.state_dict()

    if optimizer is None:
        opt_state_dict = None
    else:
        opt_state_dict = optimizer.state_dict()

    out = {
        'state_dict.model': net_state_dict,
        'state_dict.optimizer': opt_state_dict,
    }
    torch.save(out, fp)


def load_params(fp, device_id):
    device = torch.device(device_id)
    params = torch.load(fp, map_location=device)
    return params


def load_model(fp, network, optimizer=None, device_id='cpu'):
    obj = load_params(fp, device_id)
    model_state_dict = obj['state_dict.model']
    optimizer_state_dict = obj['state_dict.optimizer']

    if optimizer is not None and optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    network.load_state_dict(model_state_dict)


def get_structure_description(network):
    '''

    Return
    ------
    list of layer description

    '''
    # des = str(network).split('\n  ')[1:-1]
    try:
        des = str(network).split('\n  ')[1:]
        des[-1] = des[-1].replace('\n)', '')
    except Exception:
        des = ''

    return des


def save_structure_description(out_fp, network):
    des = get_structure_description(network)
    write_lines(out_fp, des)


def save_record(fp, record):
    '''
    info: list of tuples
    '''
    write_json(fp, record)


# Manager
class TrainingManager(object):
    '''
    Managing model training
    '''
    def __init__(self, networks, optimizers, names, output_dir=None, save_rate=1, script_path=None):
        '''
        networks: list of `network`

        optimizers: list of `optimizer`

        names: list of str
            names for the networks

        save_rate: int
            save every save_rate epochs

        '''
        if len(networks) > 1:
            # assert(names is not None and len(names) == len(networks))
            assert(len(names) == len(networks))

        assert(len(networks) == len(optimizers))
        assert(len(names) == len(set(names)))

        # self.score_higher_better = score_higher_better
        self.script_path = script_path

        self.networks = networks
        self.optimizers = optimizers
        self.names = names

        self.output_dir = output_dir
        self.save_rate = save_rate

        self.model_dir = os.path.join(output_dir, 'model')
        self.record_dir = os.path.join(output_dir, 'record')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.record_dir):
            os.mkdir(self.record_dir)

        self.best_va_metrics = None

        # Records
        self.records = {0: {'time': self.get_current_pretty_time(), 'epoch': 'Training starts'}}

    def update_records(self, current_record, epoch, records):
        record = {
            'time': self.get_current_pretty_time(),
            'epoch': epoch,
            'record': current_record
        }
        records[epoch] = record

    def save_initial(self):
        # Save structure description
        description_fp_list = [
            os.path.join(self.output_dir, 'structure_description.{}.csv').format(name)
            for name in self.names
        ]
        for network, description_fp in zip(self.networks, description_fp_list):
            save_structure_description(description_fp, network)

        # Copy script
        if self.script_path is not None:
            out_script_fp = os.path.join(
                self.output_dir, 'training_script.py')
            shutil.copy(self.script_path, out_script_fp)

    def save_middle(self, epoch, record, va_metrics=None):
        # Save in the middle
        if epoch % self.save_rate == 0:
            # Save the params at this epoch
            # for name, network, optimizer in zip(self.names, self.networks, self.optimizers):
            for name, network, optimizer in zip(self.names, self.networks, self.optimizers):
                params_fp = os.path.join(self.model_dir, 'params.{}.@{}.torch'.format(name, epoch))
                save_params(params_fp, network, optimizer)

            '''
            # Save the best params so far
            if va_metrics is not None:
                for metric_name, _, _ in va_metrics:
                    save_best_models_so_far(metric_name, epoch, self.output_dir, self.names)
            '''

        # Save record
        record_fp = os.path.join(self.record_dir, 'record.@{}.json'.format(epoch))

        self.update_records(record, epoch, self.records)
        write_json(record_fp, self.records)

        # Save latest
        self.save_latest(epoch)

    def save_latest(self, epoch):
        # Save record
        record_fp = os.path.join(self.record_dir, 'record.latest.json')
        write_json(record_fp, self.records)

        # Save params
        params_fp_list = [
            os.path.join(self.model_dir, 'params.{}.latest.torch'.format(name))
            for name in self.names
        ]
        for network, optimizer, params_fp in zip(self.networks, self.optimizers, params_fp_list):
            save_params(params_fp, network, optimizer)

    def check_best_va_metrics(self, va_metrics, epoch):
        '''
        va_metrics: list of tuples
            each tuple is of the form (metric_name, value, higher_or_lower)

            higher_or_lower:
                'higher_better': higher metric is better
                'lower_better': lower metric is better

        '''
        # Initilize best_va_metrics
        if self.best_va_metrics is None:
            for _, _, higher_or_lower in va_metrics:
                assert(higher_or_lower in ['higher_better', 'lower_better'])

            init_epoch = -1
            self.best_va_metrics = dict(
                [(metric_name, {'epoch': init_epoch, 'value': np.inf}) if higher_or_lower == 'lower_better' else
                 (metric_name, {'epoch': init_epoch, 'value': -np.inf})
                 for metric_name, _, higher_or_lower in va_metrics]
            )

        # Check best loss
        # deco = decorator_for_save_best(self.output_dir, self.networks, self.optimizers, self.names, epoch, metric_name='loss')

        for metric_name, value, higher_or_lower in va_metrics:
            best_value = self.best_va_metrics[metric_name]['value']
            self.best_va_loss, is_updated = check_best_value(best_value, value, higher_or_lower)
            if is_updated:
                self.best_va_metrics[metric_name] = {'epoch': epoch, 'value': value}
                save_best_models(metric_name, epoch, best_value, self.output_dir, self.networks, self.optimizers, self.names)

        return deepcopy(self.best_va_metrics)

    def get_resumed_epoch(self, latest_record):
        resumed_epoch = max([int(key) for key in latest_record])

        return resumed_epoch

    def resume_training(self, resumed_model_id, resumed_save_dir):
        '''
        resumed_model_id:
            resume training the model with model id <model_id>
        '''

        base_model_dir = os.path.join(resumed_save_dir, resumed_model_id)
        param_dir = os.path.join(base_model_dir, 'model')

        # Get record
        record_fp = os.path.join(base_model_dir, 'record', 'record.latest.json')
        records = read_json(record_fp)
        records = {int(key): records[key] for key in records}
        self.records = records

        # Get latest saved epoch
        resumed_epoch = self.get_resumed_epoch(records)

        # Resume networks and optimizers
        for name, network, optimizer in zip(self.names, self.networks, self.optimizers):
            param_fp = os.path.join(param_dir, 'params.{}.latest.torch'.format(name))
            load_model(param_fp, network, optimizer)

        # Set best loss and best score
        self.best_va_metrics = records[resumed_epoch]['record']['best_va_metrics']

        next_epoch = resumed_epoch + 1

        return next_epoch

    def print_record(self, record):
        print(json.dumps(record, indent=4, sort_keys=True))

    def print_record_in_one_line(self, record):
        print('. '.join(
            ['{}: {}'.format(metric_name, ', '.join(
                ['{}: {}'.format(nn, vv) for nn, vv in info.items()]))
             for metric_name, info in record.items()]))

    def get_current_pretty_time(self):
        return time.strftime('%Y/%m/%d %H:%M:%S')

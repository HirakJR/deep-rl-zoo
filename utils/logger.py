import pickle
from typing import Dict
import os
import time
import csv
import torch


class Logger:
    '''
    Logs statistics of the training run, and save them as csv or pickle files.
    Reference: OpenAI's Spinning Up Logger, CS 285 (UCB) data viz handout

    Arguments:

        dir (str): Name of directory to save log files to. If ``None``, defaults to ``/log_dir``.
        filename (str): Name of log file, defaults to ``log.txt``.
        exp_name (str): Experiment name, defaults to ``exp`` + timestamp.

    '''
    def __init__(self, dir: str = "log_dir/",
                        filename : str = 'log.txt',
                        exp_name: str = None):
        '''
        Initialize Logger
        '''
        self.dir = dir
        self.filename = filename
        if exp_name is None:
            self.exp_name = f"exp_{int(time.time())}"
        else: self.exp_name = exp_name

        if os.path.exists(os.path.join(self.dir, self.exp_name)):
            print("Path exists, overwriting content")
        else:
            os.makedirs(os.path.join(self.dir, self.exp_name))
        
        self.out_file = os.path.join(self.dir, self.exp_name, self.filename)
        self._base_name = os.path.splitext(self.filename)[0]
        self._pkl_file = os.path.join(self.dir, self.exp_name, self._base_name + '.pkl')


    def log(self, msg: Dict):
        '''
        Takes a dictionary of the required parameters to be logged,writes it to a txt file
        '''
        with open(self.out_file, 'a') as f:
            f.write(str(msg)+'\n')
            f.close()
    
    
    def dump_csv(self):
        '''
        Takes the saved log txt file and creates a csv file, helpful for using with plotter
        '''
        with open(self.out_file, 'r') as txtfile:
            dict_list = [eval(line.strip()) for line in txtfile]
        csv_file = os.path.join(self.dir, self.exp_name, self._base_name + '.csv')

        with open(csv_file, 'w') as f:
            writer = csv.DictWriter(f, dict_list[0].keys())
            writer.writeheader()

            for dict in dict_list:
                writer.writerow(dict)
        return csv_file

    def dump_pickle(self):
        '''
        Dumps the log file to a pickle file
        '''
        with open(self.out_file, 'r') as txtfile:
            dict_list = [eval(line.strip()) for line in txtfile]
            with open(self._pkl_file, 'wb') as pklfile:
                pickle.dump(dict_list, pklfile)
                
    def load_pickle(self):
        '''
        Unpickles the pickle file to a txt log file
        '''
        file = None
        if os.path.exists(self._pkl_file):
            with open(self._pkl_file, 'rb') as f:
                file = pickle.load(f)
        unpickled_path = os.path.join(self.dir, self.exp_name, self.filename + 'unpickled')
        with open(unpickled_path, 'w') as f:
            f.write(str(file))
            
    
class TorchLogger(Logger):
    '''
    Logger for Pytorch, contains helper functions for saving and checkpointing models
    '''
    def __init__(self, dir: str = "log_dir/",
                        filename : str = 'torchlog.txt',
                        exp_name: str = None,
                        dict: Dict = None):
        '''
        Arguments:

        dir (str): Name of directory to save log files to. If ``None``, defaults to ``log_dir/``.
        filename (str): Name of log file, defaults to ``log.txt``, defaults to ``None``
        exp_name (str): Experiment name, defaults to ``exp`` + timestamp.

        '''
        super().__init__(dir=dir, filename=filename, exp_name=exp_name)

        self.dict = dict
        if self.dict is None:
            print("Checkpoint dictionary missing")

        self.checkpoint_dir = os.path.join(self.dir,self.exp_name,'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, step: int):
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_{step}.pt')
        torch.save(self.dict, path)


    def save_best_checkpoint(self, filename: str, run_name: str):
        '''
        Arguments:
            filename: name of the algorithm
            run_name: exp_name 
        '''
        dir = 'runs/'
        self._best_dir = os.path.join(dir, filename)
        os.makedirs(self._best_dir, exist_ok=True)
        self._best_path = os.path.join(self._best_dir,f'{run_name}.pt')
        torch.save(self.dict, self._best_path)    
    
    def load_checkpoint(self, path):
        return torch.load(path)
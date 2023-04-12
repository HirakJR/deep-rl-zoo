import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import time
from PIL import Image
import os

def plot(data: pd.DataFrame,
            x: str = None,
            y: str = None,
            hue: str = None,
            style: str = None,
            legend: str = 'auto',
            save: bool = False,
            name: str = None):
        '''
        Create time-series plot
        Arguments:
            data (DataFrame): pandas dataframe object
            x (str, key in data): x-axis (column from dataframe)
            y (str, key in data): y-axis (column from dataframe)
            hue (str, key in data): group lines to produce lines of different color
            style (str, key in data): group lines to produce lines of different markers
            legend: 'auto' or 'false'
            save (Bool): saves plot (default: False)
            path (str): plot image save path
        '''
        sns.set_theme(context = 'notebook',style = 'darkgrid',palette='pastel')
        plt.figure()
        sns.lineplot(data = data,
                     x = x,
                     y = y,
                     hue = hue,
                     style = style,
                     legend = legend)
        if save:
                os.makedirs('../plots/', exist_ok=True)
                if name is not None:
                    plt.savefig(f'../plots/{name}')
                else:
                    plt.savefig(f'../plots/{str(int(time.time()))}')
        plt.show()

def make_gif(img_list: List,
             name: str):
    '''
    Takes a list of images and creates a gif.
    Intended use is to monitor agent learning
    Arguments:
        img_list (List): list of images
        name (str): name for the saved gif
    '''
    images = list(Image.open(file).convert('RGB') for file in img_list)
    os.makedirs('../gifs/', exist_ok=True)
    images[0].save(f'../gifs/{name}', save_all=True, append_images=images[1:], duration=100, loop=0)
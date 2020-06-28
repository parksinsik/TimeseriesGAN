#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils


# In[16]:


def time_series_to_plot(time_series, dpi=25, feature_index=0, n_images_per_row=4, titles=None):
    
    images = []
    
    for i , series in enumerate(time_series.detach()):
        
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        
        if titles:
            ax.set_title(titles[i])
            
        ax.plot(series[:, feature_index].numpy())
        fig.canvas.draw()
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        
        images.append(data)
        plt.close(fig)
        
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    grid = vutils.make_grid(images.detach(), nrow=n_images_per_row)
    
    return grid


def tensor_to_string_list(tensor):
    scalar_list = tensor.squeeze().numpy().tolist()
    return ["%.6f" % scalar for scalar in scalar_list]


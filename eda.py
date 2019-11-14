# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:03:41 2019

@author: Stephanie
"""

%matplotlib inline
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

read_xyz_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['x', 'y', 'z',  'r', 'g','b'], header = None)
read_xyz_data('SD2_full_home_simplified_1cm.xyz')
%%time
fig, m_axs = plt.subplots(1, 3, figsize = (20, 5))
ax_names = 'xyz'
for i, c_ax in enumerate(m_axs.flatten()):
    plot_axes = [x for j, x in enumerate(ax_names) if j!=i]
    c_ax.scatter(df[plot_axes[0]],
                df[plot_axes[1]],
                c=df[['r', 'g', 'b']].values/255, 
                 s=1
                )
    c_ax.set_xlabel(plot_axes[0])
    c_ax.set_ylabel(plot_axes[1])

%%time
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
ax.scatter(
            df['x'], df['y'], df['z'],
            c=df[['r', 'g', 'b']].values/255, s=3)  
ax.view_init(15, 165)




%matplotlib inline
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

all_paths = [os.path.join(path, file) for path, _, files in os.walk(top = os.path.join('..', 'input')) 
             for file in files if ('.labels' in file) or ('.txt' in file)]
label_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
all_files_df = pd.DataFrame({'path': all_paths})
all_files_df['basename'] = all_files_df['path'].map(os.path.basename)
all_files_df['id'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[0])
all_files_df['ext'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[1][1:])
all_files_df.sample(3)


all_training_pairs = all_files_df.pivot_table(values = 'path', columns = 'ext', index = ['id'], aggfunc = 'first').reset_index()
all_training_pairs

_, test_row = next(all_training_pairs.dropna().tail(1).iterrows())
print(test_row)
read_label_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['class'], index_col = False)
read_xyz_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b'], header = None) #x, y, z, intensity, r, g, b
read_joint_data = lambda c_row, rows: pd.concat([read_xyz_data(c_row['txt'], rows), read_label_data(c_row['labels'], rows)], axis = 1)
read_joint_data(test_row, 10)

%%time
full_df = read_joint_data(test_row, None)


test_df = full_df[(full_df.index % 50)==0]
print(full_df.shape[0], 'rows', test_df.shape[0], 'number of filtered rows')


%%time
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
ax.scatter(
            test_df['x'], test_df['y'], test_df['z'],
            c=test_df[['r', 'g', 'b']].values/255, s=3)  
ax.view_init(15, 165)

ax.view_init(45, 220)
fig.savefig('3D_rendering.png', dpi = 300)
fig


%%time
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')  
for (c_key, c_value) in label_names.items():
    c_df = test_df[test_df['class']==c_key]
    ax.plot(c_df['x'], c_df['y'], c_df['z'], '.', label = c_value, alpha = 0.5)  
ax.legend()
ax.view_init(15, 165)
fig.savefig('3d_labels.png', dpi = 300)



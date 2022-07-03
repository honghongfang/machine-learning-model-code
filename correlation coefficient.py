# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:14:39 2022

@author: fanghonghong
"""

# numpy implement
import numpy as np
a = np.array([80,75,70,65,60])
b = np.array([70,66,68,64,62])
c = np.array([70, 66, 68, 64, 62])
d = np.vstack((a, b, c))
np.corrcoef(d)
# pandas implement
import pandas as pd
df = pd.DataFrame(
        [        
                [80, 70, 60],
                [75, 66, 55],
                [70, 68, 55],
                [65, 64, 45],
                [60, 62, 64]
        ],index=['A','B','C','D','E'],columns=['math','physics','chemical']
        )
df.corr()


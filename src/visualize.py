import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def box_plot(data):
    sns.boxplot(x="variable", y="value", data=pd.melt(data))
    plt.show()
import pandas as pd
from utils.plotter import plot_distributions
import os
import utils.consts  as cons
df=pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED ,'fram_count_preprocessed.csv'))
plot_distributions(df, 'fram' ,color='red')


# df=pd.read_csv( os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED ,'steno_count_preprocessed.csv'))
# plot_distributions( df, 'steno' , color='blue')

# df=pd.read_csv( os.path.join(cons.PATH_PROJECT_DATA_STENO_DIR ))
# df['sex'].value_counts()

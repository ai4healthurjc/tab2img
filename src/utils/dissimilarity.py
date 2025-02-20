import numpy as np
import pandas as pd
import itertools
from scipy.stats import spearmanr, pointbiserialr, chi2_contingency, pearsonr
from phik import phik_from_array
import scipy.spatial as ss
from scipy.special import digamma
from math import log
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import phik
import scipy.stats as stats
from utils.preprocessing import identify_feature_type, identify_type_features
from sklearn.preprocessing import MinMaxScaler
from utils.plotter import plot_heatmap

def cramers_corr(var_x, var_y):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = pd.crosstab(var_x, var_y).values
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def compute_corr_mixed_data(data):

    n_features = data.shape[1]
    pairs = list(itertools.combinations_with_replacement(range(n_features), 2))
    m_corr = np.zeros((n_features, n_features))
    df_meta, _, _ = identify_type_features(data)

    for i, j in pairs:
        x_var = data.iloc[:, i]
        y_var = data.iloc[:, j]
        x_var_name = data.columns[i]
        y_var_name = data.columns[j]

        x_var_type = df_meta.iloc[i, df_meta.columns.get_loc('type')]
        y_var_type = df_meta.iloc[j, df_meta.columns.get_loc('type')]

        if x_var_type == 'binary':
            if y_var_type == 'binary':
                pair_corr = phik_from_array(x_var, y_var)
            elif y_var_type == 'categorical':
                pair_corr = phik_from_array(x_var, y_var)
            else: # numerical
                pair_corr = pointbiserialr(x_var, y_var).correlation
        elif x_var_type == 'categorical':
            if y_var_type == 'binary':
                pair_corr = phik_from_array(x_var, y_var)
            elif y_var_type == 'categorical':
                pair_corr = phik_from_array(x_var, y_var)
            else: # numerical
                pair_corr = phik_from_array(x_var, y_var)
        else:
            if y_var_type == 'binary':
                pair_corr = phik_from_array(x_var, y_var)
            elif y_var_type == 'categorical':
                pair_corr = phik_from_array(x_var, y_var)
            else: # numerical
                pair_corr, _ = pearsonr(x_var, y_var)

        print('({},{})-({}, {})-({}, {})->{}'.format(i, j, x_var_name, y_var_name, x_var_type, y_var_type, pair_corr))

        m_corr[i, j] = pair_corr
        m_corr[j, i] = pair_corr

    v_var_names = data.columns
    df_corr = pd.DataFrame(m_corr, columns=v_var_names, index=v_var_names)
    print(df_corr)

    return df_corr.values


# def cramers_v(x, y):
#     contingency = pd.crosstab(x, y)
#     chi2 = chi2_contingency(contingency)[0]
#     n = len(x)
#     return np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

# def cramers_v(x, y):
#     confusion_matrix = pd.crosstab(x,y)

#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     phi2 = chi2 / n
#     r, k = confusion_matrix.shape
#     phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
#     rcorr = r - ((r-1)**2)/(n-1)
#     kcorr = k - ((k-1)**2)/(n-1)
#     return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def cramers_v(var1, var2):
    data = pd.crosstab(var1, var2).values
    chi_2 = stats.chi2_contingency(data)[0]
    n = data.sum()
    phi_2 = chi_2 / n
    r, k = data.shape
    return np.sqrt(phi_2 / min((k-1), (r-1)))


def eta_squared(categorical_var, numeric_var):
    # Crear una tabla de contingencia
    contingency_table = pd.crosstab(categorical_var, numeric_var)

    # Calcular la chi cuadrado
    chi2, _, _, _ = chi2_contingency(contingency_table)

    # Calcular el tamaño de efecto (Eta cuadrado)
    num_rows, num_cols = contingency_table.shape
    phi_squared = chi2 / (len(categorical_var) * (min(num_rows, num_cols) - 1))
    eta_squared = phi_squared / np.sqrt(phi_squared + 1)

    return eta_squared
def Mixed_KSG(x, y, k=15):
    '''
    Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
    Using *Mixed-KSG* mutual information estimator

    Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
    y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
    k: k-nearest neighbor parameter

    Output: one number of I(X;Y)
    '''

    assert len(x) == len(y), "Lists should have the same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.values.reshape((N, 1))

    dx = len(x[0])
    if y.ndim == 1:
        y = y.values.reshape((N, 1))
    dy = len(y[0])
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point, k + 1, p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i], 1e-15, p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i], 1e-15, p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i], 1e-15, p=float('inf')))
        else:
            nx = len(tree_x.query_ball_point(x[i], knn_dis[i] - 1e-15, p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i], knn_dis[i] - 1e-15, p=float('inf')))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny)) / N
    return ans


def compute_corr_mixed_dataset_2(data, categorical_vars, numerical_vars):

    n_features = data.shape[1]
    pairs = list(itertools.combinations_with_replacement(range(n_features), 2))
    m_corr = np.zeros((n_features, n_features))

    for i, j in pairs:
        x_var_name = data.columns[i]
        y_var_name = data.columns[j]

        x_var = data.iloc[:, i]
        y_var = data.iloc[:, j]

        if x_var_name in categorical_vars: # categorical
            if y_var_name in categorical_vars:              
                
                pair_corr = cramers_v(x_var, y_var)
                #x, y = data[[x_var_name,y_var_name]].T.values
                #pair_corr=np.array(phik.phik_from_array(x, y))
                # pair_corr = phik_from_array(x_var, y_var)
                # pair_corr = Mixed_KSG(x_var,y_var)
                # pair_corr = mutual_info_classif(x_var.to_frame(), y_var, discrete_features=[True])

                # prueba=phik_matrix(x_var, y_var)
                # print(prueba)
                # pair_corr=phik_matrix(x_var, y_var).iloc[0, 0]
                # pair_corr=data.phik_matrix(interval_cols=cols)
                # pair_corr, _ = spearmanr(x_var, y_var)

            else:
                
                if x_var.nunique() == 2:
                    pair_corr = pointbiserialr(x_var, y_var).correlation
                else:

                    x, y = data[[x_var_name, y_var_name]].T.values
                    pair_corr = np.array(phik.phik_from_array(x, y, num_vars=['y']))
                # x_var = pd.DataFrame(x_var)
                # print(x_var)
            
                # y_var = pd.DataFrame(y_var)
                # print(y_var)

                # cols = [x_var_name,y_var_name]                
                
                # pair_corr = cramers_v(x_var, y_var)
                # pair_corr = phik_from_array(x_var, y_var)
                # pair_corr = Mixed_KSG(x_var,y_var)
                # pair_corr = mutual_info_classif(x_var.to_frame(), y_var, discrete_features=[True])

                # prueba=phik_matrix(x_var, y_var)
                # print(prueba)
                # pair_corr=phik_matrix(x_var, y_var).iloc[0, 0]
                # pair_corr=data.phik_matrix(interval_cols=cols)
                    print(pair_corr)


                # pair_corr, _ = spearmanr(x_var, y_var)


        else:  # numerical
            if y_var_name in categorical_vars:

                if y_var.nunique() == 2:
                        pair_corr = pointbiserialr(x_var, y_var).correlation
                else:

                    x, y = data[[x_var_name,y_var_name]].T.values
                    pair_corr=np.array(phik.phik_from_array(x, y, num_vars=['x']))

                # x_var = pd.DataFrame(x_var)
                # print(x_var)
            
                # y_var = pd.DataFrame(y_var)
                # print(y_var)

                # cols = [x_var_name,y_var_name]                
                
                # pair_corr = cramers_v(x_var, y_var)
                # pair_corr = phik_from_array(x_var, y_var)
                # pair_corr = Mixed_KSG(x_var,y_var)
                # pair_corr = mutual_info_classif(x_var.to_frame(), y_var, discrete_features=[True])

                # prueba=phik_matrix(x_var, y_var)
                # print(prueba)
                # pair_corr=phik_matrix(x_var, y_var).iloc[0, 0]
                # pair_corr=data.phik_matrix(interval_cols=cols)
            


                # pair_corr, _ = spearmanr(x_var, y_var)


            else:
                pair_corr, _ = spearmanr(x_var, y_var)
                
                # x_var = pd.DataFrame(x_var)
                # print(x_var)
            
                # y_var = pd.DataFrame(y_var)
                # print(y_var)

                # cols = [x_var_name,y_var_name]                
                
                # pair_corr = cramers_v(x_var, y_var)
                # pair_corr = phik_from_array(x_var, y_var)
                # pair_corr = Mixed_KSG(x_var,y_var)
                # pair_corr = mutual_info_classif(x_var.to_frame(), y_var, discrete_features=[True])

                # prueba=phik_matrix(x_var, y_var)
                # print(prueba)
                # pair_corr=phik_matrix(x_var, y_var).iloc[0, 0]
                # pair_corr=data.phik_matrix(interval_cols=cols)



                # pair_corr, _ = spearmanr(x_var, y_var)

        m_corr[i, j] = pair_corr
        m_corr[j, i] = pair_corr

    v_var_names = data.columns
    df_corr = pd.DataFrame(m_corr, columns=v_var_names, index=v_var_names)
    print(df_corr)
    

    return df_corr

def compute_correlations(x_var_name,y_var_name,categorical_vars,x_var,y_var,list_scalers):
    scaler_phik, scaler_cramer, scaler_point, scaler_spearman, scaler_phi = list_scalers

    if x_var_name in categorical_vars:
        if y_var_name in categorical_vars:
            if len(set(x_var)) == 2 and len(set(y_var)) == 2:
                pair_corr = np.array(phik.phik_from_array(x_var, y_var))
                pair_corr = scaler_phi.transform(np.asarray(pair_corr).reshape(-1,1))
            elif len(set(x_var)) != 2 and len(set(y_var)) != 2:
                pair_corr = cramers_corr(x_var, y_var)
                pair_corr = scaler_cramer.transform(np.asarray(pair_corr).reshape(-1,1))
            else:
                pair_corr = np.array(phik.phik_from_array(x_var, y_var))
                pair_corr = scaler_phi.transform(np.asarray(pair_corr).reshape(-1,1))
        else:
            if len(set(x_var)) == 2:
                pair_corr = pointbiserialr(x_var, y_var).correlation
                pair_corr = scaler_point.transform(np.asarray(pair_corr).reshape(-1,1))
            else:
                pair_corr = np.array(phik.phik_from_array(x_var, y_var, num_vars=[y_var_name]))
                pair_corr = scaler_phik.transform(np.asarray(pair_corr).reshape(-1,1))
    else:
        if y_var_name in categorical_vars:
            if len(set(y_var)) == 2:
                pair_corr = pointbiserialr(x_var, y_var).correlation
                pair_corr = scaler_point.transform(np.asarray(pair_corr).reshape(-1,1))
            else:
                pair_corr = np.array(phik.phik_from_array(x_var, y_var, num_vars=[x_var_name]))
                pair_corr = scaler_phik.transform(np.asarray(pair_corr).reshape(-1,1))
        else:
            pair_corr, _ = spearmanr(x_var, y_var)
            pair_corr = scaler_spearman.transform(np.asarray(pair_corr).reshape(-1,1))

    return pair_corr


def compute_correlations_normalization(pairs,data,categorical_vars):
    scaler_phik = MinMaxScaler()
    scaler_phi = MinMaxScaler()
    scaler_cramer = MinMaxScaler()
    scaler_point = MinMaxScaler()
    scaler_spearman = MinMaxScaler()
    phik_list = []
    phi_list = []
    cramer_list = []
    point_list = []
    spearman_list = []
    for i, j in pairs:
        x_var_name = data.columns[i]
        y_var_name = data.columns[j]
        x_var = data.iloc[:, i]
        y_var = data.iloc[:, j]
        if x_var_name in categorical_vars:
            if y_var_name in categorical_vars:
                if len(set(x_var)) == 2 and len(set(y_var)) == 2:
                    pair_corr = np.array(phik.phik_from_array(x_var, y_var))
                    phi_list.append(pair_corr)
                elif len(set(x_var)) != 2 and len(set(y_var)) != 2:
                    pair_corr = cramers_corr(x_var, y_var)
                    cramer_list.append(pair_corr)
                else:
                    pair_corr = np.array(phik.phik_from_array(x_var, y_var))
                    phi_list.append(pair_corr)
            else:
                if len(set(x_var)) == 2:
                    pair_corr = pointbiserialr(x_var, y_var).correlation
                    point_list.append(pair_corr)
                else:
                    pair_corr = np.array(phik.phik_from_array(x_var, y_var, num_vars=[y_var_name]))
                    phik_list.append(pair_corr)
        else:
            if y_var_name in categorical_vars:
                if len(set(y_var)) == 2:
                    pair_corr = pointbiserialr(x_var, y_var).correlation
                    point_list.append(pair_corr)
                else:
                    pair_corr = np.array(phik.phik_from_array(x_var, y_var, num_vars=[x_var_name]))
                    phik_list.append(pair_corr)
            else:
                pair_corr, _ = spearmanr(x_var, y_var)
                spearman_list.append(pair_corr)
    if len(phik_list)>0:
        scaler_phik.fit(np.asarray(phik_list).reshape(-1, 1))
    else:
        scaler_phik=0
    if len(phi_list)>0:
        scaler_phi.fit(np.asarray(phi_list).reshape(-1, 1))
    else:
        scaler_phi=0
    if len(cramer_list)>0:
        scaler_cramer.fit(np.asarray(cramer_list).reshape(-1, 1))
    else:
        scaler_cramer=0
    if len(point_list)>0:
        scaler_point.fit(np.asarray(point_list).reshape(-1, 1))
    else:
        scaler_point=0
    if len(spearman_list)>0:
        scaler_spearman.fit(np.asarray(spearman_list).reshape(-1, 1))
    else:
        scaler_spearman=0
    return scaler_phik,scaler_cramer,scaler_point,scaler_spearman, scaler_phi
def compute_corr_mixed_dataset(data, categorical_vars,save_path ):

    n_features = data.shape[1]

    pairs = list(itertools.combinations_with_replacement(range(n_features), 2))

    m_corr = np.zeros((n_features, n_features))
    scalers=compute_correlations_normalization(pairs, data, categorical_vars)
    for i, j in pairs:
        x_var_name = data.columns[i]
        y_var_name = data.columns[j]
        x_var = data.iloc[:, i]
        y_var = data.iloc[:, j]
        pair_corr=compute_correlations(x_var_name, y_var_name, categorical_vars, x_var, y_var,scalers)
        m_corr[i, j] = pair_corr
        m_corr[j, i] = pair_corr

    v_var_names = data.columns
    df_corr = pd.DataFrame(m_corr, columns=v_var_names, index=v_var_names)
    plot_heatmap(df_corr, save_path,'original')
    return df_corr
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 22:18:03 2021

https://realpython.com/knn-python/
https://www.askpython.com/python/examples/k-fold-cross-validation
to do everything without special libraries: https://towardsdatascience.com/build-knn-from-scratch-python-7b714c47631a
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import power_transform
import math
import warnings
import xlsxwriter
from sklearn.model_selection import RandomizedSearchCV



def import_data(file, output):
    Mdata=pd.read_excel(file)
    output_index = Mdata.columns.get_loc(output)
    Mdata=Mdata.iloc[:,0:output_index+1]
    Mdata=Mdata.drop(columns=["ID paciente ",'Lactatorraquia prequirúrgica (Lactato en LCR)','Lactatorraquia durante Ommaya'])
    #Mdata=Mdata.drop(columns="ID paciente ")
    
    return Mdata


def sel_var(data,var):
    data=data[var]
    
    return data

def fill_na(Mdata, categorical_var):
    # Subset of data of positive cases
    Mdata_pos = Mdata[Mdata.DVP==1]
    Mdata_neg = Mdata[Mdata.DVP==0]

    for header in Mdata.head():
        if header in categorical_var:
            cat_var_mod_pos = Mdata_pos[header].mode()
            cat_var_mod_neg = Mdata_neg[header].mode()
            
            # Filling missing data (categorical var.) with most freq. values for the output var. (ya)
            Mdata_pos[header] = Mdata_pos[header].fillna(value=int(cat_var_mod_pos.tolist()[0]))
            Mdata_neg[header] = Mdata_neg[header].fillna(value=int(cat_var_mod_neg.tolist()[0]))
            
        else:
            cont_var_mean_pos = Mdata_pos[header].mean()
            cont_var_mean_neg = Mdata_neg[header].mean()
            # Filling missing data (continuous var.) with most freq. values for the output var. (ya)
            Mdata_pos[header] = Mdata_pos[header].fillna(value=int(cont_var_mean_pos))
            Mdata_neg[header] = Mdata_neg[header].fillna(value=int(cont_var_mean_neg))
             
    frames = [Mdata_pos, Mdata_neg]
    result = pd.concat(frames)
    
    return result

    
def normalize(data):
    headers = data.columns
    X = data.iloc[:,0:-1] 
    y = data.iloc[:,-1]
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(X)
    norm_data = np.c_[norm_data,y]
    norm_data = pd.DataFrame(norm_data)
    norm_data.columns=headers
    
    return norm_data

def normalize2(data, cat_var):
    
    data_cont = data.drop(columns=cat_var)
    data_cat = data[cat_var]
    
    headers_cont = data_cont.columns
    headers_cat = data_cat.columns
    
    headers = data.columns
    X = data.iloc[:,0:-1] 
    y = data.iloc[:,-1]
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(X)
    norm_data = np.c_[norm_data,y]
    norm_data = pd.DataFrame(norm_data)
    norm_data.columns=headers
    
    return norm_data

def shuffle(dt, seed):
    if isinstance(dt, pd.DataFrame) == False:
        dt = pd.DataFrame(dt)
    dt_rand = dt.sample(frac=1, random_state=seed)
    dt_rand = dt_rand.values
    dt_rand = pd.DataFrame(dt_rand)
    
    return dt_rand
    
    
def stratified_cv(X_data, y_data, model, k_folds):
    skf = StratifiedKFold(n_splits=k_folds)
    result = cross_val_score(model , X_data, y_data, cv = skf)
    
    return result


def cross_val(X_data, y_data, model, k_folds):
    # Another method for cross-validation (not stratified)
    ##############################################################################
    kf = KFold(n_splits=k_folds)
    result = cross_val_score(model , X_data, y_data, cv = kf)
    
    return result


def optimize_KNN(X_data, y_data, k_folds):
    # Optimiztion of parameters
    ##############################################################################
    # By default, the GridSearchCV uses a 5-fold cross-validation. However, if it detects that a classifier is passed, rather than a regressor, it uses a stratified 5-fold.
    # KFold is a cross-validator that divides the dataset into k folds. Stratified is to ensure that each fold of dataset has the same proportion of observations with a given label.
    max_neighbors = math.ceil((len(X_data[:,0])/k_folds)*(k_folds-1))
    parameters = {"n_neighbors": range(1, max_neighbors+1),
                  "weights": ["uniform", "distance"]}

    gridsearch = GridSearchCV(KNeighborsClassifier(), parameters, cv=k_folds, verbose = 1)
    #gridsearch = RandomizedSearchCV(KNeighborsClassifier(), parameters, cv=k_folds, verbose = 1) to speed up the search but loose accuracy
    gridsearch.fit(X_data, y_data) 
    best_k = gridsearch.best_params_["n_neighbors"]
    best_weights = gridsearch.best_params_["weights"]
    best_score = gridsearch.best_score_
    
    return best_k, best_weights, best_score


def correlation(data, corr_type, threshold): # corr_type= 'pearson', 'spearman', 'kendall'
    # Find correlation of variables and output
    val = data.values
    X = val[:,0:-1] 
    corr = [0]*len(X[0])
    for i in range(len(X[0])):
        corr[i] = data.iloc[:,len(X[0])].corr(data.iloc[:,i], method=corr_type)
         
    corr = pd.DataFrame(corr)
    lst = abs(corr[0])>threshold
    lst = lst.tolist()
    lst.append(True) # to keep last column containing outuput variable, we add a last True row
    data = data.loc[:,lst]
    headers = data.columns
    
    return data, headers, corr


def outliers(data, cat_var): # not depurated function - Do not use
    headers=data.columns
    for cat in cat_var:
        if cat in headers:
            continuous_data=data.drop(columns=cat)

    for k, v in continuous_data.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
        print("Column {} outliers = {} => {}%".format(k,len(v_col),round((perc),3)))
        
        if len(v_col)>0:
            dat = data[k].values
            dat = dat.reshape(-1,1)
            dat_transform=power_transform(dat, method='box-cox')
            dat_transform=dat_transform.reshape(len(dat_transform),)
            data[k]=dat_transform
            #data[k]= np.log(data[k])
        
    for k, v in data.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
        print("Column {} outliers = {} => {}%".format(k,len(v_col),round((perc),3)))
    
    return data


    
###############################################################################
# var = ['ductus','Ritmo Extracción', 'Lactatorraquia prequirúrgica (Lactato en LCR)', 'Lactatorraquia durante Ommaya','DVP']

# var_s4 = ['ductus','Ritmo Extracción', 'Lactatorraquia durante Ommaya','DVP']

var = ['ductus','Ritmo Extracción','DVP']

var_s4 = ['ductus','Ritmo Extracción','DVP']


file = 'datos_V2.xls'

categorical_var = ['Sexo', 'Grado Hemorragia Clasificación de Papile','Punciones lumbares','ductus',
                   'Enterocolitis Necrotizante','sepsis','Mala respuesta ecográfica durante punciones']

output_var = 'DVP'


def main():
    warnings.filterwarnings("ignore")
    seeds = [4, 42, 128]
    corr_level = 0.3
    k_folds_list = list(range(2,11))
    data_results=[]
    
    Mdata = import_data(file,output_var)
    
    # Fill empty data
    Mdata1 = fill_na(Mdata, categorical_var)
    
    #Normalize X data (output var wont get normalized)
    Mdata1 = normalize(Mdata1)
            
    # Chose option A or B, not selecting any, means that you test with all variables
    ###############################3
    Mdata2, var_use, corr = correlation(Mdata1,'pearson',corr_level)   ## Option A - Var according to correltaion level
    Mdata3 = sel_var(Mdata1,var)                 ## Option B - Pre-defined variables
    Mdata4 = sel_var(Mdata1,var_s4)
    
    Mdatas = [Mdata1,Mdata2,Mdata3,Mdata4]
    scenario = 0
    
    for Mdata in Mdatas:
        scenario = scenario+1
        # Fill empty data
        # Mdata = fill_na(Mdata, categorical_var)
        if scenario==1:
            variables_usadas = 'all'
        elif scenario==2:
            variables_usadas = var_use
        elif scenario==3:
            variables_usadas = var
        
        # Find and soften outliers:
        # Mdata = outliers(Mdata,categorical_var)
        
        for k_folds in k_folds_list:
            for seed in seeds:
                # Shuffle it
                Mdata = shuffle(Mdata, seed)
                
                # Separar variable objetivo (ya) del resto (Xa)
                X = Mdata.iloc[:,0:-1] 
                y = Mdata.iloc[:,-1]
                
                #Normalize X data
                X = X.values
                
                # Find best K-neighbors, and best associated score
                best_k, best_weights, best_score = optimize_KNN(X, y, k_folds)
                
                # Build the model for double checking, with simple cross validation
                knn_model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
                cv_res = cross_val(X, y, knn_model, k_folds)
                
                # Building DataFrame with results of iterations
                data_results.append([scenario,variables_usadas,k_folds,seed,best_k,best_score,
                                     cv_res.mean()])
        
    data_results=pd.DataFrame(data_results)
    data_results.columns=['scenario','variables','K-folds','seed','Optimal K-neighbors','Gridsearch Acc','CV acc']
    with pd.ExcelWriter('results_KNN_V3_no_lactorraquia.xlsx') as writer:  
        data_results.to_excel(writer, sheet_name='results')
        corr.to_excel(writer, sheet_name='correlations')
    # data_results.to_excel('results_KNN_V3_with_newVAR.xlsx', sheet_name='results') 
    # corr.to_excel('results_KNN_V3_with_newVAR.xlsx', sheet_name='correlations')
           
    return data_results, corr


if __name__ == "__main__":
    results, correlacion = main()

# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model.ridge import Ridge
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.svm.classes import SVR
from sklearn.utils import shuffle
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
warnings.filterwarnings("ignore")


# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.constraints import maxnorm
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor


# In[3]:


## For each data concept we want to replace the missing values with its mean across the row; 
def imputeMissingDataWithMeanValue(data):
    columns = columnNameBuilder("Monthly Return Base Currency", 120)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Twelve Month Yield", 36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Fund Size aggr from share classes (Monthly) Base Currency",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("% Asset in Top Ten Holdings",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Asset Alloc Bond % (Long)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Asset Alloc Bond % (Short)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Asset Alloc Cash % (Long)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Asset Alloc Cash % (Short)",36)    
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Asset Alloc Equity % (Long)",36) 
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Asset Alloc Equity % (Short)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Asset Alloc Other % (Long)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Asset Alloc Other % (Short)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Equity Style Factor P/E (Long)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Equity Style Factor P/E (Short)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Equity Style Factor P/B (Long)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Equity Style Factor P/B (Short)",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Average Eff Duration",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Average Coupon",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Average Price",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = columnNameBuilder("Rating Overall",36)
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)    
    
    columns = ["Board of Directors Fee Amount Year Base Currency 32",
              "Board of Directors Fee Amount Year Base Currency 20",
              "Board of Directors Fee Amount Year Base Currency 8"]
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = ["Board of Directors Fee Year 32",
              "Board of Directors Fee Year 20",
              "Board of Directors Fee Year 8"]
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = ["Annual Report Net Expense Ratio Year 32",
               "Annual Report Net Expense Ratio Year 20",
               "Annual Report Net Expense Ratio Year 8"]
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    
    columns = ["Firm % Assets Longest Manager Tenure 0-3 Years",
              "Firm % Assets Longest Manager Tenure 12-15 Years",
              "Firm % Assets Longest Manager Tenure 3-6 Years",
              "Firm % Assets Longest Manager Tenure 6-9 Years",
              "Firm % Assets Longest Manager Tenure 9-12 Years",
              "Firm Manager Retention Rate 5 Year"]
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = [
              "Firm % Assets Manager Investment $0",
              "Firm % Assets Manager Investment $1-$10,000",
              "Firm % Assets Manager Investment $10,001-$50,000",
              "Firm % Assets Manager Investment $100,001-$500,000",
              "Firm % Assets Manager Investment $50,001-$100,000",
              "Firm % Assets Manager Investment $500,001-$1 million",
              "Firm % Assets Manager Investment Null",
              "Firm % Assets Manager Investment Over $1 million"]
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = [
              "Firm % Share Classes in Above Average Fee Level - Distribution",
              "Firm % Share Classes in Average Fee Level - Distribution",
              "Firm % Share Classes in Below Average Fee Level - Distribution",
              "Firm % Share Classes in High Fee Level - Distribution",
              "Firm % Share Classes in Low Fee Level - Distribution",
              "Firm % Assets Longest Manager Tenure 15+ Years",
              "Firm Average Fee Level - Distribution"]
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    columns = ["Turnover Ratio % Year 32",
               "Turnover Ratio % Year 20",
               "Turnover Ratio % Year 8"]    
    data.loc[:, columns] = imputeMissingDataWithMean(data.loc[:,columns].values)
    data.loc[:,"Primary Prospectus Benchmark"] = data.loc[:, "Primary Prospectus Benchmark"].fillna(' ')
    data.loc[:, "Portfolio ESG Score"] = data.loc[:, "Portfolio ESG Score"].fillna(0)
    data.loc[:, "Portfolio Controversy Score"] = data.loc[:, "Portfolio Controversy Score"].fillna(0)
    data.loc[:, "Portfolio Environmental Score"] = data.loc[:, "Portfolio Environmental Score"].fillna(0)
    data.loc[:, "Portfolio Governance Score"] = data.loc[:, "Portfolio Governance Score"].fillna(0)
    data.loc[:, "Portfolio Social Score"] = data.loc[:, "Portfolio Social Score"].fillna(0)
    data.loc[:, "Portfolio Sustainability Score"] = data.loc[:, "Portfolio Sustainability Score"].fillna(0)
    return data


# In[4]:


def columnNameBuilder(columnName, numberOfItems):
    columns = [ columnName + ' ' + str(i) for i in range(numberOfItems)]
    return columns


# In[5]:


## Missing value
# Return0, Return 10, Retunr 15= missing
# 120 data elements  Mean = sum(data elements 120) / (120-3)
def imputeMissingDataWithMean(data):
    return np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=1)[:, np.newaxis], data)


# In[6]:


def encodeData(data):
    
    balanced_objectives = ['Asset Allocation', 'Balanced', 'Equity-Income', 'Growth and Income', 
                           'Income', 'Multi-Asset Global']
    
    
    le = LabelEncoder()
    
    category = data["RankCategory"].values
    data["RankCategory"] = le.fit_transform(category)

    period = data["period"].values
    data["period"] = le.fit_transform(period)
    
    firm = data["Firm Name"].values
    data["Firm Name"] = le.fit_transform(firm)
    
    name = data["Name"].values
    data["Name"] = le.fit_transform(name)
    

    
    rankcategory = data["RankCategory"].values
    data["RankCategory"] = le.fit_transform(rankcategory)
    
    objective = data["Prospectus Objective"].values
    data["Prospectus Objective"] = le.fit_transform(objective)
    
    col_names = columnNameBuilder("Average Credit Quality", 36)
    for i in range(36):
        col_name = col_names[i]
        data[col_name].fillna('0', inplace=True)
        col_val = data[col_name].values
        data[col_name] = le.fit_transform(col_val)
        
    retire = data["Available For Retirement Plan"].values
    data["Available For Retirement Plan"] = le.fit_transform(retire)
    
    insurance = data["Available In Insurance Product"].values
    data["Available In Insurance Product"] = le.fit_transform(insurance)
    
    data['Primary Prospectus Benchmark'].fillna('0', inplace=True)
    benchmark = data["Primary Prospectus Benchmark"].values
    data["Primary Prospectus Benchmark"] = le.fit_transform(benchmark)

    return data


# In[7]:


def baseline_model_896(optimizer='adam', init='glorot_uniform'):
    # create model
    #optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model = Sequential()
    model.add(Dense(896, activation='relu', kernel_initializer = 'normal', input_shape=(896,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[8]:


def train_data_nn(X_train, y_train):
    
    np.random.seed(42)
    # create model
    estimator = KerasRegressor(build_fn=baseline_model_896, epochs=200, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, random_state=42)
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)  
    # grid search epochs, batch size and optimizer
    #optimizers = ['rmsprop', 'adam', 'sgd']
    #init = ['glorot_uniform', 'normal', 'uniform']
    #epochs = [50, 100, 150]
    #batches = [5, 10, 20]
    #param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
    #grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
    #grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))
    #return grid_result.best_estimator_
    print("RMSE:", results.std())
    return estimator


# In[9]:


def visualize_learning_curve(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[10]:


def train_and_predict_new(Xtrain, Xtest):
    X = Xtrain
    y = X['PeerRank']
    X.drop("PeerRank", inplace=True, axis=1) 
    
    null_cols = X.columns[X.isnull().all()]
    X.drop(null_cols, inplace=True, axis=1)
    nunique = X.apply(pd.Series.nunique)
    null_col_uni = nunique[nunique == 1].index
    X.drop(null_col_uni, inplace=True, axis=1)
    
    X_test = Xtest
    
    X_test.drop(null_cols, inplace=True, axis=1)   
    X_test.drop(null_col_uni, inplace=True, axis=1)

    print('Train size:', X.shape, ' Test size:', X_test.shape)
    X_train =X
    X_val = X
    y_train = y
    y_val = y

    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)
    X_val  = scale.fit_transform(X_val)
  
    #print(np.all(np.isfinite(X_train)))
    #print(np.all(np.isfinite(X_test)))
    
    #print(np.any(np.isnan(X_train)))
    #print(np.any(np.isnan(X_test)))
  
    #print(np.any(np.isnan(y.values)))


    estimator = train_data_nn(X_train, y_train)
    history = estimator.fit(X_val, y_val, validation_split=0.3, epochs=200, batch_size=5, verbose=0)
    #visualize_learning_curve(history)
    pred = estimator.predict(X_test)
    test_df = pd.DataFrame({'y_pred': pred})    
    return test_df


# In[ ]:


df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
#df_pred = pd.read_csv("./data/submission.csv")
#df_test['PeerRank'] = df_pred['y_pred'].values
train_num = len(df_train)
df_test.insert(0, 'PeerRank', 0)
dataset = pd.concat(objs=[df_train, df_test], axis=0)
dataset.drop(".id", axis=1, inplace=True)
dataset_shuffled = shuffle(dataset)
dataset = encodeData(dataset)
dataset = imputeMissingDataWithMeanValue(dataset)
dataset.fillna(0, inplace=True)
#df_train = dataset
df_train = dataset[:train_num]
#print(dataset.select_dtypes(include=['object']).dtypes)
df_test = dataset[train_num:]
df_test.drop('PeerRank', inplace=True, axis=1)
print("Train Data:", df_train.shape)
print("Test Data:", df_test.shape)


# In[ ]:


test_df = train_and_predict_new(df_train, df_test)


# In[ ]:


print(len(test_df))


# In[ ]:


submission = test_df
submission.sort_index(inplace=True)
submission.loc[submission['y_pred'] < 0, 'y_pred'] = 0
submission.loc[submission['y_pred'] > 100, 'y_pred'] = 100
submission.to_csv("./data/submission.csv", index=False)


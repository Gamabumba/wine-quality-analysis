#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from ydata_profiling import ProfileReport

import optuna
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ## Dataset Description  
# **country - The country that the wine is from  
# designation - The vineyard within the winery where the grapes that made the wine are from  
# points - The number of points WineEnthusiast rated the wine on a scale of 1-100  
# price - The cost for a bottle of the wine  
# province - The province or state that the wine is from  
# region_1 - The wine growing area in a province or state (ie Napa)  
# region_2 - Sometimes there are more specific regions specified within a wine growing area  
# title - The title of the wine review, which often contains the vintage if you're interested in extracting that feature  
# variety - The type of grapes used to make the wine (ie Pinot Noir)  
# winery - The winery that made the wine**

# In[2]:


get_ipython().run_line_magic('matplotlib', '')


# In[3]:


df = pd.read_csv('#\\winemag-data-130k-v2.csv')


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:


df.duplicated().sum()


# In[8]:


df.isna().sum()


# In[9]:


(df.isna().sum() / len(df)) * 100


# In[10]:


profile = ProfileReport(df, title="Wine Report", explorative=True)


# In[11]:


profile.to_file("wine_report.html")


# In[12]:


profile


# In[13]:


df.describe()


# **It can be seen that the ratings have outliers at both the upper and lower end. This is inevitable, as some wines can taste very bad and some very good.  
# The price also has outliers, which seems normal.**

# In[14]:


plt.figure(figsize=(16,5))
g = sns.countplot(x='points', data=df)


# **The scores are mostly concentrated between 86-90 points. But there are wines that received scores below 82 and 100 points.  
# The distribution tends to be normal**

# In[15]:


plt.figure(figsize=(16,9))
df['country'].value_counts().head(8).plot.bar()


# **Most ratings from Britain and France**

# In[16]:


df['price'].value_counts().sort_index().plot.area(
    figsize = (16,7),
    title = 'Prices of wine'
)


# In[17]:


plt.figure(figsize=(25,10))
g = sns.histplot(data = np.log(df['price']), kde = True)


# **The prices for most wines are generally in the 25-40 range.  
# However, there are wines that are very expensive.**

# In[18]:


df1 = df.copy()
df1['log_price'] = np.log(df1['price'])
plt.figure(figsize=(15, 10))
sns.boxenplot(data = df1, x="points", y="log_price", k_depth="trustworthy")


# **It can be seen that there is a significant positive correlation between price and the score a wine receives  
# Wines with a price range of 0 to 500 receive poor scores. Scores for cheaper wines are split between 80 and 100. This suggests that cheaper wines can be equally liked by all critics.**

# In[19]:


import missingno as msno


# In[20]:


msno.bar(df)


# In[21]:


msno.heatmap(df)


# In[22]:


#Working with nuls
df.fillna({'country' : 'US'}, inplace = True)

#Fills in the most frequent value by country/province
most_province = df.groupby('country')['province'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df['province'] = df.apply(lambda row: most_province[row['country']] if pd.isna(row['province']) else row['province'], axis=1)

most_region = df.groupby('country')['region_1'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df['region_1'] = df.apply(lambda row: most_region[row['country']] if pd.isna(row['region_1']) else row['region_1'], axis=1)

most_price = df.groupby('province')['price'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df['price'] = df.apply(lambda row: most_price[row['province']] if pd.isna(row['price']) else row['price'], axis=1)

most_designation = df.groupby('province')['designation'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df['designation'] = df.apply(lambda row: most_designation[row['province']] if pd.isna(row['designation']) else row['designation'], axis=1)


# In[23]:


#We fill it up with mode
df['variety'] = df['variety'].fillna(df['variety'].mode().squeeze())
df['region_1'] = df['region_1'].fillna(df['region_1'].mode().squeeze())
df['designation'] = df['designation'].fillna(df['designation'].mode().squeeze())


# In[24]:


df['log_price'] = np.log(df['price'])
df['log_price'] = df['log_price'].fillna(df['log_price'].mean().squeeze())
del df['price']


# In[25]:


del df['description']
del df['taster_name']
del df['taster_twitter_handle']
del df['Unnamed: 0']
del df['region_2']
del df['title']


# In[26]:


(df.isna().sum() / len(df)) * 100


# **Working with emissions**

# In[27]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='points')
plt.show()


# In[28]:


Q1 = df['points'].quantile(0.25) 
Q3 = df['points'].quantile(0.75)  
IQR = Q3 - Q1 

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['points'] >= lower_bound) & (df['points'] <= upper_bound)]

df.describe()


# **Coding of categorical**

# In[29]:


df.info()


# In[30]:


encoder = TargetEncoder()

df.loc[:, 'country_encoded'] = encoder.fit_transform(df['country'], df['points'])
df.loc[:, 'province_encoded'] = encoder.fit_transform(df['province'], df['points'])
df.loc[:, 'region_1_encoded'] = encoder.fit_transform(df['region_1'], df['points'])
df.loc[:, 'designation_encoded'] = encoder.fit_transform(df['designation'], df['points'])
df.loc[:, 'winery_encoded'] = encoder.fit_transform(df['winery'], df['points'])
df.loc[:, 'variety_encoded'] = encoder.fit_transform(df['variety'], df['points'])

df = df.drop(columns=['country', 'province', 'region_1', 'designation', 'winery', 'variety'])


# **Why not a one-hot encoder?  
# A large number of categories  
# Problem of semantic interpretation (the relationship between variables will not be taken into account)  
# For regression with a continuous target variable, methods that take into account the relationship of categories with the target feature are more convenient**

# In[31]:


df.info()


# In[32]:


SEED = 42

train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)


# In[33]:


X = train_df[['log_price', 'country_encoded', 'region_1_encoded', 'province_encoded', 'designation_encoded', 'winery_encoded', 'variety_encoded']]
y = train_df.points
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=SEED)


# In[36]:


def objective_xgboost(trial):
    params = {
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),  
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9), 
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),  
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
    }

    # Training a model with early stopping
    model = XGBRegressor(**params, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions and quality assessment
    pred_test = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, pred_test, squared=False)  # Вычисляем RMSE

    return rmse_test  # Target Metric for Minimization

cd_study = optuna.create_study(direction='minimize')
cd_study.optimize(objective_xgboost, n_trials=50, timeout=600)


# In[35]:


def objective_lightgbm(trial):
    params = {
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True), 
        "subsample": 1.0 if trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]) == "goss" else trial.suggest_float("subsample", 0.5, 1.0),  # Conditional for GOSS
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10) if trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]) != "goss" else 0,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True), 
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)  
    }

    
    
    model = LGBMRegressor(**params, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)]
    )
    
    pred_test = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, pred_test, squared=False)  # Вычисляем RMSE

    return rmse_test  

cd_study = optuna.create_study(direction='minimize')
cd_study.optimize(objective_lightgbm, n_trials=50)


# In[36]:


def objective_catboost(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10),
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "bagging_temperature": trial.suggest_uniform("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_uniform("random_strength", 1.0, 20.0),
    }
    model = CatBoostRegressor(**params, random_seed=42, verbose=0)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    pred_test = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, pred_test, squared=False)

    return rmse_test 

cd_study = optuna.create_study(direction='minimize')
cd_study.optimize(objective_catboost, n_trials=50)


# In[37]:


X_test = test_df[['log_price', 'country_encoded', 'region_1_encoded', 'province_encoded', 'designation_encoded', 'winery_encoded', 'variety_encoded']]
y_test = test_df.points


# In[42]:


params = {
    'booster': 'gbtree', 
          'learning_rate': 0.05220172077936295, 
          'n_estimators': 297, 
          'max_depth': 10, 
          'min_child_weight': 0.3473376654454266, 
          'gamma': 1.662119342958949e-06, 
          'subsample': 0.656951123747606, 
          'colsample_bytree': 0.8679843653587084, 
          'lambda': 7.064245725689333, 
          'alpha': 0.01640411004594726
         }

xgb = XGBRegressor(**params, random_state=SEED).fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print('Naive RMSE:', root_mean_squared_error(y_test, y_test - y_test.mean()))
print('RMSE:', root_mean_squared_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R^2:', r2_score(y_test, y_pred))


# In[43]:


params = {
    'boosting_type': 'gbdt',
          'num_leaves': 259, 
          'learning_rate': 0.12339718851328456, 
          'n_estimators': 121, 
          'max_depth': 19, 
          'min_child_samples': 7, 
          'min_child_weight': 0.017544130687354366, 
          'subsample': 0.9463886403247853, 
          'subsample_freq': 3, 
          'colsample_bytree': 0.55620773125197, 
          'reg_alpha': 0.8025361455431897, 
          'reg_lambda': 0.0016045313386671848
         }

lgb = LGBMRegressor(**params, random_state=SEED).fit(X_train, y_train)
y_pred = lgb.predict(X_test)

print('Naive RMSE:', root_mean_squared_error(y_test, y_test - y_test.mean()))
print('RMSE:', root_mean_squared_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R^2:', r2_score(y_test, y_pred))


# In[47]:


params = {
    'depth': 9, 
    'learning_rate': 0.12028135806659379, 
    'l2_leaf_reg': 0.7820130506963898, 
    'iterations': 998, 
    'bagging_temperature': 0.6294431235160689, 
    'random_strength': 12.182887407186382
}

cat = CatBoostRegressor(**params, random_state=SEED).fit(X_train, y_train, verbose=False)
y_pred = cat.predict(X_test)

print('Naive RMSE:', root_mean_squared_error(y_test, y_test - y_test.mean()))
print('RMSE:', root_mean_squared_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R^2:', r2_score(y_test, y_pred))


# In[45]:


import shap

explainer = shap.Explainer(lgb)
shap_values = explainer(X_train)

shap.summary_plot(shap_values, X_train, feature_names = X_train.columns.tolist())
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)


# <!-- Зеленый блок успеха -->
# <div style="border: 1px solid #4CAF50; padding: 10px; background-color: #e8f5e9; border-radius: 5px;">
#     <strong>Conclusion:</strong> lightGBM found to be the best model for predicting wine ratings
# </div>

# In[ ]:





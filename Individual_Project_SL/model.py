#!/usr/bin/env python
# coding: utf-8

# In[36]:


#get_ipython().run_line_magic('reset', '-f')


# In[37]:


import pandas as pd
import numpy as np
import random
import re
import gc

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score, KFold


# In[38]:


gc.collect()


# In[39]:


listings = pd.read_csv("train.csv")


# In[40]:


listings


# In[41]:


random.sample(list(listings["neighborhood_overview"].values), 1)


# In[42]:


positive_words = np.array(["close", "night life", "favorite", "famous",
                           "!", "amazing", "easy", "lovely", "easily",
                           "great", "shopping", "restaurants", "explore",
                           "historic", "hotspot", "relax",
                           "perfect", "unforgettable", "inspiring", "convenient",
                           "theaters", "attractions", "walking", "playground",
                           "park", "clubs", "quiet", "market",
                           "best", "library", "local", "sights",
                           "heart", "charming", "museum"])


# In[43]:


def parse_descriptions(descriptions):
    parsed_info = {
        'bedrooms': [],
        'beds': [],
        'baths': []
    }

    for description in descriptions:
        brms = re.split(r'(\d+)\s*bedroom',description)
        bds = re.split(r'(\d+)\s*bed(?:s\b)?(?!room)',description)
        bths = re.split(r'(\d+(\.\d+)?)\s*(shared|private)?\s*bath',description)

        # print(brms)
        # print(bds)
        # print(bths)

        if len(brms) > 1:
            parsed_info["bedrooms"].append(float(brms[1]))
        else:
            parsed_info["bedrooms"].append(0)
        if len(bds) > 1:
            parsed_info["beds"].append(float(bds[1]))
        else: 
            parsed_info["beds"].append(0)
        if len(bths) > 1:
            parsed_info["baths"].append(float(bths[1]))
        else:
            parsed_info["baths"].append(0)

    return parsed_info


# In[44]:


descriptions = listings["name"].values
apartment_stats  = parse_descriptions(descriptions)


# In[45]:


listings_cleaned = pd.concat([listings.drop(columns=["bedrooms"]), pd.DataFrame(apartment_stats)[["bedrooms", "baths"]]], axis=1)


# In[46]:


listings_cleaned["host_response_rate"] = listings_cleaned["host_response_rate"].apply(lambda r: float(r.split("%")[0]) if isinstance(r, str) else r)


# In[47]:


listings_cleaned.columns


# In[48]:


listings_cleaned["last_review"] = pd.to_datetime(listings_cleaned["last_review"])
most_recent_date = listings_cleaned["last_review"].max()
listings_cleaned["recency"] = 1+(most_recent_date-listings_cleaned["last_review"]).dt.days


# In[49]:


def apt_type(description):
    return description.split(" ")[0].upper()

listings_cleaned["apt_type"] = listings_cleaned["name"].apply(lambda d: apt_type(d))


# In[50]:


listings_cleaned["neighborhood_positivity"] = listings_cleaned["neighborhood_overview"].apply(
    lambda d: np.sum([word in d.lower() for word in positive_words]) if not pd.isna(d) else 0
)


# In[51]:


listings_cleaned["host_description_score"] = listings_cleaned["host_about"].apply(
    lambda d: len(d) if not pd.isna(d) else 0
)

listings_cleaned["host_since"] = pd.to_datetime(listings_cleaned["host_since"])
listings_cleaned["host_recency"] = 1+(most_recent_date-listings_cleaned["host_since"]).dt.days


# In[52]:


#Spatial dependency
neighborhood_prices = []

epsilon = 1
R = 6371

for i, data in listings_cleaned.iterrows():
    x = np.deg2rad(data["latitude"])
    y = np.deg2rad(data["longitude"])
    lat = np.deg2rad(listings_cleaned["latitude"].values)
    long = np.deg2rad(listings_cleaned["longitude"].values)
    xdiff = x-lat
    ydiff = y-long
    
    d = np.sin(xdiff / 2)**2 + np.cos(x) * np.cos(lat) * np.sin(ydiff / 2)**2
    c = 2 * np.arctan2(np.sqrt(d), np.sqrt(1-d))
    km = R*c

    indices = np.where(km <= epsilon)[0]
    neighborhood = indices[indices != i]
    neighborhood_price = np.mean(listings_cleaned.iloc[neighborhood]["price"])

    prices = neighborhood_prices.append(neighborhood_price)

listings_cleaned["neighborhood_price"] = neighborhood_prices


# In[53]:


listings_cleaned.columns


# In[54]:


dummies = ["host_location", "host_identity_verified", "property_type",
                  "room_type", "has_availability"]
continuous = [    "accommodates", "baths", "bedrooms", "beds", "recency", "neighborhood_price"]


# In[55]:


listings_cleaned[["price"]+continuous].corr()


# In[56]:


pipe_numeric = Pipeline(steps=[
  ('impute',KNNImputer()),
  ('standardize', StandardScaler())
])

pipe_categorical = Pipeline(steps=[
  ('impute',SimpleImputer(strategy='most_frequent')),
  ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", pipe_numeric, continuous),
        ("categorical", pipe_categorical, dummies)
    ]
)

knn_pipe = Pipeline(steps=[
    ("preprocessors", preprocessor),
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
    ('model', KNeighborsRegressor(n_neighbors=10, weights='distance'))
])


# In[57]:


params = {
  #'preprocessors__categorical__select_percentile__percentile': [50],
  'preprocessors__numerical__impute__n_neighbors': list(range(5, 101, 5)),
  'poly_features__degree': [3],
  'model__n_neighbors': list(range(5, 101, 5)),
  'model__weights': ['distance']
}

X = listings_cleaned[continuous+dummies]
Y = listings_cleaned["price"]
mae = make_scorer(mean_absolute_error, greater_is_better=False)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=486)
# knn_pipe.fit(X_train, Y_train)


# In[58]:


rs = RandomizedSearchCV(knn_pipe, param_distributions=params, scoring=mae, cv=5, random_state=486)


# In[59]:


#We don't have enough memory to run this
#rs.fit(X,Y)


# In[60]:


#rs.best_params_


# In[61]:


#KNN Model 1
cv = KFold(n_splits=5,
           shuffle=True,
           random_state=0)

pipe_numeric = Pipeline(steps=[
  ('impute',KNNImputer(n_neighbors=10)),
  ('standardize', StandardScaler())   
])

knn_pipe = Pipeline(steps=[
    ("preprocessors", preprocessor),
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
    ('model', KNeighborsRegressor(n_neighbors=10, weights='distance'))
])

# knn_pipe.fit(X_train, Y_train)
# knn_yhat = knn_pipe.predict(X_test)

#knn_cv_error = cross_val_score(knn_pipe, X, Y, cv=cv, scoring=mae)
#-np.mean(knn_cv_error)


# In[62]:


#KNN Model 2
cv = KFold(n_splits=5,
           shuffle=True,
           random_state=0)

pipe_numeric = Pipeline(steps=[
  ('impute',KNNImputer(n_neighbors=10)),
  ('standardize', StandardScaler())   
])

knn_pipe = Pipeline(steps=[
    ("preprocessors", preprocessor),
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
    ('model', KNeighborsRegressor(n_neighbors=20, weights='distance'))
])

# knn_pipe.fit(X_train, Y_train)
# knn_yhat = knn_pipe.predict(X_test)

#knn_cv_error = cross_val_score(knn_pipe, X, Y, cv=cv, scoring=mae)
#-np.mean(knn_cv_error)


# In[63]:


#KNN Model 3
cv = KFold(n_splits=10,
           shuffle=True,
           random_state=0)

pipe_numeric = Pipeline(steps=[
  ('impute',KNNImputer(n_neighbors=15)),
  ('standardize', StandardScaler())   
])

knn_pipe = Pipeline(steps=[
    ("preprocessors", preprocessor),
    ('poly_features', PolynomialFeatures(degree=1, include_bias=False)),
    ('model', KNeighborsRegressor(n_neighbors=15, weights='distance'))
])

# knn_pipe.fit(X_train, Y_train)
# knn_yhat = knn_pipe.predict(X_test)

#knn_cv_error = cross_val_score(knn_pipe, X, Y, cv=cv, scoring=mae)
#-np.mean(knn_cv_error)


# In[64]:


#Ridge regression
# alphas = np.logspace(-6, 6, 30)

# ridge_pipe = Pipeline(steps=[
#     ("preprocessors", preprocessor),
#     ('poly_features', PolynomialFeatures(degree=1, include_bias=False)),
#     ('model', RidgeCV(alphas=np.logspace(-6, 6, 30), cv=10))
# ])

# ridge_pipe.fit(X_train, Y_train)
# yhat = ridge_pipe.predict(X_test)
# mean_absolute_error(Y_test, yhat)


# In[65]:


#Lasso regression
# lasso_pipe = Pipeline(steps=[
#     ("preprocessors", preprocessor),
#     ('poly_features', PolynomialFeatures(degree=1, include_bias=False)),
#     ('model', LassoCV(alphas=np.logspace(-6, 6, 30), cv=10))
# ])

# lasso_pipe.fit(X_train, Y_train)
# lasso_yhat = ridge_pipe.predict(X_test)
# mean_absolute_error(Y_test, lasso_yhat)


# In[66]:


#Final model
pipe_numeric = Pipeline(steps=[
  ('impute',KNNImputer(n_neighbors=10)),
  ('standardize', StandardScaler())   
])

knn_pipe = Pipeline(steps=[
    ("preprocessors", preprocessor),
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
    ('model', KNeighborsRegressor(n_neighbors=10, weights='distance'))
])

test = pd.read_csv("test.csv")


# In[67]:


test_cleaned = pd.concat([test.drop(columns=["bedrooms"]), pd.DataFrame(parse_descriptions(test["name"].values))[["bedrooms", "baths"]]], axis=1)
test_cleaned["host_response_rate"] = test_cleaned["host_response_rate"].apply(lambda r: float(r.split("%")[0]) if isinstance(r, str) else r)
test_cleaned["last_review"] = pd.to_datetime(test_cleaned["last_review"])
most_recent_date = test_cleaned["last_review"].max()
test_cleaned["recency"] = 1+(most_recent_date-test_cleaned["last_review"]).dt.days
test_cleaned["apt_type"] = test_cleaned["name"].apply(lambda d: apt_type(d))
test_cleaned["neighborhood_positivity"] = test_cleaned["neighborhood_overview"].apply(
    lambda d: np.sum([word in d.lower() for word in positive_words]) if not pd.isna(d) else 0
)
test_cleaned["host_description_score"] = test_cleaned["host_about"].apply(
    lambda d: len(d) if not pd.isna(d) else 0
)

test_cleaned["host_since"] = pd.to_datetime(test_cleaned["host_since"])
test_cleaned["host_recency"] = 1+(most_recent_date-test_cleaned["host_since"]).dt.days

#Spatial dependency
neighborhood_prices = []

epsilon = 1
R = 6371

for i, data in test_cleaned.iterrows():
    x = np.deg2rad(data["latitude"])
    y = np.deg2rad(data["longitude"])
    lat = np.deg2rad(test_cleaned["latitude"].values)
    long = np.deg2rad(test_cleaned["longitude"].values)
    xdiff = x-lat
    ydiff = y-long
    
    d = np.sin(xdiff / 2)**2 + np.cos(x) * np.cos(lat) * np.sin(ydiff / 2)**2
    c = 2 * np.arctan2(np.sqrt(d), np.sqrt(1-d))
    km = R*c

    indices = np.where(km <= epsilon)[0]
    neighborhood = indices[indices != i]
    neighborhood_price = np.mean(listings_cleaned.iloc[neighborhood]["price"])

    prices = neighborhood_prices.append(neighborhood_price)

test_cleaned["neighborhood_price"] = neighborhood_prices


# In[31]:


Xtest = test_cleaned[continuous+dummies]

knn_pipe.fit(X, Y)
knn_pipe.predict(Xtest)


# In[63]:


Y


# ##### 

# #### 

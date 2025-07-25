
df = df.drop(columns=['remaining_lease', 'lease_commence_date'])
df = df.drop(columns = ['block', 'street_name'])


X = df.drop(columns='resale_price')
y = df['resale_price']



X

X.info()


y


y.info()


import sklearn
from sklearn.model_selection import train_test_split


X_train,X_temp, y_train, y_temp = train_test_split(X,y, test_size= 0.2, random_state = 42)


X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp, test_size= 0.5, random_state=42)


print('Training set shape:', X_train.shape, y_train.shape)
print('Validation set shape: ', X_val.shape, y_val.shape)
print('Test set shape: ', X_test.shape, y_test.shape)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



numerical_features = ['floor_area_sqm', 'year', 'remaining_lease_months']


numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])



from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder



nominal_features = ['month', 'town_name', 'flatm_name']


ordinal_features = ['flat_type']


flat_type_categories = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'
]

passthrough_features = ['storey_range']


nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])



ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=[flat_type_categories],
                               handle_unknown='use_encoded_value',unknown_value=-1))])




from sklearn.compose import ColumnTransformer



preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('nom', nominal_transformer, nominal_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('passthrough', 'passthrough', passthrough_features) 
    ],
    remainder='passthrough', n_jobs=-1 
)


preprocessor


from sklearn.linear_model import LinearRegression


lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


lr_pipeline


lr_pipeline.fit(X_train, y_train)


y_val_pred = lr_pipeline.predict(X_val)




y_val_pred


from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = root_mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)


print(f'Validation Mean Absolute Error (MAE): {val_mae}')
print(f'Validation Mean Squared Error (MSE): {val_mse}')
print(f'Validation Root Mean Squared Error (RMSE): {val_rmse}')
print(f'Validation R-squared (R2): {val_r2}')



from sklearn.linear_model import Ridge


ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1))  
])

ridge_pipeline.fit(X_train, y_train)


y_val_pred_ridge = ridge_pipeline.predict(X_val)


val_mae_ridge = mean_absolute_error(y_val, y_val_pred_ridge)
val_mse_ridge = mean_squared_error(y_val, y_val_pred_ridge)
val_rmse_ridge = root_mean_squared_error(y_val, y_val_pred_ridge)
val_r2_ridge = r2_score(y_val, y_val_pred_ridge)


print(f'Ridge Validation (MAE): {val_mae_ridge}')
print(f'Ridge Validation (MSE): {val_mse_ridge}')
print(f'Ridge Validation (RMSE): {val_rmse_ridge}')
print(f'Ridge Validation (R2): {val_r2_ridge}')


le_coefs = lr_pipeline.named_steps['regressor'].coef_


ridge_coefs = ridge_pipeline.named_steps['regressor'].coef_


feature_names = (
    numerical_features +
    list(ridge_pipeline.named_steps['preprocessor'].transformers_[1]
[1].named_steps['onehot'].get_feature_names_out(nominal_features)) + ordinal_features + passthrough_features)


coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Linear Regression Coefficients': le_coefs,
    'Ridge Regression Coefficients': ridge_coefs
})


coef_df = coef_df.melt(id_vars='Feature', var_name='Model', value_name='Coefficient')


plt.figure(figsize=(14, 8))
sns.barplot(data=coef_df, x='Feature', y='Coefficient', hue='Model', palette=['darkblue', 'orange'])
plt.title('Bar Plot of Coefficients for Linear and Ridge Regression')
plt.xlabel('Features')
plt.ylabel('Coefficient Magnitude')
plt.xticks(rotation=90)
plt.grid(True,axis = 'y')
plt.legend(loc='upper left')
plt.show()


from sklearn.linear_model import Lasso

lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=1, max_iter=5000))  
])

lasso_pipeline.fit(X_train, y_train)


y_val_pred_lasso = lasso_pipeline.predict(X_val)    


val_mae_lasso = mean_absolute_error(y_val, y_val_pred_lasso)    
val_mse_lasso = mean_squared_error(y_val, y_val_pred_lasso)
val_rmse_lasso = root_mean_squared_error(y_val, y_val_pred_lasso)
val_r2_lasso = r2_score(y_val, y_val_pred_lasso)


print(f'Lasso Validation (MAE): {val_mae_lasso}')
print(f'Lasso Validation (MSE): {val_mse_lasso}')
print(f'Lasso Validation (RMSE): {val_rmse_lasso}')
print(f'Lasso Validation (R2): {val_r2_lasso}')


from sklearn.linear_model import ElasticNet


en_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(l1_ratio=0.5))])

en_pipeline.fit(X_train, y_train)


y_val_pred_en = en_pipeline.predict(X_val)

y_val_en_mae = mean_absolute_error(y_val, y_val_pred_en)
y_val_en_mse = mean_squared_error(y_val, y_val_pred_en)
y_val_en_rmse = root_mean_squared_error(y_val, y_val_pred_en)
y_val_en_r2 = r2_score(y_val, y_val_pred_en)

print(f'ElasticNet Validation (MAE): {y_val_en_mae}')
print(f'ElasticNet Validation (MSE): {y_val_en_mse}')
print(f'ElasticNet Validation (RMSE): {y_val_en_rmse}')
print(f'ElasticNet Validation (R2): {y_val_en_r2}')




lr_coefs = lr_pipeline.named_steps['regressor'].coef_


lasso_coefs = lasso_pipeline.named_steps['regressor'].coef_


feature_names= (
    numerical_features +
    list(ridge_pipeline.named_steps['preprocessor'].transformers_[1]
[1].named_steps['onehot'].get_feature_names_out(nominal_features)) + ordinal_features + passthrough_features
)


coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Linear Regression': lr_coefs,
    'Lasso Regression': lasso_coefs
})


coef_df = coef_df.melt(id_vars='Feature', var_name='Model', value_name='Coefficient')


plt.figure(figsize=(14,8))
sns.barplot(data=coef_df, x='Feature', y='Coefficient', hue='Model', palette=['darkblue', 'orange'])

plt.xlabel('Feature')
plt.ylabel('Coefficient Magnitude')
plt.title('Bar Plot of Linear Regression and Lasso Regression Coefficients')
plt.xticks(rotation=90)
plt.grid(True,axis = 'y')
plt.legend(loc='upper left')
plt.show()


from sklearn.model_selection import GridSearchCV

param_grid = {
    'regressor__alpha': [0.1, 1, 10, 100, 1000],
    'regressor__fit_intercept': [True, False]
}



ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

lasso_pipeline=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso())
])


ridge_grid_search = GridSearchCV(ridge_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
ridge_grid_search.fit(X_train,y_train)


lasso_grid_search = GridSearchCV(lasso_pipeline,param_grid, cv=5, scoring='r2', n_jobs=-1 )
lasso_grid_search.fit(X_train,y_train)



print('Best Ridge Parameters:' ,ridge_grid_search.best_params_)


print('Best Lasso PArameters: ', lasso_grid_search.best_params_)


best_ridge_model = ridge_grid_search.best_estimator_
y_val_pred_ridge = best_ridge_model.predict(X_val)


val_mae_ridge = mean_absolute_error(y_val, y_val_pred_ridge)
val_mse_ridge = mean_squared_error(y_val, y_val_pred_ridge)
val_rmse_ridge = root_mean_squared_error(y_val, y_val_pred_ridge)
val_r2_ridge = r2_score(y_val, y_val_pred_ridge)

print(f'Ridge Validation MAE: {val_mae_ridge}')
print(f'Ridege Validation MSE: {val_mse_ridge}')
print(f'Ridge validation RMSE: {val_rmse_ridge}')
print(f'Ridge Validation r2: {val_r2_ridge}')



best_lasso_model = lasso_grid_search.best_estimator_
y_val_pred_lasso = best_lasso_model.predict(X_val)

val_mae_lasso = mean_absolute_error(y_val,y_val_pred_lasso)
val_mse_lasso = mean_squared_error(y_val,y_val_pred_lasso)
val_rmse_lasso = root_mean_squared_error(y_val,y_val_pred_lasso)
val_r2_lasso = r2_score(y_val, y_val_pred_lasso)

print(f'Lasso Validation MAE: {val_mae_lasso}')
print(f'Lasso Validation MSE: {val_mse_lasso}')
print(f'Lasso Validation RMSE: {val_rmse_lasso}')
print(f'Lasso Validation r2: {val_r2_lasso}')


from sklearn.model_selection import RandomizedSearchCV


param_grid = {
    'regressor__alpha': [0.1, 1, 10, 100, 1000],
    'regressor__fit_intercept': [True, False]

}


ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])


lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso())
])


ridge_random_search = RandomizedSearchCV(ridge_pipeline, param_distributions=param_grid, n_iter=10, cv=5, scoring='r2', random_state=42, n_jobs=-1) 
ridge_random_search.fit(X_train, y_train)


lasso_random_search = RandomizedSearchCV(lasso_pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
lasso_random_search.fit(X_train, y_train)



best_lasso_model = lasso_grid_search.best_estimator_


y_test_pred_lasso = best_lasso_model.predict(X_test)


test_mae_lasso = mean_absolute_error(y_test, y_test_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
test_rmse_lasso = root_mean_squared_error(y_test, y_test_pred_lasso)
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)


print('Best Lasso Regression Model, Final Test Metrics:')
print(f'Lasso Validation MAE: {test_mae_lasso}')
print(f'Lasso Validation MSE: {test_mse_lasso}')
print(f'Lasso Validation RMSE: {test_rmse_lasso}')
print(f'Lasso Validation r2: {test_r2_lasso}')


best_ridge_model = ridge_grid_search.best_estimator_


y_test_pred_ridge = best_ridge_model.predict(X_test)


test_mae_ridge = mean_absolute_error(y_test,y_test_pred_ridge)
test_mse_ridge = mean_squared_error(y_test,y_test_pred_ridge)
test_rmse_ridge = root_mean_squared_error(y_test,y_test_pred_ridge)
test_r2_ridge = r2_score(y_test,y_test_pred_ridge)


print('Best Ridge Regression Model, Final Test Metrics:')
print(f'Ridge Validation MAE: {test_mae_ridge}')
print(f'Ridge Validation MSE: {test_mse_ridge}')
print(f'Ridge Validation RMSE: {test_rmse_ridge}')
print(f'Ridge Validation r2: {test_r2_ridge}')





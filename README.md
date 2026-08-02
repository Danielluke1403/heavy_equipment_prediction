# Heavy Machinery Price Prediction

This project is my submission for the Machine Learning Practice (MLP) course.
The task is to predict the resale price of used heavy equipment (like
excavators, bulldozers, etc.) based on details about the machine and how it
was used.

## What this project does

Given information about a piece of heavy equipment (manufacture year, hours
used, vendor, machine type, and so on), the goal is to predict its target
sale price. This is a regression problem.

## Steps followed in the notebook

1. **Exploratory Data Analysis (EDA)** - looked at missing values, data types,
   target distribution, and relationships between features using charts
2. **Handling missing values**
   - Categorical columns: missing values filled with "Unknown"
   - Numerical columns: left as missing on purpose, since the models used
     (CatBoost, LightGBM, XGBoost) can handle missing values internally
3. **Feature engineering** - created new features such as `AssetAge`,
   `UsagePerYear`, `HoursPerYear`, and extracted the transaction year, month,
   and quarter from the date column
4. **Encoding categorical features** - handled differently depending on the
   model, since each library supports categories in its own way:
   - CatBoost: used its built-in `cat_features` support
   - LightGBM: categorical columns converted to pandas `category` dtype
   - XGBoost: used `OrdinalEncoder` inside a `ColumnTransformer`, since
     XGBoost cannot take raw text categories directly
5. **No feature scaling** - not needed, since all three models used are
   tree-based and are not affected by the scale of the input features
6. **Model building** - trained and compared three models:
   - CatBoost
   - LightGBM
   - XGBoost
7. **Hyperparameter tuning** - used `RandomizedSearchCV` on the LightGBM model
8. **Model comparison** - compared all models using RMSLE (Root Mean Squared
   Log Error) and picked the best performing combination

## Notes

- Only the XGBoost model uses a formal scikit-learn `Pipeline`, since
  CatBoost and LightGBM handle categorical features natively and did not
  need a `ColumnTransformer`
- No data leakage was introduced: all feature engineering was applied
  separately to the train and test sets, and any fitted preprocessing (like
  the XGBoost encoder) was fit only on the training data

## Author
  Daniel Luke 

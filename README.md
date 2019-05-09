# Credit default risk model
The following is a modelling test task I did for a finance company. The data was provided by them, and since it is not mine it is withheld here.

## Summary
This report describes the creation of a model designed to predict the probability of credit default of loan customers. The model is based on a structured data set containing 1200 customers whose default outcome is known. For code, details, and further comments, see the separate Jupyter notebook.

The modelling procedure consisted of three main steps. First, a rough analysis of the data and its features was made. Using the conclusions from this analysis, the data was cleaned and some feature engineering was done in order to prepare the data for the model. Finally, models were trained and evaluated, and parameters were tuned. A logistic regression model was used as a benchmark, but mainly various XGBoost models were tried out. The model I finally settled for is an isotonically calibrated XGBoost model trained on 80 percent of the data and tested on the rest. It achieves a Brier score of 0.104, an accuracy of 0.896, and an area under the ROC curve of 0.860.

## Data exploration
In order to form a general impression of the data, some examples were first ocularly inspected. Then elementary properties and statistics of the data and its features were considered; for instance the data types of the features, the number of missing values for each feature, and statistics such as mean value and quartiles where applicable. After this, crude histograms were drawn in order to illustrate the relationship between the features and the target. Again, the purpose of these plots was not to highlight any particularly important aspect of the data, but rather to inform further considerations. Finally, some features were studied in more detail. 

Most features were straightforward continuous (float) features that can be left unaltered, for example the 'external_score' and 'income_tax' features. A few were straightforward categorical features, but several were float features which clearly should be categorical, for example the 'email_domain' and 'lender_id'} features; the former presumably representing the domain of the customer's email address, and the latter the lender of the loan that the customer seeks to get Anyfin to assist with. 

The 'customer_postal' feature (the postal code of the customer) required some special consideration: it should be transformed into a categorical feature, but clearly has too many categories as is. A few features turned out to be completely non-informative, such as 'debt_request_count', 'housing_rent', and 'housing_cost' since all customers had the same values. 

Note: Throughout, it was assumed that the features were reasonably named, for instance that the 'income_tax' feature represents the income tax of the customer in a certain time period. 

## Data cleaning and feature engineering
Now the data was processed in accordance with the above findings. The non-informative features were dropped, and the presently continuous features that should be categorical were transformed into categorical features and subsequently one-hot encoded. The 'customer_postal' feature in particular was handled by extracting the first digit of the postal code as a category, and the missing values were replaced with an "unknown" value. 

After the missing 'customer_postal' values had been taken care of, there were only a couple of examples left with missing values, so these were simply dropped. (In case of values missing in more examples it would have been worthwhile to impute them by for instance median values.) 

The 'e_mal_count','a_mal_count','e_mal_active_amount', and 'a_mal_active_amount' features had the property that any non-zero value led to a default, so these were turned into categorical variables with only values "zero" and "non-zero". 

I did not spot any obviously out of bounds values (for example customer ages of several hundreds of years old) during the exploration phase, and so did remove any examples for such a reason.

## Models and metrics
The XGBoost model was chosen as the main model because of it's well-known performance on structured data, although a simple logistic regression was also trained as a benchmark. 

Since the objective was to predict probabilities, the models were optimized for the binary logistic loss function, and the Brier score was used to decide between models and parameter choices. In order to evaluate different options and then get final performance metrics, a train/validation/ test split of 60/20/20 percent was used. Well-performing parameters were found via a grid search, and a plain XGBoost model was compared to XGBoost models that were calibrated using the sigmoid and the isotonic methods respectively. Calibration can be used to improve probability estimates, and indeed the isotonically calibrated model outperformed the other models on the validation set. The isotonic method is in general prone to overfit on small data sets, but I saw no sign of this in the performance on neither the validation set nor the test set. 

The final result (trained on everything but the test set) was a Brier score of 0.104, an accuracy of 0.896, and an area under the ROC curve of 0.860 for the isotonically calibrated XGBoost model. This can be compared to a Brier score of 0.154, an accuracy of 0.846, and an area under the ROC curve of 0.788 for the benchmark logistic regression model. 

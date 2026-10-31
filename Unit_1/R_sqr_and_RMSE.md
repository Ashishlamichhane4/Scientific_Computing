# R² Score Metric

The R² score measures how well a model's predictions explain the variation in the actual target values.
## Why Do We Use the R² Score?

We use the R² score to understand how well a regression model captures the differences in actual target values, rather than merely predicting an average value. It helps us evaluate whether the model has learned useful patterns and relationships between the input features and the target variable.

By comparing the model's predictions with the actual values, we can determine how well the model explains the variation in the target variable. We can also use the R² score to compare different regression models and identify which model performs better on the same dataset.

However, a high R² score does not necessarily mean that the model will make accurate predictions on new, unseen data. Therefore, we should evaluate the model using a separate test dataset to understand how well it generalizes.


### Mathematical Formula

The R² score is calculated using:

$$
R^2 = 1 - \frac{SSE}{SST}
$$

Where:

- **SSE (Sum of Squared Errors):** Measures the unexplained variation between actual and predicted values.

$$
SSE = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

- **SST (Total Sum of Squares):** Measures the total variation of actual values around their mean.

$$
SST = \sum_{i=1}^{n}(y_i-\bar{y})^2
$$


### Interpretation

- **R² = 1:** Perfect predictions.
- **R² = 0:** Model performs as well as predicting the mean.
- **R² < 0:** Model performs worse than predicting the mean.
- **R² = 0.75:** Model explains 75% of the variation in the target variable.


## RMSE (Root Mean Squared Error)

RMSE measures the magnitude of prediction errors made by a regression model. It calculates the square root of the average squared differences between actual and predicted values.

### Mathematical Formula

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

Where:

- **n:** Number of observations.
- **yᵢ:** Actual value.
- **ŷᵢ:** Predicted value.

### Calculation Steps

1. Calculate the difference between actual and predicted values.
2. Square each difference.
3. Calculate the mean of the squared errors (MSE).
4. Take the square root of MSE.

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

$$
RMSE = \sqrt{MSE}
$$

### Interpretation

- **Lower RMSE:** Generally indicates better predictions.
- **RMSE = 0:** Perfect predictions.
- RMSE is expressed in the same units as the target variable.
- Larger prediction errors receive more penalty because the errors are squared.

### R² vs RMSE

| R² Score | RMSE |
|---|---|
| Measures variation explained by the model. | Measures prediction error. |
| Higher is generally better. | Lower is better. |
| Has no unit. | Uses the same unit as the target. |

**Important:** R² tells us how well the model explains variation in the target variable, whereas RMSE tells us the magnitude of its prediction errors.

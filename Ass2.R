# Load necessary libraries
library(ggplot2)
library(GGally)
library(lubridate)
library(reshape2)
library(dplyr)
library(tidyr)
library(corrplot)
library(zoo)
library(forecast)
library(tseries)
library(prophet)
library(rugarch)
library(TSA)
library(tsDyn)
library(caret)
library(Metrics)
library(vars)
library(randomForest)
library(nnet)
library(xgboost)
library(keras)
library(tensorflow)

#Read CSV file
msft <- read.csv("MSFT.csv")
#Check the data
msft

#=========================================================

#2.1 Data Overview
print("Number of observations: ")
nrow(msft)

print("Time span: ")
range(msft$Date)

print("Key variables: ")
names(msft)

#Data format
str(msft)

#=========================================================

#2.2 Data Quality Checking
#Check missing values
print("Count of total missing values: ")
colSums(is.na(msft))

#Data Summary
summary(msft)

#Addressing missing values
#msft <- na.omit(msft)
#This is used to overcome the missing values.
#However, there is no missing values in this dataset. 
#Thus, we no need to address it.

data <- msft$Adj.Close

#Handling outliers
Q1 <- quantile(data, 0.25)
Q3 <- quantile(data, 0.75)
IQR_value <- Q3 - Q1

lower_bound <- Q1 - 1.5 * IQR_value
upper_bound <- Q3 + 1.5 * IQR_value

outliers <- msft[data < lower_bound | data > upper_bound, ]
msft <- msft[!(data < lower_bound | data > upper_bound), ]

print("Outliers detected:")
print(outliers)

print("Data after removing outliers:")
print(head(msft)) 

ggplot(data.frame(value = data), aes(x = value)) +
  geom_boxplot() +
  ggtitle("Boxplot after Handling Outliers")

#=========================================================

#Data Preparation
df <- data.frame(Date = msft$Date, Adj.Close = msft$Adj.Close)
df

#Convert the selected column into time series data
start_date <- min(df$Date)
start_year <- year(start_date)
start_month <- month(start_date)

Y <- ts(df$Adj.Close, frequency = 365, start = c(start_year, start_month))

#=========================================================

#2.3 Data Visualization
#1. Time Series Plots
plot(Y, xlab = "Year", ylab = "Adjusted Close Price ($millions)", main = "Adjusted Close Price from March 1986 to March 2022")

#2. Histogram
ggplot(df, aes(x = Adj.Close)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 1, alpha = 0.5, position = "identity") +
  geom_density(alpha = 0.5) + 
  labs(title = "Histogram of Adjusted Close Prices with Kernel Density Estimate", 
       x = "Adjusted Price",
       y = "Density")

#3. Heatmap of Correlation
corr_matrix <- cor(msft %>% select_if(is.numeric))

corrplot(corr_matrix, method = "color", type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 45, addCoef.col = "black")

#=========================================================

#3.0 Exploratory Data Analysis (EDA)
#3.1 Trend Analysis
#1. Addictive Decomposition
additive <- decompose(Y, type = "additive")
plot(additive)

#2. Multiplicative Decomposition
multiplicative <- decompose(Y, type = "multiplicative")
plot(multiplicative)

#3. 12-month Moving Average
ma_trend <- ma(Y, order = 12)
plot(Y, xlab = "Year", ylab = "Adjusted Close Price ($millions)",
     main = "Adjusted Close Price with 12-Month Moving Average")
lines(ma_trend, col = "black")
legend("topright", legend = "12-Month Moving Average", col = "black", lty = 1)

#=========================================================

#3.2 Seasonality Analysis
#1. Seasonal Decomposition of Time Series by Loess (STL)
adj.close_stl <- stl(Y, s.window = "periodic")
plot(adj.close_stl)

#2. Moving Average for Seasonal Analysis
seasonal <- additive$seasonal
plot(seasonal, xlab = "Year", ylab = "Seasonal Component", main = "Seasonal Component of Adjusted Close Price")

#=========================================================

#3.3 Stationary Tests
#3.3.1 ADF Test
adf.test(Y)

#3.3.2 KPSS Test
kpss.test(Y)

#=========================================================

#3.4 Autocorrelation and Partial Autocorrelation
#ACF
acf(Y, main = "ACF for Adj.Close Prices")

#PACF
pacf(Y, main = "PACF for Adj.Close Prices")

#=========================================================

#3.5 Transformation
trans <- log(Y)

plot(trans, xlab = "Year", main = "Adjusted Close Price from March 1986 to March 2022 after transformation")

#Since the difference between transformation and original is significant, we opt to transfrom the original data set

Y <- log(Y)

#=========================================================

#4.0 Train Test Split
#Define the percentage for trainingdata
train_percentage <- 0.8

#Calculate the number of rows for the training set
num_train_rows <- floor(nrow(df) * train_percentage)

#Create training data set
train <- df[1:num_train_rows, ]
train

#Create test data set
test <- df[(num_train_rows + 1):nrow(df), ]
test

train_ts <- window(Y, end = c(start_year, start_month + num_train_rows - 1))
test_ts <- window(Y, start = c(start_year, start_month + num_train_rows))

train$Adj.Close <- as.numeric(as.character(train$Adj.Close))

#=========================================================

#Preparing for Evaluation

calculate_mape <- function(actual, predicted) {
  return(mean(abs((actual - predicted) / actual)) * 100)
}

calculate_mae <- function(actual, predicted) {
  return(mean(abs(actual - predicted)))
}

calculate_rmse <- function(actual, predicted) {
  return(sqrt(mean((actual - predicted)^2)))
}

#=========================================================

#4.1 Model Identification and Model Fitting
#1. ARIMA Model
arima_fit <- auto.arima(train$Adj.Close, ic = "aic", trace = TRUE)
arima_forecast_fit <- forecast(arima_fit, h = length(test_ts))$mean
summary(arima_fit)
checkresiduals(arima_fit)

#---------------------------------------------------------------------------------------

#2. Seasonal Arima (SARIMA)
sarima_fit <- auto.arima(train$Adj.Close, seasonal = TRUE)
summary(sarima_fit)
checkresiduals(sarima_fit)
sarima_forecast <- forecast(sarima_fit, h = length(test_ts))$mean
sarima_forecast_ts <- ts(sarima_forecast, start = start(test_ts), frequency = 365)

#Visualize
acf(residuals(sarima_fit), main = "ACF of Residuals")
pacf(residuals(sarima_fit), main = "PACF of Residuals")
adf.test(residuals(sarima_fit))
kpss.test(residuals(sarima_fit))

#Calculate important metrics
mape_value <- calculate_mape(test_ts, sarima_forecast_ts)
mae_value <- calculate_mae(test_ts, sarima_forecast_ts)
rmse_value <- calculate_rmse(test_ts, sarima_forecast_ts)
adf_p <- adf.test(residuals(sarima_fit))$p.value
kpss_p <- kpss.test(residuals(sarima_fit))$p.value

#Combine results into a metrics dataframe
sarima_metrics <- data.frame(
  Model = "SARIMA",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#---------------------------------------------------------------------------------------

#3. Exponential Smoothing (Holt-Winters)
hw_train <- ts(train$Adj.Close, frequency = 365)
hw_fit <- HoltWinters(hw_train, alpha = 0.2, beta = 0.1, gamma = 0.1, seasonal = "additive")
summary(hw_fit)
checkresiduals(hw_fit)
hw_forecast <- forecast(hw_fit, h = length(test_ts))$mean
hw_forecast_ts <- ts(hw_forecast, start = start(test_ts), frequency = 365)

#Visualize
acf(residuals(hw_fit), main = "ACF of Residuals")
pacf(residuals(hw_fit), main = "PACF of Residuals")
adf.test(residuals(hw_fit))
kpss.test(residuals(hw_fit))

#Calculate important metrics
mape_value <- calculate_mape(test_ts, hw_forecast_ts)
mae_value <- calculate_mae(test_ts, hw_forecast_ts)
rmse_value <- calculate_rmse(test_ts, hw_forecast_ts)
adf_p <- adf.test(residuals(hw_fit))$p.value
kpss_p <- kpss.test(residuals(hw_fit))$p.value

#Combine results into a metrics dataframe
hw_metrics <- data.frame(
  Model = "Holt-Winters",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#---------------------------------------------------------------------------------------

#4. Prophet
prophet_train <- data.frame(ds = train$Date, y = train$Adj.Close)
prophet_fit <- prophet(prophet_train, daily.seasonality=TRUE)
future <- make_future_dataframe(prophet_fit, periods = length(test_ts))
prophet_forecast <- predict(prophet_fit, future)
prophet_forecast_ts <- ts(prophet_forecast$yhat[(nrow(prophet_train)+1):nrow(prophet_forecast)], start = start(test_ts), frequency = 365)
residuals_prophet <- prophet_train$y - prophet_forecast$yhat[1:nrow(prophet_train)]
checkresiduals(residuals_prophet)

#Visualize
acf(residuals_prophet, main = "ACF of Residuals")
pacf(residuals_prophet, main = "PACF of Residuals")
adf.test(residuals_prophet)
kpss.test(residuals_prophet)

#Calculate important metrics
mape_value <- calculate_mape(test_ts, prophet_forecast_ts)
mae_value <- calculate_mae(test_ts, prophet_forecast_ts)
rmse_value <- calculate_rmse(test_ts, prophet_forecast_ts)
adf_p <- adf.test(residuals_prophet)$p.value
kpss_p <- kpss.test(residuals_prophet)$p.value

#Combine results into a metrics dataframe
prophet_metrics <- data.frame(
  Model = "Prophet",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#---------------------------------------------------------------------------------------
#5. GARCH Model
spec_open <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                        mean.model = list(armaOrder = c(1, 1), include.mean = TRUE))
garch_fit <- ugarchfit(spec_open, train$Adj.Close)
summary(garch_fit)
garch_forecast <- ugarchforecast(garch_fit, n.ahead = length(test_ts))
garch_forecast <- fitted(garch_forecast)
garch_forecast_ts <- ts(garch_forecast, start = start(test_ts), frequency = 365)
residuals_garch <- residuals(garch_fit, standardize = TRUE)
residuals_garch <- ts(residuals_garch, frequency = 365)
checkresiduals(residuals_garch)

#Visualize
acf(residuals_garch, main = "ACF of Residuals")
pacf(residuals_garch, main = "PACF of Residuals")
adf.test(residuals_garch)
kpss.test(residuals_garch)

#Calculate important metrics
mape_value <- calculate_mape(test_ts, garch_forecast_ts)
mae_value <- calculate_mae(test_ts, garch_forecast_ts)
rmse_value <- calculate_rmse(test_ts, garch_forecast_ts)
adf_p <- adf.test(residuals_garch)$p.value
kpss_p <- kpss.test(residuals_garch)$p.value

#Combine results into a metrics dataframe
garch_metrics <- data.frame(
  Model = "GARCH",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#---------------------------------------------------------------------------------------

#6. TBATS Model
tbats_fit <- tbats(train$Adj.Close)
summary(tbats_fit)
tbats_forecast <- forecast(tbats_fit, h = length(test_ts))$mean
tbats_forecast_ts <- ts(tbats_forecast, start = start(test_ts), frequency = 365)
checkresiduals(residuals(tbats_fit))

#Visualize
acf(residuals(tbats_fit), main = "ACF of Residuals")
pacf(residuals(tbats_fit), main = "PACF of Residuals")
adf.test(residuals(tbats_fit))
kpss.test(residuals(tbats_fit))

#Calculate important metrics
mape_value <- calculate_mape(test_ts, tbats_forecast_ts)
mae_value <- calculate_mae(test_ts, tbats_forecast_ts)
rmse_value <- calculate_rmse(test_ts, tbats_forecast_ts)
adf_p <- adf.test(residuals(tbats_fit))$p.value
kpss_p <- kpss.test(residuals(tbats_fit))$p.value

#Combine results into a metrics dataframe
tbats_metrics <- data.frame(
  Model = "TBATS",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#---------------------------------------------------------------------------------------

#7. Random Forest
rf_fit <- randomForest(Adj.Close ~ Date, data = train)
summary(rf_fit)
rf_forecast <- predict(rf_fit, newdata = test)
rf_forecast_ts <- ts(rf_forecast, start = start(test_ts), frequency = 365)
residuals_rf <- train$Adj.Close - predict(rf_fit, newdata = train)
checkresiduals(residuals_rf)

#Visualize
acf(residuals_rf, main = "ACF of Residuals")
pacf(residuals_rf, main = "PACF of Residuals")
adf.test(residuals_rf)
kpss.test(residuals_rf)

#Calculate important metrics
mape_value <- calculate_mape(test_ts, rf_forecast_ts)
mae_value <- calculate_mae(test_ts, rf_forecast_ts)
rmse_value <- calculate_rmse(test_ts, rf_forecast_ts)
adf_p <- adf.test(residuals_rf)$p.value
kpss_p <- kpss.test(residuals_rf)$p.value

#Combine results into a metrics dataframe
rf_metrics <- data.frame(
  Model = "Random Forest",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#---------------------------------------------------------------------------------------

#8. Neural Networks (NNETAR)
nn_fit <- nnetar(train$Adj.Close)
summary(nn_fit)
nn_forecast <- forecast(nn_fit, h = length(test_ts))$mean
nn_forecast_ts <- ts(nn_forecast, start = start(test_ts), frequency = 365)
residuals_nn <- residuals(nn_fit)
residuals_nn <- na.omit(residuals_nn)
checkresiduals(residuals_nn)

#Visualize
acf(residuals_nn, main = "ACF of Residuals")
pacf(residuals_nn, main = "PACF of Residuals")
adf.test(residuals_nn)
kpss.test(residuals_nn)

#Calculate important metrics
mape_value <- calculate_mape(test_ts, nn_forecast_ts)
mae_value <- calculate_mae(test_ts, nn_forecast_ts)
rmse_value <- calculate_rmse(test_ts, nn_forecast_ts)
adf_p <- adf.test(residuals_nn)$p.value
kpss_p <- kpss.test(residuals_nn)$p.value

#Combine results into a metrics dataframe
nn_metrics <- data.frame(
  Model = "Neural Networks",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#---------------------------------------------------------------------------------------

#9. STL Decomposition
stl_fit <- stl(train_ts, s.window = "periodic")
stl_forecast <- forecast(stl_fit, h = length(test_ts))$mean
stl_forecast_ts <- ts(stl_forecast, start = start(test_ts), frequency = 365)
residuals_stl <- as.numeric(stl_fit$time.series[, "remainder"])
residuals_stl <- na.omit(residuals_stl)
residuals_stl <- residuals_stl[is.finite(residuals_stl)]
checkresiduals(residuals_stl)

#Visualize
acf(residuals_stl, main = "ACF of Residuals")
pacf(residuals_stl, main = "PACF of Residuals")
adf.test(residuals_stl)
kpss.test(residuals_stl)

#Calculate important metrics
mape_value <- calculate_mape(test_ts, stl_forecast_ts)
mae_value <- calculate_mae(test_ts, stl_forecast_ts)
rmse_value <- calculate_rmse(test_ts, stl_forecast_ts)
adf_p <- adf.test(residuals_stl)$p.value
kpss_p <- kpss.test(residuals_stl)$p.value

#Combine results into a metrics dataframe
stl_metrics <- data.frame(
  Model = "STL Decomposition",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#------------------------------------------------------------

#10. ETS Model
stlf_fit <- stlf(train_ts, h = length(test_ts))
stlf_forecast <- forecast(stlf_fit)
stlf_forecast_ts <- ts(stlf_forecast$mean, start = start(test_ts), frequency = 365)
summary(stlf_fit)
checkresiduals(residuals(stlf_fit), lag = 1)

#Calculate important metrics
mape_value <- calculate_mape(test_ts, stlf_forecast_ts)
mae_value <- calculate_mae(test_ts, stlf_forecast_ts)
rmse_value <- calculate_rmse(test_ts, stlf_forecast_ts)
adf_p <- adf.test(residuals(stlf_fit))$p.value
kpss_p <- kpss.test(residuals(stlf_fit))$p.value

#Combine results into a metrics dataframe
stlf_metrics <- data.frame(
  Model = "STLF",
  MAPE = mape_value,
  MAE = mae_value,
  RMSE = rmse_value,
  ADF_pvalue = adf_p,
  KPSS_pvalue = kpss_p
)

#------------------------------------------------------------

#11. Additive Model
additive_fit <- tslm(train_ts ~ trend + season)
summary(additive_fit)
checkresiduals(residuals(additive_fit), lag = 1)

#=========================================================

#5.1 Residual Diagnostics 
#5.1.1 White Noise Test
Box.test(residuals(arima_fit), type = "Ljung-Box")
Box.test(residuals(sarima_fit), type = "Ljung-Box")
Box.test(residuals(hw_fit), type = "Ljung-Box")
Box.test(residuals_prophet, type = "Ljung-Box")
Box.test(residuals_garch, type = "Ljung-Box")
Box.test(residuals(tbats_fit), type = "Ljung-Box")
Box.test(residuals_rf, type = "Ljung-Box")
Box.test(residuals(nn_fit), type = "Ljung-Box")
Box.test(residuals_stl, type = "Ljung-Box")
Box.test(residuals(stlf_fit), type = "Ljung-Box")
Box.test(residuals(additive_fit), type = "Ljung-Box")

#=========================================================

#5.2 Performance Metrics Evaluation

#Combine all metrics into one dataframe
combined_metrics <- rbind(sarima_metrics, hw_metrics, prophet_metrics, garch_metrics, tbats_metrics, rf_metrics, nn_metrics, stl_metrics, stlf_metrics)

#Print combined metrics
print(combined_metrics)

#=========================================================

#6.1 Forecasting using Best Fit Model

fitted_values <- fitted(nn_fit)
std_residuals <- sd(residuals_nn)

#Set prediction interval
alpha <- 0.05 
z <- qnorm(1 - alpha / 2)

#Calculate lower and upper bounds
lower_bound <- fitted_values - z * std_residuals
upper_bound <- fitted_values + z * std_residuals

#Create a data frame for the predictions and intervals
nn_predictions <- data.frame(
  Fitted = fitted_values,
  Lower_Bound = lower_bound,
  Upper_Bound = upper_bound
)

#Plot the fitted values with confidence intervals
plot(train$Adj.Close, type = "l", col = "blue", main = "Neural Network Predictions with Confidence Intervals")
lines(nn_predictions$Fitted, col = "red")
lines(nn_predictions$Lower_Bound, col = "green", lty = 2)
lines(nn_predictions$Upper_Bound, col = "green", lty = 2)


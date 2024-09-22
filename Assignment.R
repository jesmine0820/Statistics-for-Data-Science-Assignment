# Load necessary libraries
library(ggplot2)
library(GGally)
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

# Read the csv file
msft <- read.csv("MSFT.csv")

# 2.1 Data Overview
print("Number of observations: ")
nrow(msft)

print("Time span: ")
range(msft$Date)

print("Key variables: ")
names(msft)

# 2.2 Data Quality Checking
# Check missing values
print("Count of total missing values: ")
colSums(is.na(msft))
summary(msft)

# Addressing missing values
# msft <- na.omit(msft)
# This is used to overcome the missing values.
# However, there is no missing values in this dataset.
# Thus, we no need to address it.

# Check for outliers or anomalies
# Boxplot for Open prices
ggplot(msft, aes(y = Open)) +
  geom_boxplot() +
  labs(title = "Boxplot of Open Prices")

# Boxplot for Volume
ggplot(msft, aes(y = Volume)) +
  geom_boxplot() +
  labs(title = "Boxplot of Volume")

# Scatter plot for Open vs Close prices
ggplot(msft, aes(x = Open, y = Close)) +
  geom_point() +
  labs(title = "Scatter Plot of Open vs Close Prices")

# Scatter plot for High vs Low prices
ggplot(msft, aes(x = High, y = Low)) +
  geom_point() +
  labs(title = "Scatter Plot of High vs Low Prices")

# Line plot with annotations for Volume
ggplot(msft, aes(x = Date, y = Volume)) +
  geom_line() +
  geom_point(data = msft[msft$Volume > quantile(msft$Volume, 0.99), ], color = "red") +
  labs(title = "Volume Over Time with Anomalies Highlighted")

# Data Preprocessing
# Convert the Date column to Date format
msft$Date <- as.Date(msft$Date)

# Important date information 
dates <- msft$Date
years <- msft$Year
months <- msft$Month

# Columns
low <- msft$Low
high <- msft$High
volume <- msft$Volume
adj_close <- msft$Adj.Close
close <- msft$Close
open <- msft$Open

# Handling Outliers
handle_outliers <- function(data, column, threshold = 3) {
  z_scores <- abs((data[[column]] - mean(data[[column]], na.rm = TRUE)) / sd(data[[column]], na.rm = TRUE))
  data[[column]][z_scores > threshold] <- NA
  data[[column]] <- na.approx(data[[column]], rule = 2) 
  return(data)
}

msft <- handle_outliers(msft, "Open")
msft <- handle_outliers(msft, "Close")
msft <- handle_outliers(msft, "Volume")
msft <- handle_outliers(msft, "High")
msft <- handle_outliers(msft, "Low")
msft <- handle_outliers(msft, "Adj.Close")

# 2.3 Data Visualization
# 1. Time Series Plots
# Time Series for Open, Close, and Adj Close Prices
ggplot(msft, aes(x = Date)) +
  geom_line(aes(y = Open, color = "Open")) +
  geom_line(aes(y = Close, color = "Close")) +
  geom_line(aes(y = Adj.Close, color = "Adj Close")) +
  labs(title = "Time Series of Open, Close, and Adj Close Prices", y = "Price", color = "Legend") +
  scale_color_manual(values = c("Open" = "blue", "Close" = "green", "Adj Close" = "red"))

# Time Series for Max and Min Prices
msft <- msft %>%
  mutate(Max_Price = pmax(Open, High, Low, Close, Adj.Close),
         Min_Price = pmin(Open, High, Low, Close, Adj.Close))

ggplot(msft, aes(x = Date)) +
  geom_line(aes(y = Max_Price, color = "Max Price")) +
  geom_line(aes(y = Min_Price, color = "Min Price")) +
  labs(title = "Time Series of Max and Min Prices", y = "Price", color = "Legend") +
  scale_color_manual(values = c("Max Price" = "purple", "Min Price" = "orange"))

# 2. Histogram
# Histogram of Open, Close, and Adj Close Prices with Kernel Density Estimate
msft_long <- msft %>%
  pivot_longer(cols = c(Open, Close, Adj.Close),
               names_to = "Price_Type",
               values_to = "Price")

str(msft_long)

ggplot(msft_long, aes(x = Price, fill = Price_Type)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 1, alpha = 0.5, position = "identity") +
  geom_density(alpha = 0.5) +
  labs(title = "Histogram of Open, Close, and Adj Close Prices with Kernel Density Estimate", 
       x = "Price", 
       y = "Density") +
  facet_wrap(~Price_Type, scales = "free")


# 3. Box Plots
# Box Plots for Prices
msft_long <- msft %>%
  pivot_longer(cols = c(Open, Close, Adj.Close, High, Low),
               names_to = "Price_Type",
               values_to = "Price")

ggplot(msft_long, aes(x = Price_Type, y = Price, fill = Price_Type)) +
  geom_boxplot() +
  labs(title = "Box Plots for Prices", x = "Price Type", y = "Price")

# 4. Heatmap of Correlation
msft_numeric <- msft %>%
  pivot_longer(cols = c(Open, Close, Adj.Close, High, Low, Volume),
               names_to = "Variable",
               values_to = "Value")

msft_wide <- msft_numeric %>%
  pivot_wider(names_from = Variable, values_from = Value)

corr_matrix <- cor(msft_wide %>% select_if(is.numeric), use = "complete.obs")

corrplot(corr_matrix, method = "color", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")

# 5. PairPlot using GGally
ggpairs(msft, 
        columns = c("Low", "High", "Volume", "Adj.Close", "Close", "Open"), 
        upper = list(continuous = "points"), 
        lower = list(continuous = "points"), 
        diag = list(continuous = "densityDiag"))

# 3.0 Exploratory Data Analysis (EDA)
# Trend Analysis
# 1. Linear Regression
lm_open <- lm(Open ~ Date, data = msft)
msft$Open_Trend <- predict(lm_open)

lm_close <- lm(Close ~ Date, data = msft)
msft$Close_Trend <- predict(lm_close)

lm_adj_close <- lm(Adj.Close ~ Date, data = msft)
msft$Adj.Close_Trend <- predict(lm_adj_close)

# Plot Linear Regression Trends
ggplot(msft, aes(x = Date)) +
  geom_line(aes(y = Open, color = "Open")) +
  geom_line(aes(y = Open_Trend, color = "Open Trend")) +
  geom_line(aes(y = Close, color = "Close")) +
  geom_line(aes(y = Close_Trend, color = "Close Trend")) +
  geom_line(aes(y = Adj.Close, color = "Adj Close")) +
  geom_line(aes(y = Adj.Close_Trend, color = "Adj Close Trend")) +
  labs(title = "Linear Regression Trends for Open, Close, and Adj Close Prices", y = "Price", color = "Legend") +
  scale_color_manual(values = c("Open" = "blue", "Open Trend" = "lightblue", "Close" = "green", "Close Trend" = "lightgreen", "Adj Close" = "red", "Adj Close Trend" = "pink"))

# 2. Moving Averages
# Calculate rolling mean with NA values
msft$Open_MA <- rollmean(msft$Open, k = 30, fill = NA, align = "right")
msft$Close_MA <- rollmean(msft$Close, k = 30, fill = NA, align = "right")
msft$Adj.Close_MA <- rollmean(msft$Adj.Close, k = 30, fill = NA, align = "right")

# Forward-fill the NA values introduced by the rolling mean
if (!all(is.na(msft$Open_MA))) {
  msft$Open_MA <- na.locf(msft$Open_MA, na.rm = FALSE)
}

if (!all(is.na(msft$Close_MA))) {
  msft$Close_MA <- na.locf(msft$Close_MA, na.rm = FALSE)
}

if (!all(is.na(msft$Adj.Close_MA))) {
  msft$Adj.Close_MA <- na.locf(msft$Adj.Close_MA, na.rm = FALSE)
}

# Plot the Moving Averages
ggplot(msft, aes(x = Date)) +
  geom_line(aes(y = Open, color = "Open")) +
  geom_line(aes(y = Open_MA, color = "Open MA")) +
  geom_line(aes(y = Close, color = "Close")) +
  geom_line(aes(y = Close_MA, color = "Close MA")) +
  geom_line(aes(y = Adj.Close, color = "Adj Close")) +
  geom_line(aes(y = Adj.Close_MA, color = "Adj Close MA")) +
  labs(title = "Moving Averages for Open, Close, and Adj Close Prices", y = "Price", color = "Legend") +
  scale_color_manual(values = c("Open" = "blue", "Open MA" = "lightblue", "Close" = "green", "Close MA" = "lightgreen", "Adj Close" = "red", "Adj Close MA" = "pink"))

# 3.2 Seasonality Analysis
# We use Seasonal Decomposition of Time Series by Loess(STL)
stl_open <- stl(ts(msft$Open, frequency = 365), s.window = "periodic")
plot(stl_open)

stl_close <- stl(ts(msft$Close, frequency = 365), s.window = "periodic")
plot(stl_close)

stl_adj_close <- stl(ts(msft$Adj.Close, frequency = 365), s.window = "periodic")
plot(stl_adj_close)

# 3.3 Stationary Tests
# 3.3.1 ADF Test
adf.test(msft$Open)
adf.test(msft$Close)
adf.test(msft$Adj.Close)

# 3.3.2 KPSS Test
kpss.test(msft$Open)
kpss.test(msft$Close)
kpss.test(msft$Adj.Close)

# 3.4 Autocorrelation and Partial Autocorrelation
# ACF
acf(msft$Open, main = "ACF for Open Prices")
acf(msft$Close, main = "ACF for Close Prices")
acf(msft$Adj.Close, main = "ACF for Adj.Close Prices")

# PACF
pacf(msft$Open, main = "PACF for Open Prices")
pacf(msft$Close, main = "PACF for Close Prices")
pacf(msft$Adj.Close, main = "PACF for Adj.Close Prices")

# 3.5 Transformation
# We use Log Transformation and apply them to Open, Close and Adj.Close price
msft$Open_log <- log(msft$Open)
msft$Close_log <- log(msft$Close)
msft$Adj.Close_log <- log(msft$Adj.Close)

# Time Series Plot for Open, Close, and Adj.Close Prices
ggplot(msft, aes(x = Date)) +
  geom_line(aes(y = Open, color = "Open")) +
  geom_line(aes(y = Close, color = "Close")) +
  geom_line(aes(y = Adj.Close, color = "Adj.Close")) +
  labs(title = "Time Series of Open, Close, and Adj.Close Prices", y = "Price", color = "Legend") +
  scale_color_manual(values = c("Open" = "blue", "Close" = "green", "Adj.Close" = "red"))

# Time Series Plot for High and Low Prices
ggplot(msft, aes(x = Date)) +
  geom_line(aes(y = High, color = "High")) +
  geom_line(aes(y = Low, color = "Low")) +
  labs(title = "Time Series of High and Low Prices", y = "Price", color = "Legend") +
  scale_color_manual(values = c("High" = "purple", "Low" = "orange"))

# Time Series Plot for Volume
ggplot(msft, aes(x = Date, y = Volume)) +
  geom_line(color = "black") +
  labs(title = "Time Series of Volume", y = "Volume")

# Train-Test Split
train_data <- msft %>% filter(Date >= as.Date("1986-01-01") & Date <= as.Date("2001-12-31"))
test_data <- msft %>% filter(Date >= as.Date("2002-01-01") & Date <= as.Date("2014-12-31"))

# 4.1 Model Identification and Model Fitting
# 1. ARIMA Model
arima_open <- auto.arima(train_data$Open)
arima_forecast_open <- forecast(arima_open, h = nrow(test_data))$mean
rmse_arima_open <- rmse(test_data$Open, arima_forecast_open)
mae_arima_open <- mae(test_data$Open, arima_forecast_open)
mape_arima_open <- mape(test_data$Open, arima_forecast_open)

# 2. Seasonal ARIMA (SARIMA)
sarima_open <- auto.arima(train_data$Open, seasonal = TRUE)
sarima_forecast_open <- forecast(sarima_open, h = nrow(test_data))$mean
rmse_sarima_open <- rmse(test_data$Open, sarima_forecast_open)
mae_sarima_open <- mae(test_data$Open, sarima_forecast_open)
mape_sarima_open <- mape(test_data$Open, sarima_forecast_open)

# 3. Exponential Smoothing (Holt-Winters)
hw_open <- HoltWinters(ts(train_data$Open, frequency = 365))
hw_forecast_open <- forecast(hw_open, h = nrow(test_data))$mean
rmse_hw_open <- rmse(test_data$Open, hw_forecast_open)
mae_hw_open <- mae(test_data$Open, hw_forecast_open)
mape_hw_open <- mape(test_data$Open, hw_forecast_open)

# 4. Prophet
prophet_open <- prophet(data.frame(ds = train_data$Date, y = train_data$Open))
future_open <- make_future_dataframe(prophet_open, periods = nrow(test_data))
forecast_open <- predict(prophet_open, future_open)
prophet_forecast_open <- forecast_open$yhat[-(1:nrow(train_data))]
rmse_prophet_open <- rmse(test_data$Open, prophet_forecast_open)
mae_prophet_open <- mae(test_data$Open, prophet_forecast_open)
mape_prophet_open <- mape(test_data$Open, prophet_forecast_open)

# 5. GARCH Model
spec_open <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                        mean.model = list(armaOrder = c(1, 1), include.mean = TRUE))
garch_open <- ugarchfit(spec_open, train_data$Open)
garch_forecast_open <- ugarchforecast(garch_open, n.ahead = nrow(test_data))@forecast$seriesFor
rmse_garch_open <- rmse(test_data$Open, garch_forecast_open)
mae_garch_open <- mae(test_data$Open, garch_forecast_open)
mape_garch_open <- mape(test_data$Open, garch_forecast_open)

# 6. VAR Model
var_model <- VAR(train_data[, c("Open", "Close")], p = 2)
var_forecast <- predict(var_model, n.ahead = nrow(test_data))
var_forecast_open <- var_forecast$fcst$Open[, "fcst"]
rmse_var_open <- rmse(test_data$Open, var_forecast_open)
mae_var_open <- mae(test_data$Open, var_forecast_open)
mape_var_open <- mape(test_data$Open, var_forecast_open)

# 7. TBATS Model
tbats_model <- tbats(ts(train_data$Open, frequency = 365))
tbats_forecast <- forecast(tbats_model, h = nrow(test_data))
tbats_forecast_open <- tbats_forecast$mean
rmse_tbats_open <- rmse(test_data$Open, tbats_forecast_open)
mae_tbats_open <- mae(test_data$Open, tbats_forecast_open)
mape_tbats_open <- mape(test_data$Open, tbats_forecast_open)

# 8. Random Forest
rf_model <- randomForest(Open ~ ., data = train_data[, c("Open", "Close", "High", "Low", "Volume")])
rf_forecast <- predict(rf_model, newdata = test_data)
rmse_rf_open <- rmse(test_data$Open, rf_forecast)
mae_rf_open <- mae(test_data$Open, rf_forecast)
mape_rf_open <- mape(test_data$Open, rf_forecast)

# 9. Neural Networks (NNETAR)
nnetar_model <- nnetar(train_data$Open, frequency = 365)
nnetar_forecast <- forecast(nnetar_model, h = nrow(test_data))
nnetar_forecast_open <- nnetar_forecast$mean
rmse_nnetar_open <- rmse(test_data$Open, nnetar_forecast_open)
mae_nnetar_open <- mae(test_data$Open, nnetar_forecast_open)
mape_nnetar_open <- mape(test_data$Open, nnetar_forecast_open)

# 5.1 Residual Diagnostics
# 5.1.1 White Noise Test
Box.test(residuals(arima_open), type = "Ljung-Box")
Box.test(residuals(arima_close), type = "Ljung-Box")
Box.test(residuals(arima_adj_close), type = "Ljung-Box")

# Ljung-Box Test
checkresiduals(arima_open)
checkresiduals(arima_close)
checkresiduals(arima_adj_close)

# Residual Plotting
ggplot(data.frame(Residuals = residuals(arima_open)), aes(x = 1:length(Residuals), y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residual Plot for ARIMA Model (Open Prices)", x = "Time", y = "Residuals")

ggplot(data.frame(Residuals = residuals(arima_close)), aes(x = 1:length(Residuals), y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residual Plot for ARIMA Model (Close Prices)", x = "Time", y = "Residuals")

ggplot(data.frame(Residuals = residuals(arima_adj_close)), aes(x = 1:length(Residuals), y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residual Plot for ARIMA Model (Adj.Close Prices)", x = "Time", y = "Residuals")

# 5.2 Performance Metrics
# Ensure the fitted values cover the same time period as the actual values
fitted_open <- fitted(arima_open)
fitted_close <- fitted(arima_close)
fitted_adj_close <- fitted(arima_adj_close)

# Combine fitted values with the test data
combined_open <- c(fitted_open, rep(NA, nrow(msft) - length(fitted_open)))
combined_close <- c(fitted_close, rep(NA, nrow(msft) - length(fitted_close)))
combined_adj_close <- c(fitted_adj_close, rep(NA, nrow(msft) - length(fitted_adj_close)))

# Calculate performance metrics
rmse_open <- rmse(msft$Open, combined_open)
mae_open <- mae(msft$Open, combined_open)
aic_open <- AIC(arima_open)
bic_open <- BIC(arima_open)
mape_open <- mape(msft$Open, combined_open)

print(paste("RMSE (Open):", rmse_open))
print(paste("MAE (Open):", mae_open))
print(paste("AIC (Open):", aic_open))
print(paste("BIC (Open):", bic_open))
print(paste("MAPE (Open):", mape_open))

rmse_close <- rmse(msft$Close, combined_close)
mae_close <- mae(msft$Close, combined_close)
aic_close <- AIC(arima_close)
bic_close <- BIC(arima_close)
mape_close <- mape(msft$Close, combined_close)

print(paste("RMSE (Close):", rmse_close))
print(paste("MAE (Close):", mae_close))
print(paste("AIC (Close):", aic_close))
print(paste("BIC (Close):", bic_close))
print(paste("MAPE (Close):", mape_close))

rmse_adj_close <- rmse(msft$Adj.Close, combined_adj_close)
mae_adj_close <- mae(msft$Adj.Close, combined_adj_close)
aic_adj_close <- AIC(arima_adj_close)
bic_adj_close <- BIC(arima_adj_close)
mape_adj_close <- mape(msft$Adj.Close, combined_adj_close)

print(paste("RMSE (Adj.Close):", rmse_adj_close))
print(paste("MAE (Adj.Close):", mae_adj_close))
print(paste("AIC (Adj.Close):", aic_adj_close))
print(paste("BIC (Adj.Close):", bic_adj_close))
print(paste("MAPE (Adj.Close):", mape_adj_close))

# Forecasting Errors and Prediction Intervals
forecast_open <- forecast(arima_open, h = 30)
plot(forecast_open)

forecast_close <- forecast(arima_close, h = 30)
plot(forecast_close)

forecast_adj_close <- forecast(arima_adj_close, h = 30)
plot(forecast_adj_close)

# Cross-Validation for Time Series
# Define the time series split function
time_series_split <- function(data, train_start, train_end, test_start, test_end) {
  train <- data %>% filter(Date >= train_start & Date <= train_end)
  test <- data %>% filter(Date >= test_start & Date <= test_end)
  return(list(train = train, test = test))
}

# Define the forecast function (e.g., ARIMA)
forecast_function <- function(train) {
  auto.arima(train$Open)
}

# Define the time periods for the splits
train_start <- as.Date("1986-01-01")
train_end <- as.Date("2001-12-31")
test_start <- as.Date("2002-01-01")
test_end <- as.Date("2014-12-31")

# Perform time series split
split <- time_series_split(msft, train_start, train_end, test_start, test_end)
train_data <- split$train
test_data <- split$test

# Fit the model on the training data
model <- forecast_function(train_data)

# Forecast on the test data
forecast <- forecast(model, h = nrow(test_data))$mean

# Calculate performance metrics
rmse_value <- rmse(test_data$Open, forecast)
mae_value <- mae(test_data$Open, forecast)
mape_value <- mape(test_data$Open, forecast)

print(paste("RMSE:", rmse_value))
print(paste("MAE:", mae_value))
print(paste("MAPE:", mape_value))

# Plot the actual vs forecasted values
plot_data <- data.frame(Date = test_data$Date, Actual = test_data$Open, Forecast = forecast)
ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Forecast, color = "Forecast")) +
  labs(title = "Actual vs Forecasted Open Prices", x = "Date", y = "Price", color = "Legend") +
  scale_color_manual(values = c("Actual" = "blue", "Forecast" = "red"))

# Create a summary table for performance metrics
performance_metrics <- data.frame(
  Model = c("ARIMA", "SARIMA", "Holt-Winters", "Prophet", "GARCH", "VAR", "TBATS", "Random Forest", "NNETAR"),
  RMSE = c(rmse_arima_open, rmse_sarima_open, rmse_hw_open, rmse_prophet_open, rmse_garch_open, rmse_var_open, rmse_tbats_open, rmse_rf_open, rmse_nnetar_open),
  MAE = c(mae_arima_open, mae_sarima_open, mae_hw_open, mae_prophet_open, mae_garch_open, mae_var_open, mae_tbats_open, mae_rf_open, mae_nnetar_open),
  MAPE = c(mape_arima_open, mape_sarima_open, mape_hw_open, mape_prophet_open, mape_garch_open, mape_var_open, mape_tbats_open, mape_rf_open, mape_nnetar_open)
)

# Print the summary table
print(performance_metrics)

# Reshape the data for ggplot2
performance_metrics_long <- reshape2::melt(performance_metrics, id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Plot the performance metrics
ggplot(performance_metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Metric, scales = "free_y") +
  labs(title = "Model Performance Metrics", x = "Model", y = "Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 6.1
# Train the Random Forest model on the entire dataset
rf_model <- randomForest(Open ~ ., data = msft[, c("Open", "Close", "High", "Low", "Volume")])

# Forecast the next 30 days
forecast_horizon <- 365
future_dates <- seq(max(msft$Date) + 1, by = "day", length.out = forecast_horizon)

# Create future data with all necessary predictor variables
future_data <- data.frame(
  Date = future_dates,
  Close = rep(mean(msft$Close), forecast_horizon),
  High = rep(mean(msft$High), forecast_horizon),
  Low = rep(mean(msft$Low), forecast_horizon),
  Volume = rep(mean(msft$Volume), forecast_horizon)
)

# Predict using the Random Forest model
rf_forecast <- predict(rf_model, newdata = future_data)

# Plot the forecast
plot_data <- data.frame(Date = future_dates, Forecast = rf_forecast)
ggplot(plot_data, aes(x = Date, y = Forecast)) +
  geom_line(color = "blue") +
  labs(title = "Random Forest Forecast for the Next 30 Days", x = "Date", y = "Open Price")

# Calculate confidence intervals for Random Forest forecasts
# Since Random Forest does not inherently provide confidence intervals, we can use bootstrapping

# Function to calculate confidence intervals using bootstrapping
bootstrap_confidence_intervals <- function(model, newdata, n_samples = 1000, alpha = 0.05) {
  predictions <- replicate(n_samples, {
    boot_data <- newdata[sample(nrow(newdata), replace = TRUE), ]
    predict(model, newdata = boot_data)
  })
  lower_ci <- apply(predictions, 1, quantile, alpha / 2)
  upper_ci <- apply(predictions, 1, quantile, 1 - alpha / 2)
  return(list(lower_ci = lower_ci, upper_ci = upper_ci))
}

# Calculate confidence intervals
ci <- bootstrap_confidence_intervals(rf_model, future_data)

# Add confidence intervals to the forecast plot
plot_data$Lower_CI <- ci$lower_ci
plot_data$Upper_CI <- ci$upper_ci

ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Forecast, color = "Forecast")) +
  geom_ribbon(aes(ymin = Lower_CI, ymax = Upper_CI), alpha = 0.2) +
  labs(title = "Random Forest Forecast with Confidence Intervals", x = "Date", y = "Open Price", color = "Legend") +
  scale_color_manual(values = c("Forecast" = "blue"))

# Function to update the Random Forest model dynamically
update_forecast <- function(model, new_data) {
  updated_model <- randomForest(Open ~ ., data = new_data[, c("Open", "Close", "High", "Low", "Volume")])
  return(updated_model)
}

# Simulate updating the model with new data
# Assuming new_data is the new data available after the initial forecast
new_data <- msft[1:365, ]  # Example: Use the first 30 rows as new data
updated_model <- update_forecast(rf_model, new_data)

# Forecast using the updated model
updated_forecast <- predict(updated_model, newdata = future_data)

# Plot the updated forecast
updated_plot_data <- data.frame(Date = future_dates, Forecast = updated_forecast)
ggplot(updated_plot_data, aes(x = Date, y = Forecast)) +
  geom_line(color = "blue") +
  labs(title = "Updated Random Forest Forecast", x = "Date", y = "Open Price")

# Prepare the data for VAR model
var_data <- msft[, c("Open", "Close", "High", "Low", "Volume")]

# Fit the VAR model
var_model <- VAR(var_data, p = 2)

# Forecast the next 30 days using the VAR model
var_forecast <- predict(var_model, n.ahead = 365)

# Extract the forecasts for the "Open" variable
var_forecast_open <- var_forecast$fcst$Open[, "fcst"]

# Plot the VAR forecast
var_plot_data <- data.frame(Date = future_dates, Forecast = var_forecast_open)
ggplot(var_plot_data, aes(x = Date, y = Forecast)) +
  geom_line(color = "blue") +
  labs(title = "VAR Forecast for the Next 365 Days", x = "Date", y = "Open Price")

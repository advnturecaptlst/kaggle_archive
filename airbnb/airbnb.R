# required libraries
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

# set random seed
set.seed(13)

# load training and test data
df_train = read_csv("~/Documents/kaggle/airbnb/input/train_users.csv")
df_test = read_csv("~/Documents/kaggle/airbnb/input/test_users.csv")
labels = df_train['country_destination']
df_train = df_train[-grep('country_destination', colnames(df_train))]

# ndcg5 prediction function definition 
ndcg5 <- function(preds, dtrain) {
  
  labels <- getinfo(dtrain,"label")
  num.class = 12
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x <- ifelse(top==labels,1,0)
  dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg <- mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
}


# combine training and test datasets
df_all = rbind(df_train,df_test)

# drop date_first_booking column
df_all = df_all[-c(which(colnames(df_all) %in% c('date_first_booking')))]

# set NaN/null values to -1
df_all[is.na(df_all)] <- -1

# break out date_account_created by year, month and day
dac = as.data.frame(str_split_fixed(df_all$date_account_created, '-', 3))
df_all['dac_year'] = dac[,1]
df_all['dac_month'] = dac[,2]
df_all['dac_day'] = dac[,3]
df_all = df_all[,-c(which(colnames(df_all) %in% c('date_account_created')))]

# split timestamp_first_active in year, month and day
df_all[,'tfa_year'] = substring(as.character(df_all[,'timestamp_first_active']), 1, 4)
df_all['tfa_month'] = substring(as.character(df_all['timestamp_first_active']), 5, 6)
df_all['tfa_day'] = substring(as.character(df_all['timestamp_first_active']), 7, 8)
df_all = df_all[,-c(which(colnames(df_all) %in% c('timestamp_first_active')))]

# process age feature
df_all[df_all$age < 14 | df_all$age > 100,'age'] <- -1

# process ohe features
ohe_feats = c('gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser')
dummies <- dummyVars(~ gender + signup_method + signup_flow + language + affiliate_channel + affiliate_provider + first_affiliate_tracked + signup_app + first_device_type + first_browser, data = df_all)
df_all_ohe <- as.data.frame(predict(dummies, newdata = df_all))
df_all_combined <- cbind(df_all[,-c(which(colnames(df_all) %in% ohe_feats))],df_all_ohe)

# split training and test data
X = df_all_combined[df_all_combined$id %in% df_train$id,]
y <- recode(labels$country_destination,"'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11")
X_test = df_all_combined[df_all_combined$id %in% df_test$id,]

# train xgboost
xgb <- xgboost(data = data.matrix(X[,-1]), 
               label = y, 
               eta = 0.075,
               max_depth = 9, 
               nround=300, 
               subsample = 0.75,
               colsample_bytree = 0.85,
               eval_metric = ndcg5,
               objective = "multi:softprob",
               num_class = 12,
               nthread = 15
)

# predict values in test set
y_pred <- predict(xgb, data.matrix(X_test[,-1]))

# extract the 5 highest probability classes
predictions <- as.data.frame(matrix(y_pred, nrow=12))
rownames(predictions) <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')
top_predictions <- as.vector(apply(predictions, 2, function(x) names(sort(x)[12:8])))

# create output data
ids <- NULL
for (i in 1:NROW(X_test)) {
  idx <- X_test$id[i]
  ids <- append(ids, rep(idx,5))
}

submission <- NULL
submission$id <- ids

# 5 most probable predictions selected for output
submission$country <- top_predictions

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "rsub.csv", quote=FALSE, row.names = FALSE)
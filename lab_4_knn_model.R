library(tidyverse)
library(tidymodels)


all_cores <- parallel::detectCores(logical = FALSE)

#install.packages("doParallel")

library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})


full_train <- read_csv("data/train.csv")

data_sample <- dplyr::sample_frac(full_train, size = 0.005)

# create initial splits 
set.seed(3000)

data_split <- initial_split(data_sample)

train_split <- training(data_split)
test_split <- testing(data_split)

class(data_split)
class(train_split)
class(test_split)


# K-fold cross validation 

set.seed(3000)

cv_splits <- vfold_cv(train_split, v = 10)




library(tidyverse)
library(tidymodels)
library(kknn)
library(doParallel)

full_train <- read_csv("data/train.csv") %>% 
  mutate(classification = factor(classification,
                                 levels = 1:4,
                                 labels = c("far below", "below", "meets", "exceeds"),
                                 ordered = TRUE))

data_sample <- dplyr::sample_frac(full_train, size = 0.005)

# create initial splits 
set.seed(3000)

data_split <- initial_split(data_sample)

train_split <- training(data_split)
test_split <- testing(data_split)

class(data_split)
class(train_split)
class(test_split)

# use the training data to create a k-fold cross-validation data object
set.seed(3000)
cv_splits <- vfold_cv(train_split)

# basic recipe
rec <- 
  recipe(classification ~ enrl_grd + lat + lon + gndr, data = train_split) %>%
  step_mutate(enrl_grd = factor(enrl_grd), gndr = factor(gndr)) %>%
  step_meanimpute(lat, lon) %>%
  step_unknown(enrl_grd, gndr) %>% 
  step_dummy(enrl_grd, gndr) %>% 
  step_normalize(lat, lon)

# parallel processing code from the assignment

all_cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})




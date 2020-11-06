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

#data_sample <- dplyr::sample_frac(full_train, size = 0.005)

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


# basic recipe
knn01_rec <- 
  recipe(classification ~ enrl_grd + lat + lon + gndr, data = train_split) %>%
  step_mutate(classification = ifelse (classification < 3, "below", "proficient")) %>%
  step_mutate(enrl_grd = factor(enrl_grd), gndr = factor(gndr)) %>%
  step_meanimpute(lat, lon) %>%
  step_unknown(enrl_grd, gndr) %>% 
  step_dummy(enrl_grd, gndr) %>% 
  step_normalize(lat, lon)

# set model
knn01_mod <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

translate(knn01_mod)

#install.packages("kknn")
library(kknn)

fit1 <- fit_resamples(knn01_mod, knn01_rec, cv_splits)
saveRDS(fit1, "fit1.Rds")





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

#Initial split
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

# set model
knn2_mod <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification") %>% 
  set_args(neighbors = tune(),
           weight_func = tune(),
           dist_power = tune())

# make a non-regular grid

knn_params <- parameters(neighbors(range = c(1, 20)),
                         weight_func(),
                         dist_power())

knn_sfd <- grid_max_entropy(knn_params, size = 25)

knn_sfd %>%
  ggplot(aes(neighbors, dist_power)) +
  geom_point(aes(color = weight_func))



fit2 <- tune::tune_grid(
  knn2_mod, 
  rec, 
  cv_splits,
  grid = knn_sfd,
  control = tune::control_resamples(save_pred = TRUE))


fit2 %>% 
  show_best(metric = "roc_auc", n = 5)

best <- fit2 %>% 
  select_best(metric = "roc_auc")

mod_final <- knn2_mod %>% 
  finalize_model(best)

rec_final <- rec %>% 
  finalize_recipe(best)

cl <- makeCluster(8)
registerDoParallel(cl)

final_res <- last_fit(mod_final, 
                      preprocessor = rec_final, 
                      split = cv_splits)
stopCluster(cl)

final_res %>% 
  collect_metrics()
library(tidyverse)
library(tidymodels)

full_train <- read_csv(here::here("data", "train.csv")) %>% 
  select(-classification)


data_sample <- dplyr::sample_frac(full_train, size = 0.01)

set.seed(3000)
data_split <- initial_split(full_train)
train <- training(data_split)
cv <- vfold_cv(train)
train

rec <- recipe(score ~ ., train) %>% 
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt)),
              score = factor(score)) %>% 
  update_role(contains("id"), ncessch, new_role = "id vars") %>% 
  step_novel(all_nominal()) %>%
  step_unknown(all_nominal()) %>%
  step_nzv(all_predictors()) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id vars")) %>%
  step_dummy(all_nominal(), -has_role("id vars")) %>%
  step_nzv(all_predictors())


# specify model 

library(baguette)

mod <- bag_tree() %>% 
  set_mode("classification") %>% 
  set_args(cost_complexity = 0, min_n = 2) %>% 
  set_engine("rpart", times = 50) 


m1_start <- Sys.time()
m1 <- fit_resamples(mod, rec, cv)
m1_end <- Sys.time()
m1_end - m1_start

show_best(m1, "roc_auc")

show_best(m1, "accuracy")


# slide 16 - function to pull roc_auc from the model 

small_cv <- vfold_cv(train, v = 2)

pull_auc <- function(b) {
  # specify model
  mod <- bag_tree() %>% 
    set_mode("classification") %>% 
    set_args(cost_complexity = 0, min_n = 2) %>% 
    set_engine("rpart", times = b)
  # fit model to full training dataset
  m <- fit_resamples(mod, rec, small_cv)
  show_best(m, "roc_auc")
}

# slide 18 - evaluate b

library(future)
plan(multisession)


library(tictoc)
tic()
bags <- map_df(seq(1, 200, 15), pull_auc) 
toc()


# slide 19 - learning curve 

bags %>% 
  mutate(b = seq(5, 200, 15)) %>% 
  ggplot(aes(b, mean)) +
  geom_line() +
  geom_point()

# Slide 20 - model tuning 

mod_tune <- bag_tree() %>% 
  set_mode("classification") %>% 
  set_args(cost_complexity = tune(), min_n = tune()) %>% 
  set_engine("rpart", times = 50) 

tree_grid <- grid_max_entropy(cost_complexity(), min_n(), size = 20)

plan(multisession)
tic()
bag_tune <- tune_grid(mod_tune, rec, cv, grid = tree_grid)
toc()


# slide 21 - best hyper parameters 

select_best(bag_tune, "roc_auc")

# finalize model 

final_mod <- mod_tune %>% 
  finalize_model(select_best(bag_tune, "roc_auc"))

# test fit 

final_fit <- last_fit(final_mod, rec, splt)
collect_metrics(final_fit)




# Random forest for bagging 

splt_reg <- initial_split(data_sample)
train_reg <- training(splt_reg)
cv_reg <- vfold_cv(train_reg)


rec_reg <- recipe(score ~ ., train) %>% 
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt)),
              score = factor(score)) %>% 
  update_role(contains("id"), ncessch, new_role = "id vars") %>% 
  step_novel(all_nominal()) %>%
  step_unknown(all_nominal()) %>%
  step_nzv(all_predictors()) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id vars")) %>%
  step_dummy(all_nominal(), -has_role("id vars")) %>%
  step_nzv(all_predictors())

#  single tree 

tune_single_tree <- decision_tree() %>% 
  set_mode("regression") %>% 
  set_engine("rpart") %>% 
  set_args(cost_complexity = tune(),
           min_n = tune())

params <- parameters(cost_complexity(), min_n())

grd <- grid_max_entropy(params, size = 50)

cl <- parallel::makeCluster(parallel::detectCores())
doParallel::registerDoParallel(cl)
single_tree_grid <- tune_grid(
  tune_single_tree,
  rec_reg,
  cv_reg,
  grid = grd
)
parallel::stopCluster(cl)
foreach::registerDoSEQ() 

show_best(single_tree_grid, "rmse")


# slide 42 

single_tree <- tune_single_tree %>% 
  finalize_model(select_best(single_tree_grid, "rmse"))

single_tree_fit <- last_fit(single_tree, rec_reg, splt_reg)
single_tree_fit$.metrics

# Bagging 

prepped_reg <- rec_reg %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  select(-contains("id"), -ncessch, -tst_dt)

ncol(prepped_reg) - 1

# Slide 44 - function 

pull_rmse <- function(b) {
  # specify model
  mod <- rand_forest() %>% 
    set_mode("regression") %>% 
    set_engine("ranger") %>% 
    set_args(mtry = 28,
             min_n = 2,
             trees = b)

  m <- fit(mod, score ~ ., prepped_reg)

  tibble(rmse = sqrt(m$fit$prediction.error))
}

# Estimate 

tic()
bags_reg <- map_df(seq(1, 500, 25), pull_rmse) 
toc()

# slide 46 - plot 

bags_reg %>% 
  mutate(b = seq(1, 500, 25)) %>% 
  ggplot(aes(b, rmse)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = 155, color = "magenta", lwd = 1.3)

# slide 47 tune min_n

tune_min_n <- function(n) {
  mod <- rand_forest() %>% 
    set_mode("regression") %>% 
    set_engine("ranger") %>% 
    set_args(mtry = 28,
             min_n = n,
             trees = 200)

  m <- fit(mod, score ~ ., prepped_reg)

  tibble(rmse = sqrt(m$fit$prediction.error))
}


tic()
optimal_n <- map_df(seq(2, 170, 2), tune_min_n) 
toc()

# slide 49 check learning curve 

optimal_n %>% 
  mutate(n = seq(2, 170, 2)) %>% 
  ggplot(aes(n, rmse)) +
  geom_line() +
  geom_point()

# slide 50 - finalized bagged model

mod <- rand_forest() %>% 
  set_mode("regression") %>% 
  set_engine("ranger") %>% 
  set_args(mtry = 28,
           min_n = 54,
           trees = 200)
final_fit <- last_fit(mod, rec_reg, splt_reg)
final_fit$.metrics[[1]]






library(tidyverse)
library(tidymodels)
library(kknn)
library(doParallel)

source("lab4_fit2.R")

fit2_result <- readRDS("lab4_fit2.Rds")

fit2_result %>% 
  show_best(metric = "roc_auc", n = 5)

best <- fit2_result %>% 
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

saveRDS(final_res, "final_fit.Rds")
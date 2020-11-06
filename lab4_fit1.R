source("lab4_common.R")

# set model
knn1_mod <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

fit1 <- fit_resamples(knn1_mod, rec, cv_splits)

saveRDS(fit1, "lab4_fit1.Rds")

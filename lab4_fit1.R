source("lab4_common.R")

# set model
knn01_mod <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

fit1 <- fit_resamples(knn01_mod, rec, cv_splits)

saveRDS(fit1, "lab4_fit1.Rds")

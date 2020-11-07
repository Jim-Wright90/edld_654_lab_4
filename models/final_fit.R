source("lab4_common.R")


knn2_mod <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification") %>% 
  set_args(neighbors = tune(),
           weight_func = tune(),
           dist_power = tune())

knn_params <- parameters(neighbors(range = c(1, 20)),
                         weight_func(),
                         dist_power())

knn_sfd <- grid_max_entropy(knn_params, size = 25)

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

saveRDS(final_res, "final_fit.Rds")
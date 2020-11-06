source("lab4_common.R")

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
  ggplot(aes(neighbors, dist_power))+
  geom_point(aes(color = weight_func))



fit2 <- tune::tune_grid(
  knn2_mod, 
  rec, 
  cv_splits,
  grid = knn_sfd,
  control = tune::control_resamples(save_pred = TRUE))

saveRDS(fit2, "lab4_fit2.Rds")

# Define accuracies
benchmark_accuracy <- 0.15  # 15%

# Accuracies of ANN and RF models
ann_accuracy <- 0.148       # 14.8%
rf_accuracy <- 0.102        # 10.2%

# Number of test samples
n_test <- 60

# Prop.test for ANN
ann_test <- prop.test(x = ann_accuracy * n_test, n = n_test, p = benchmark_accuracy, alternative = "two.sided", conf.level = 0.95)

# Prop.test for Random Forest
rf_test <- prop.test(x = rf_accuracy * n_test, n = n_test, p = benchmark_accuracy, alternative = "two.sided", conf.level = 0.95)

# Printing the results
print(ann_test)
print(rf_test)

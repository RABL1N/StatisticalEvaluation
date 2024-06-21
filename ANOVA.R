setwd("working_directory")  # Change this to the directory where your CSV is saved

# Import the data
accuracy_data <- read.csv("model_accuracies.csv")

# Check the first few rows of the data
head(accuracy_data)

# Convert accuracies to a long format for ANOVA
library(tidyr)
accuracy_long <- pivot_longer(accuracy_data, cols = c(ANN_accuracies, RF_accuracies), names_to = "Model", values_to = "Accuracy")

# Check the structure of the data
str(accuracy_long)

# Perform one-way ANOVA
anova_result <- aov(Accuracy ~ Model, data = accuracy_long)

# Summarize the ANOVA results
summary(anova_result)


setwd('C:/c/Rawls/online_courses/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing')
dataset = read.csv('data.csv')

#taking care of missing data
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = T)), 
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = T)), 
                     dataset$Salary)

#Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No','Yes'),
                         labels = c(0, 1))

#splittng the dataset into Training & Test set
#install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == T)
test_set = subset(dataset, split == F)

# Feature Scaling 
training_set[, 2:3] = scale(training_set[, 2:3]) # remove factor columns
test_set[, 2:3] = scale(test_set[, 2:3])






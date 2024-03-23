library(ggplot2)

data("diamonds")
dim(diamonds)
qplot(diamonds$price, bins = 15)

diamond_samp <- sample(diamonds$price, 
                       1000)

qplot(diamond_samp, bins = 15)

bootsamp <- sample(diamond_samp, 
                   1000, 
                   replace = TRUE)

qplot(bootsamp, bins = 15)

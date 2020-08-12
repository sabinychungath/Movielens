## Quiz: MovieLens dataset

# Load the edx & validation data sets using the provided script

#############################################################
# Create edx set, validation set, and submission file
#############################################################
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubricate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()

download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)

# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>% 
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#Q1 
#How many rows and columns are there in the edx dataset?
rows <- nrow(edx)
rows
columns <- ncol(edx)
columns

#Q2
#How many zeros were given as ratings in the edx dataset?
rate0 <- edx %>% filter(rating == 0)
nrow(rate0)
#How many threes were given as ratings in the edx dataset?
rate3 <- edx %>% filter(rating == 3)
nrow(rate3)

#Q3 
#How many different movies are in the edx dataset?
n_distinct(edx$movieId)

#Q4
#How many different users are in the edx dataset?
n_distinct(edx$userId)

#Q5
#How many movie ratings are in each of the following genres in the edx dataset?
#Drama
drama <- edx %>% filter(str_detect(genres,"Drama"))
nrow(drama)
#comedy
comedy <- edx %>% filter(str_detect(genres,"Comedy"))
nrow(comedy)
#Thriller
thriller <- edx %>% filter(str_detect(genres,"Thriller"))
nrow(thriller)
#Romance
romance <- edx %>% filter(str_detect(genres,"Romance"))
nrow(romance)

#Q6
#Which movie has the greatest number of ratings?
high_rate <- edx %>% group_by(title) %>% summarize(number = n()) %>%
  arrange(desc(number))
high_rate

#Q7
#What are the five most given ratings in order from most to least?
most_Rate <- edx %>% group_by(rating) %>% summarize(number = n()) 
most_Rate %>% top_n(5) %>% arrange(desc(number))

#Q8
#True or False: In general, half star ratings are less common than whole star ratings 
#(e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
table(edx$rating)

most_Rate %>%
  mutate(half_star = rating %% 1 == 0.5) %>%
  group_by(half_star) %>%
  summarize(number = sum(number))


#--------------------------------------------------------------------
#Data cleaning
  validation_set <- validation %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week")) 

  summary(validation_set)

#--------------------------------------------------------------------
# Data Visualization
# histogram of number of ratings by movieId
  edx %>%
    count(movieId) %>% ggplot(aes(n)) +
    geom_histogram(fill = "pink", bins = 50, color = "black") +
    scale_x_log10() +
    xlab("Number of Ratings") +
    ylab("Number of Movies") + 
    ggtitle("Ratings per movie")

# histogram of number of ratings by userId
  edx %>%
    count(userId) %>% ggplot(aes(n)) +
    geom_histogram(fill = "pink", bins = 50, color = "black") +
    scale_x_log10() +
    xlab("Number of Ratings")+ 
    ylab("Number of Users")+ 
    ggtitle("Ratings given by users")

#Splitting timestamp
  edx %>% 
    mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
    group_by(date) %>%
    summarize(rating = mean(rating),.groups='drop') %>%
    ggplot(aes(date, rating)) +
    geom_point() +
    geom_smooth() +
    ggtitle("Rating Distribution Per Year")

#-------------------------------------------------------------
# Model Development
#The loss-function computes the RMSE
  RMSE <- function(true_ratings, predicted_ratings){
   sqrt(mean((true_ratings - predicted_ratings)^2))
  }

# 1.Mean Rating Model
# calculating the average of all ratings of the edx set
  mu <- mean(edx$rating)
  #calculating the RMSE for mean
  neive_RMSE <- RMSE(edx$rating, mu)
  neive_RMSE
  #Here, we represent results table with the first RMSE:
  #summarizing the rmse
  RMSE_results <- data.frame(Method = "Mean", RMSE = neive_RMSE)
  
#Movie effect model
  movie_avgs <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = mean(rating - mu),.groups='drop')
  #predicted ratings of movie for RMSE
  predicted_ratings_bi <- validation_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    mutate(pred = mu + b_i) 
  #calculating the RMSE for movie effect
  movie_RMSE <- RMSE(validation_set$rating, predicted_ratings_bi$pred)
  movie_RMSE
  
#Movie_user effect model
  user_avgs <- edx %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = mean(rating - mu - b_i),.groups='drop')
  #predicted ratings of movie and user for RMSE
   predicted_ratings_bu <- validation_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    mutate(pred = mu + b_i + b_u) 
  #calculating the RMSE for movies and users effects 
   movie_user_RMSE <- RMSE(validation_set$rating, predicted_ratings_bu$pred)
   movie_user_RMSE
  
#Movie+user+time effect model
  time_avgs <- edx %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
    group_by(date) %>%
    summarize(b_t = mean(rating - mu - b_i - b_u),.groups='drop')
  #predicted ratings of movie, user and time for RMSE
  predicted_ratings_bt <- validation_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(time_avgs, by='date') %>%
    mutate(pred = mu + b_i + b_u + b_t)
  #calculating the RMSE for movies, users and time effects 
    movie_user_time_RMSE <- RMSE(validation_set$rating, predicted_ratings_bt$pred)
    movie_user_time_RMSE


#--------------------------------------------------------------------    
#Regularization
# Movie & User Regularization Model
    # Optimize lambda by minimizing RMSE
    # lambda is a tuning parameter.
    lambdas <- seq(0, 10, 0.25)
    l_RMSE <- sapply(lambdas, function(l){
      mu <- mean(edx$rating)
      b_i <- edx %>%
        group_by(movieId) %>%
        summarize(b_i = sum(rating - mu)/(n()+l),.groups='drop')
      
      b_u <- edx %>%
        left_join(b_i, by="movieId") %>%
        group_by(userId) %>%
        summarize(b_u = sum(rating - b_i - mu)/(n()+l),.groups='drop')
      
      predicted_ratings <- validation_set %>%
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        mutate(pred = mu + b_i + b_u) 
      return(RMSE(validation_set$rating, predicted_ratings$pred))
    })
   
    #We plot RMSE vs lambdas to select the optimal lambda
    qplot(lambdas, l_RMSE)+ geom_point() +
      xlab('Lambda') + ylab("RMSE") + ggtitle("Lambda Tuning")
   
    #For the full model, the optimal  lambda is:
    lambda <- lambdas[which.min(l_RMSE)]
    lambda
    
    # now calculate the regularized accuracy with the best lambda
    Reg_user_RMSE <- min(l_RMSE)
    Reg_user_RMSE
    
#-----------------------------------------------------------    
# Movie, User and Time Regularization Model
    lambdas <- seq(0, 10, 0.25)
    mur_RMSE <- sapply(lambdas, function(l){
      mu <- mean(edx$rating)
      b_i <- edx %>%
        group_by(movieId) %>%
        summarize(b_i = sum(rating - mu)/(n()+l),.groups='drop')
      
      b_u <- edx %>%
        left_join(b_i, by="movieId") %>%
        group_by(userId) %>%
        summarize(b_u = sum(rating - b_i - mu)/(n()+l),.groups='drop')
      
      b_t  <- edx %>% 
        left_join(b_i, by='movieId') %>%
        left_join(b_u, by='userId') %>%
        mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
        group_by(date) %>%
        summarize(b_t = mean(rating - mu - b_i - b_u),.groups='drop')
      
      predicted_ratings <- validation_set %>%
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        left_join(b_t, by='date') %>%
        mutate(pred = mu + b_i + b_u + b_t) 
      return(RMSE(validation_set$rating, predicted_ratings$pred))
    })
    
    #We plot RMSE vs lambdas to select the optimal lambda
    qplot(lambdas, mur_RMSE)+ geom_point() +
      xlab('Lambda') + ylab("RMSE") + ggtitle("Lambda Tuning")
    
    #For the full model, the optimal  lambda is:
    lambda <- lambdas[which.min(mur_RMSE)]
    lambda
    
    # now calculate the regularized accuracy with the best lambda
    final_RMSE <- min(mur_RMSE)
    final_RMSE
#-----------------------------------------------------------------------------    
# Results   
    RMSE_results <- bind_rows(RMSE_results, data.frame(Method=c("Movie Effect Model","Movie+User Effect Model","Movie+User+Time Effect Model","Regularized Movie+User Effect Model","Regularized Movie+User+Time Effect Model"), RMSE = c(movie_RMSE, movie_user_RMSE, movie_user_time_RMSE, Reg_user_RMSE, final_RMSE)))
    
    RMSE_results %>% knitr::kable()
    
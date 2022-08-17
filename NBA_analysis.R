## loading the required packages
library(dplyr)
library(tidyr)
library(cluster)
library(factoextra)
library(ggplot2)
library(randomForest)
library(arm)
library(stringr)
library(gridExtra)
library(formattable)
library(readr)
library(datasets)
library(corrplot)
library(stats)
library(ggrepel)
library(textir) 
library(BBmisc)
library(rstudioapi)

#setwd("C:/Users/Alex/Videos/Captures/big data/Assigment3/Proyecto Data NBA")

#1.ANALYZING AND CLUSTERING OF NBA TOP PLAYER ----------------------------------

# load data 

traditionalStats<- read.csv("traditional_NBA_2020.csv", sep=";")
advancedStats<- read.csv("advanced_NBA_2020.csv", sep=";")

# Data pre-processing 

# Merge 1st 2 datasets
mergedData<-merge(traditionalStats,advancedStats);                      


# Remove redundant columns
mergedData<- mergedData[-c(1,42,47)] 

# Eliminate duplicates (keep only the total) (this happens when a player change from one team to another in the middle of the season)
mergedData<-mergedData[!(duplicated(mergedData$Player)), ]

# Feature extraction
mergedData$MPG<-mergedData$MP/mergedData$G; #minutes per game
mergedData$A2TO<-mergedData$AST/mergedData$TOV #assist to Turnover ratio

#check NA
sapply(mergedData, function(x) sum(is.na(x)))

#change NA to 0 in 3PT (some players dont attemp any 3 pointer)
mergedData$X3P.[is.na(mergedData$X3P.)] <- 0


#Rest Nas are irrelevant players, remove rows
mergedData<-na.omit(mergedData)


# Select threshold for subsetting
cutoff<-28.5 

# Subset data using Minutes per game
mergedData <- subset(mergedData, MPG >=cutoff );

# Subset Dataframe using relevant features
mainData<- mergedData[c(8:50)]; #relevant overall features
OffensiveData<- mergedData[c(8:20,23,26,28,29,30,31,32,33,36,39,40,41,45,47,48)]
DefensiveData<- mergedData[c(20,21,24,25,27,29,33,34,37,38,42,46,47,48)]

# Normalize data
mainData <- normalize(mainData, method= "standardize") 
OffensiveData <- normalize(OffensiveData, method= "standardize") 
DefensiveData <- normalize(DefensiveData, method= "standardize") 

# Feature Correlation plot
o=corrplot(cor(mainData),method='number')
of=corrplot(cor(OffensiveData),method='number')
de=corrplot(cor(DefensiveData),method='number')
# PCA 

# Finding covariance matrix from data
res.cov1 <- cov(mainData);
res.cov2 <- cov(OffensiveData);
res.cov3 <- cov(DefensiveData);

# Rounding it to decimal places
round(res.cov1,2);
round(res.cov2,2);
round(res.cov3,2);

# Find eigenvectors
eig1<-eigen(res.cov1)$vectors;
eig2<-eigen(res.cov2)$vectors;
eig3<-eigen(res.cov3)$vectors;

# Select 1st 2 eigenvectors
eigenv2_1<-eig1[,(1:2)];
eigenv2_2<-eig2[,(1:2)];
eigenv2_3<-eig3[,(1:2)];

# Convert to matrix for PCA calculations
statsMatrix1<-data.matrix(mainData, rownames.force = NA);
statsMatrix2<-data.matrix(OffensiveData, rownames.force = NA);
statsMatrix3<-data.matrix(DefensiveData, rownames.force = NA);

# Matrix multiplication
PrincipalComponents2_1<-statsMatrix1%*%eigenv2_1;
PrincipalComponents2_2<-statsMatrix2%*%eigenv2_2;
PrincipalComponents2_3<-statsMatrix3%*%eigenv2_3;

# Convert back to statsDF for ggplot
statsDF1<-data.frame(PrincipalComponents2_1); 
statsDF2<-data.frame(PrincipalComponents2_2);
statsDF3<-data.frame(PrincipalComponents2_3);

# PCA Plot general
ggplot(statsDF1,aes(-statsDF1$X1, -statsDF1$X2)) +
  labs(x="PC1",y="PC2")+
  geom_point(data=mergedData,aes(col =Pos, size= VORP))+
  geom_text_repel(data=mergedData, aes(label=Player), size=3+mergedData$VORP/max(mergedData$VORP))

# PCA Plot offensive
ggplot(statsDF2,aes(-statsDF2$X1, -statsDF2$X2)) +
  labs(x="PC1",y="PC2")+
  geom_point(data=mergedData,aes(col =Pos, size= VORP))+
  geom_text_repel(data=mergedData, aes(label=Player), size=3+mergedData$VORP/max(mergedData$VORP))

# PCA Plot defensive
ggplot(statsDF3,aes(-statsDF3$X1, statsDF3$X2)) +
  labs(x="PC1",y="PC2")+
  geom_point(data=mergedData,aes(col =Pos, size= VORP))+
  geom_text_repel(data=mergedData, aes(label=Player), size=3+mergedData$VORP/max(mergedData$VORP))

# k-means clustering 

# Replace index with player name (used for plot)
row.names(mainData) <- mergedData$Player 

# Clarify distance measures
res.dist <- get_dist(mainData, stand = TRUE, method = "euclidean")
fviz_dist(res.dist, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# Determinie optimal clusters through different methods
fviz_nbclust(mainData, kmeans, method = "wss")
fviz_nbclust(mainData, kmeans, method = "silhouette")
fviz_nbclust(mainData, kmeans, method = "gap_stat")

# Kmeans based on best cluster-number
km.res <- kmeans(mainData,6, nstart = 25)

# Visualize Kmeans clusters
fviz_cluster(km.res, mainData, ellipse = TRUE, ellipse.alpha= 0.1,
             palette = "jco",repel = TRUE, ggtheme = theme_minimal(), 
             main= FALSE, xlab= FALSE, ylab = FALSE)

# To see better in which cluster a player is, create a dataframe
Clusters=data.frame(sort(km.res$cluster));
head(Clusters)

# hierarchial clustering 

res.hc <- hclust(res.dist, method = "ward.D2" )

# Visualize using factoextra
fviz_dend(res.hc, k = 6, # Cut in 6 groups
          cex = 0.5, # label size
          horiz= TRUE, rect = TRUE # Add rectangle around groups
)

#very interesting plot, where we can see which player is the most similar to other
#Interesting to see that the most similar player to Lebron James, the best player of the last decade, is Luka Doncic, the emerging European player that grew up in Real Madrid

# PREDICT ALL-NBA 2020-----------------------------------------------------------------------------------------------------

all.nba <- read.csv("C:/Users/Alex/Videos/Captures/big data/Assigment3/All.NBA.1984-2018.csv", stringsAsFactors = FALSE, header = TRUE, skip = 1) #best 15 player each year
nba.players <- read.csv("C:/Users/Alex/Videos/Captures/big data/Assigment3/Seasons_Stats.csv", stringsAsFactors = FALSE, header = TRUE) #stats players 1950-2018
traditionalStats_TOTAL<- read.csv("nba_players_2020_stats_total.csv", sep=";") #https://www.basketball-reference.com/leagues/NBA_2020_totals.html #we need total stats, no per 36min, for this part

mergedData2<-merge(traditionalStats_TOTAL,advancedStats)
#Same changes as MergedData
mergedData2<- mergedData2[-c(1,43,48)]
mergedData2$Year<- 2020
mergedData2$X3P.[is.na(mergedData2$X3P.)] <- 0
sapply(mergedData2, function(x) sum(is.na(x)))
mergedData2<-na.omit(mergedData2)

#transform season 1999-2000 to year 2000 for example for All-NBA dataset
all.nba$Year <- as.numeric(substr(all.nba$Season, start = 1, stop = 4)) + 1

#Filter players from 1998 (to reduce data and get recent data, since NBA has changed a lot in the last 30 years)
nba.players.post1998 <- nba.players %>% filter(Year > 1998)
all.nba.post1998 <- all.nba %>% filter(Year > 1998 & Year < 2018)

#Eliminate unused columns
nba.players.post1998$blanl <- NULL
nba.players.post1998$blank2 <- NULL
#look for NAs
sapply(nba.players.post1998, function(x) sum(is.na(x)))
sapply(all.nba.post1998, function(x) sum(is.na(x)))
#Put 0% for players with 0 attemps in 3-pointers, some decent players never shoot from 3 but they are still important for the dataset
nba.players.post1998$X3P.[is.na(nba.players.post1998$X3P.)] <- 0
all.nba.post1998$X3P.[is.na(all.nba.post1998$X3P.)] <- 0
#other stats with NA are irrelevant players, so delete rows
nba.players.post1998<-na.omit(nba.players.post1998)
#look for NA again
sapply(nba.players.post1998, function(x) sum(is.na(x)))
sapply(all.nba.post1998, function(x) sum(is.na(x)))
#Eliminate 2 columns that mergedData2 dataset doesn't have
nba.players.post1998<-nba.players.post1998[-c(1)]

#add 2020 to the players dataset
fdf<-rbind(nba.players.post1998,mergedData2)

#check NA
sum(is.na(fdf))

#Eliminate double-counting of players, eliminate "TOT" team

fdf <- subset(fdf, !Tm == "TOT")

#same for all-nba dataset
which(all.nba.post1998$Tm == "TOT")
#except 2 players who played most of the games with 1 team:
all.nba.post1998[239, 5] <- "ATL"
all.nba.post1998[180, 5] <- "DEN"

#transform stats to stats/per game and reduce columns that are similar or not useful for this

fdf.pergame <- fdf %>% mutate(Name = Player, Position = Pos, age = Age, year = Year,  Team = Tm, Games = G, Starts = GS, Minutes = MP/G, Points = PTS/G, Rebounds = TRB/G, Assists = AST/G, Steals = STL/G, Blocks = BLK/G, Turnovers = TOV/G, Fouls = PF/G, FTs = FT/G, Threes = X3P/G, FGs = FG/G, Usage = USG., EfficiencyRating = PER, BoxPlusMinus = BPM,ShootingPercentage = eFG.)
fdf.pergame <- fdf.pergame[ , c(51:72)] #keep only the new columns created
two.digit.round <- function(x) round(x, 2) #round 2 decimals
fdf.pergame[ , c(8:22)] <- sapply(fdf.pergame[ , c(8:22)], two.digit.round)

#at least 10 games played and average 5min per game
fdf.pergame <- fdf.pergame %>% filter(Games > 10 & Minutes > 5)

#ALL-NBA and fdf.pergame datasets can't be joined together because they have different columns
#so let's create a flag in fdf.pergame when a player was ALL-NBA
#first create a ID with name,age,team and year for both datasets
fdf.pergame$ID <- str_c(substr(fdf.pergame$Name, start = 1, stop = 3), substr(fdf.pergame$age, start = 1, stop = 2), substr(fdf.pergame$Team, start = 1, stop = 3), substr(fdf.pergame$year, start = 3, stop = 4), sep = "")
all.nba.post1998$ID <- str_c(substr(all.nba.post1998$Player, start = 1, stop = 3), substr(all.nba.post1998$Age, start = 1, stop = 2), substr(all.nba.post1998$Tm, start = 1, stop = 3), substr(all.nba.post1998$Year, start = 3, stop = 4), sep = "")
fdf.pergame$All.NBA <- ifelse(fdf.pergame$ID %in% all.nba.post1998$ID, 1, 0) #create flag, if true 1
sum(fdf.pergame$All.NBA)
#should be 285 instead of 286, 15 per year, so let's see what happens 
fdf.pergame.check <- fdf.pergame %>% filter(All.NBA == 1) %>% group_by(year) %>% summarise(length(Name))
fdf.pergame.check
#1 error in 2013
fdf.pergame[fdf.pergame$year == 2013 & fdf.pergame$All.NBA == 1, ]
#James Anderson is the impostor! he is not a good player, so bring him out of this list

fdf.pergame[6029, 24] <- 0
#check again
sum(fdf.pergame$All.NBA) #good

#include ALL-NBA year 2020 manually (3 teams = 15 players) (We want to predict this!!)
#1st team
fdf.pergame[8375, 24] <- 1 #Luka Doncic
fdf.pergame[8501, 24] <- 1 #Lebron James
fdf.pergame[8369, 24] <- 1 #Giannis Antetokounmpo
fdf.pergame[8362, 24] <- 1 #Anthony Davis
fdf.pergame[8443, 24] <- 1 #James Harden
#2nd team
fdf.pergame[8693, 24] <- 1 #Kawhi Leonard
fdf.pergame[8546, 24] <- 1 #Pascal Siakam
fdf.pergame[8512, 24] <- 1 #Nikola Jokic
fdf.pergame[8549, 24] <- 1 #Damian Lillard
fdf.pergame[8642, 24] <- 1 #Chris Paul
#3rd team
fdf.pergame[8795, 24] <- 1 #Jimmy Butler
fdf.pergame[8707, 24] <- 1 #Jayson Tatum
fdf.pergame[8421, 24] <- 1 #Rudy Gobert
fdf.pergame[8696, 24] <- 1 #Ben Simmons
fdf.pergame[8747, 24] <- 1 #Russell Westbrook

#Let's start getting interesting data of our datasets
#Density graph of players with the mean
points_density <- ggplot(fdf.pergame, aes(Points)) + geom_density(fill = "skyblue") + geom_vline(aes(xintercept = mean(Points)), linetype = "dashed")
rebounds_density <- ggplot(fdf.pergame, aes(Rebounds)) + geom_density(fill = "lightpink") + geom_vline(aes(xintercept = mean(Rebounds)), linetype = "dashed")
assists_density <- ggplot(fdf.pergame, aes(Assists)) + geom_density(fill = "tomato") + geom_vline(aes(xintercept = mean(Assists)), linetype = "dashed")
turnovers_density <- ggplot(fdf.pergame, aes(Turnovers)) + geom_density(fill = "darkolivegreen3") + geom_vline(aes(xintercept = mean(Turnovers)), linetype = "dashed")
grid.arrange(points_density, rebounds_density, assists_density, turnovers_density, ncol = 2)
#points by age
fdf.by.age <- fdf.pergame %>% group_by(age) %>% summarise(Points = mean(Points), Players = length(Name))
ggplot(fdf.by.age, aes(age, Points)) + geom_point(aes(size = Players), colour = "lightblue") + geom_smooth(method = "loess", colour = "grey", se = FALSE, linetype = "dashed") + labs(y="Points per game") + theme_bw()
#We see players "top level" is around 25-30 years, mixing good physical level and a lot of experience

#Let's show the evolution of the game in NBA in the last years
year_3points<- nba.players %>% filter(Year>1979) %>% group_by(Year) %>%  summarise(X3PA = sum(X3PA))
ggplot(data=year_3points, aes(x=Year, y=X3PA)) +
  geom_bar(stat="identity" , fill = "tomato")
#The 2 outliers are 1999 and 2011 where the season was shortened to 50 and 66 games respectively due to the strike(huelga)


#See correlation between predictors
fdf.vars.matrix <- as.matrix(fdf.pergame[ , c(6:22)]) #without name,year,team...
corrplot(cor(fdf.vars.matrix), is.corr = FALSE, method = "square", type = "full")
#We see interesting correlations, mostly positive but some negatives like threes and block (people who make blocks never shoot threes)

#Predict All NBA selection for 2020

fdf.train <- fdf.pergame %>% filter(year < 2020)
fdf.test <- fdf.pergame %>% filter(year == 2020)
dim(fdf.train)
dim(fdf.test)

#Random forest

set.seed(1)  ## set a random seed
#RFmodel <- randomForest(All.NBA ~ Points + Assists + Rebounds + age + Games + Starts + Minutes + Steals + Blocks + Turnovers + Fouls + FTs + Threes + FGs + Usage + EfficiencyRating + BoxPlusMinus + ShootingPercentage, data = fdf.train)
#to solve warning message:
fdf.train$All.NBA <- as.character(fdf.train$All.NBA) # 
fdf.train$All.NBA <- as.factor(fdf.train$All.NBA)
#do regression again:
RFmodel <- randomForest(All.NBA ~ Points + Assists + Rebounds + age + Games + Starts + Minutes + Steals + Blocks + Turnovers + Fouls + FTs + Threes + FGs + Usage + EfficiencyRating + BoxPlusMinus + ShootingPercentage, data = fdf.train)
plot(RFmodel) ## check errors
varImpPlot(RFmodel) ## to look at variable importance
#the most important variable to be ALL-NBA is the efficiency of the player followed by Plus/Minus points (difference of points for the team when the player is playing and not playing) in the court and points scored by the player.

#Predict ALL-NBA for 2020
RF_pred <- predict(RFmodel, fdf.test, type = "response")
which(RF_pred==1)
table(fdf.test$All.NBA,RF_pred)
#the model choose 35 players, but we only want 15. 
#Good news, all the 15 ALL-NBA were selected in the model

#Let's try with the probability, that is, the 15 players with more probabilities to be ALL-NBA according the model
prob.RF_pred <- predict(RFmodel, fdf.test, type = "prob")
fdf.test.prob <- cbind(fdf.test, prob.RF_pred)
names(fdf.test.prob)[names(fdf.test.prob) == "1"] <- "Probability"
fdf.top15 <- fdf.test.prob  %>% top_n(n = 15, wt = Probability) %>% arrange(desc(Probability))
fdf.top15$All.NBA <- as.numeric(as.character(fdf.top15$All.NBA))
round((sum(fdf.top15$All.NBA)/length(fdf.top15$All.NBA)*100), 2)
#the model only got 50% of the players right

#Let's try random forest again but only with the 7 most significant variables

RFmodel2 <- randomForest(All.NBA ~ EfficiencyRating + BoxPlusMinus + Points + FGs + FTs + Usage,data = fdf.train)

prob.RF_pred2 <- predict(RFmodel2, fdf.test, type = "prob")
fdf.test.prob2 <- cbind(fdf.test, prob.RF_pred2)
names(fdf.test.prob2)[names(fdf.test.prob2) == "1"] <- "Probability"
fdf.top15.2 <- fdf.test.prob2  %>% top_n(n = 15, wt = Probability) %>% arrange(desc(Probability))
fdf.top15.2$All.NBA <- as.numeric(as.character(fdf.top15.2$All.NBA))
round((sum(fdf.top15.2$All.NBA)/length(fdf.top15.2$All.NBA)*100), 2)
#60%! better

#However, we are missing two important things: Playing a lot of games of the season and playing in a winning team
#Good players in bad teams, don't usually get in ALL-NBA because they outstand in the team because there are no other players who make good stats so it is easier to them make good numbers.
#We don't have the winning of each team for the training_set, that is the last 30 years wins of each team per year
#However, we can make an approximation that for be ALL-NBA his team has to win at least 47% of the games of the season 
#and play at least 50 games

perc_wins_2020<-read.csv("NBA_wins_2020.csv",sep=";") #I created this dataset in excel containing Team and % wins in regular season

colnames(perc_wins_2020) <- c("Team", "Wins") #change column names
  
wins_2020<- perc_wins_2020 %>% filter(Wins>0.47)
fdf.test.prob3 <- cbind(fdf.test, prob.RF_pred2)
fdf.test.prob3$GoodTeam <- (ifelse(fdf.test.prob3$Team %in% wins_2020$Team, 1, 0))
names(fdf.test.prob3)[names(fdf.test.prob3) == "1"] <- "Probability"
fdf.top15.3 <- fdf.test.prob3 %>% filter(GoodTeam==1) %>% filter(Games>40) %>% top_n(n = 15, wt = Probability) %>% arrange(desc(Probability))
fdf.top15.3$All.NBA <- as.numeric(as.character(fdf.top15.3$All.NBA))
round((sum(fdf.top15.3$All.NBA)/length(fdf.top15.3$All.NBA)*100), 2)
#66.66%! better. Interesting the 8 most probable are correct, the other 7 except 1 are wrong.

#Finally, let's change make the test_set data bigger to see if this year if this year was atypical or the model should be improved

fdf.train2 <- fdf.pergame %>% filter(year < 2014)
fdf.test2 <- fdf.pergame %>% filter(year > 2013)
dim(fdf.train2)
dim(fdf.test2)


set.seed(1)  ## set a random seed

fdf.train2$All.NBA <- as.character(fdf.train2$All.NBA) #solve warning error like before
fdf.train2$All.NBA <- as.factor(fdf.train2$All.NBA)
#do regression again:
RFmodel3 <- randomForest(All.NBA ~ Points + Assists + Rebounds + age + Games + Starts + Minutes + Steals + Blocks + Turnovers + Fouls + FTs + Threes + FGs + Usage + EfficiencyRating + BoxPlusMinus + ShootingPercentage, data = fdf.train2)

prob.RF_pred3 <- predict(RFmodel3, fdf.test2, type = "prob")
fdf.test.prob4 <- cbind(fdf.test2, prob.RF_pred3)
names(fdf.test.prob4)[names(fdf.test.prob4) == "1"] <- "Probability"
fdf.top15.4 <- fdf.test.prob4 %>% group_by(year)  %>% top_n(n = 15, wt = Probability) %>% arrange(year,desc(Probability))
fdf.top15.4$All.NBA <- as.numeric(as.character(fdf.top15.4$All.NBA))
round((sum(fdf.top15.4$All.NBA)/length(fdf.top15.4$All.NBA)*100), 2)
#72%!!!! better prediction considering we didn't take into account being in a good team or minimum games played

which(fdf.top15.4$All.NBA == 0 & fdf.top15.4$Probability > 0.80)
formattable(fdf.top15.4[c(4,6,34,51,52,66,67,69,70,71), ])
#Half of the errors are from this year, atypical maybe due to covid and a shorter season
#The model could be much better if we included information of the %wins of the teams in all years.
#For example let's analyze the players the model thought >80% should be in but they were not:
#4. DeMarcus Cousins in 2013, in SAC, 34% wins (seen as a toxic player)
#6 Carmelo Anthony in 2014,in NYK, 45% wins
#34 James Harden in 2016,in HOU, 50% wins (not very liked player)
#51 Karl-Anthony Towns in 2017, in MIN, 37% wins
#52 DeMarcus Cousins in 2017, in SAC, 39% wins (seen as a toxic player)
#NO DATA OF 2018 AND 2019
#66 Devin Booker in 2020, in PHO, 46% wins
#67 DeMar DeRozan in 2020, in SAS, 45% wins
#69 Brandon Ingram in 2020, in NOP, 41% wins
#70 Donovam Mitchel in 2020,in UTA, 61% wins (This player should have been included in ALL-NBA 2020!!!)
#71 Zach Lavine in 2020, in CHI, 33% wins


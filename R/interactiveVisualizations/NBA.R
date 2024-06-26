library(datasetsICR)
library(dplyr)
library(ggplot2)

data(NBA.game)
print(NBA.game)
print(typeof(NBA.game))
print(class(NBA.game))

# How many teams do we have?
n_teams <- length(unique(NBA.game$TEAM))
sprintf("There are %d teams in the NBA.", n_teams)

# How many players are listed here?
n_players <- length(unique(NBA.game$PLAYER))
sprintf("This dataset contains %d players.", n_players)

# What is the team with most players?
players_per_team <- NBA.game %>%
  group_by(NBA.game$TEAM) %>%
  summarise(player_count = n()) %>%
  rename("TEAM" = `NBA.game$TEAM`) %>%
  arrange(desc(player_count)) %>%
  top_n(5)

print(paste0("The team with most players is ", players_per_team$TEAM[1], "."))

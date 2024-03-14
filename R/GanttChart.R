library(tidyverse)
library(ggplot2)
library(dplyr)

df <- read_csv("data/trace.csv")

df <- df %>%
  group_by(worker) %>%
  mutate(Position = cur_group_id()) %>%
  mutate(Position2 = case_when(
    worker == "host" ~ 2,
    worker == "gpu 0" ~ 1,
    worker == "gpu 1" ~ 0,
  )) %>%
  ungroup()

duration <- max(df$end) - min(df$start)
df_time <- df %>%
  group_by(worker) %>%
  mutate(Time = end-start) %>%
  summarise(ComputationTime = sum(Time),
            Duration = duration-min(start),
            Position2 = Position2,
            IdleTime = Duration-ComputationTime,
            IdlePercentage = paste0(round(IdleTime/Duration, 4) * 100, "%")
  ) %>%
  unique()

print(df_time)



plt <- df %>%
  ggplot() +
  geom_segment(mapping =
    aes(
      y = fct_rev(fct_inorder(worker)),
      yend = fct_rev(fct_inorder(worker)),
      x = start,
      xend = end,
      color = name),
    linewidth = 10
  ) +
  geom_label(data=df_time, mapping = aes(x=duration, y = Position2+1, label=IdlePercentage)) +
  labs(title="Gantt Chart",
       x = "Time(s)",
       y = "Worker",
       colour = "Name") +
  theme_bw()


print(plt)
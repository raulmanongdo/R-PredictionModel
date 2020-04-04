#---------------------------------------------------------------------------------------
# Web scraping in R 
# https://towardsdatascience.com/tidy-web-scraping-in-r-tutorial-and-resources-ac9f72b4fe47
#---------------------------------------------------------------------------------------


if (!('rvest' %in% installed.packages())) {
  install.packages('rvest')
}
if (!('dplyr' %in% installed.packages())) {
  install.packages('dplyr')
}
library(rvest)
library(dplyr)

hot100page <- 'https://www.billboard.com/charts/hot-100'
hot100 <- read_html(hot100page)
hot100
str(hot100)

body_nodes <- hot100 %>% html_node('body') %>% html_children()
body_nodes %>% html_children()

rank <- hot100 %>% 
  rvest::html_nodes('body') %>% 
  xml2::xml_find_all("//span[contains(@class, 'chart-element__rank__number')]") %>% 
  rvest::html_text()
artist <- hot100 %>% 
  rvest::html_nodes('body') %>% 
  xml2::xml_find_all("//span[contains(@class, 'chart-element__information__artist')]") %>% 
  rvest::html_text()
title <- hot100 %>% 
  rvest::html_nodes('body') %>% 
  xml2::xml_find_all("//span[contains(@class, 'chart-element__information__song')]") %>% 
  rvest::html_text()

chart_df <- data.frame(rank, artist, title)
knitr::kable(
  chart_df %>% head(10)
)


  
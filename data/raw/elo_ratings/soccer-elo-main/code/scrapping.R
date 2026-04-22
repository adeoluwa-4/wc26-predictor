library(tidyverse)
library(rvest)
library(RSelenium) # because data are based on Javascript
library(stringr)
library(data.table)
# library(glue)
library(xml2)

##### eloratings site #####
# connection
remDr <- RSelenium::rsDriver(
  port = 4567L,
  browser = c("firefox"),
  version = "latest",
  chromever = NULL
)

brow <- remDr[["client"]]
brow$open()

# function to extract dataframe
scapping_element <- function(url){
  print(url)
  
  brow$navigate(url)
  Sys.sleep(5)
  pagesource <- brow$getPageSource()
  Sys.sleep(3)
  
  thisurlread <- read_html(x = pagesource[[1]]) # reading html page
  
  mycharacter <- thisurlread %>%
    html_element(xpath = '/html/body/div[1]/div[2]/div[1]/div[6]') %>%
    html_text2()
  
  mycharacter <-strsplit(x = mycharacter, split = '\n')
  mycharacter <- mycharacter[[1]]
  
  tibbleoutput <- tibble()
  for (i in 1:(length(mycharacter)/16)) {
    vector_to_match_init <- if (i != 1) {
      (i-1)*16 + 1
    } else {1}
    scap <- tibble(year = str_sub(url, -4), rank = mycharacter[[vector_to_match_init]],
                   team = mycharacter[[vector_to_match_init+1]],
                   rating = mycharacter[[vector_to_match_init+2]],
                   average_rank = mycharacter[[vector_to_match_init+3]],
                   average_rating = mycharacter[[vector_to_match_init+4]],
                   one_year_change_rank = mycharacter[[vector_to_match_init+5]],
                   one_year_change_rating = mycharacter[[vector_to_match_init+6]],
                   matches_total = mycharacter[[vector_to_match_init+7]],
                   matches_home = mycharacter[[vector_to_match_init+8]],
                   matches_away = mycharacter[[vector_to_match_init+9]],
                   matches_neutral = mycharacter[[vector_to_match_init+10]],
                   matches_wins = mycharacter[[vector_to_match_init+11]],
                   matches_losses = mycharacter[[vector_to_match_init+12]],
                   matches_draws = mycharacter[[vector_to_match_init+13]],
                   goals_for = mycharacter[[vector_to_match_init+14]],
                   goals_against = mycharacter[[vector_to_match_init+15]])
    # data in a tibble of rank, team, rating
    tibbleoutput <- tibbleoutput %>%
      bind_rows(scap)
  }
  return(tibbleoutput)
}

#### url
urlbase <- "https://www.eloratings.net/"

# empty list and loop to generate list of dataframes
datesseq <- seq(1901, 2023, 1) # 1901 until 2023
mesurls <- list()

for (i in 1:length(datesseq)) {
  mesurls[[i]] <- paste0(urlbase,datesseq[i])
}

# extract list of df
list_of_results <- list()
for (i in 1:length(mesurls)) {
  list_of_results[[i]] <- scapping_element(url = mesurls[[i]])
}

# compilation of list of dataframes
results <- rbindlist(l = list_of_results) %>%
  as_tibble()

# output
write_csv(x = results, file = "csv/ranking_soccer_1901-2023.csv")

# closing connection
remDr[["client"]]$close()

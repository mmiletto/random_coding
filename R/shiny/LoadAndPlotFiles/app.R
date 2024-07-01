# read a file and create some interactive line plots

library(shiny)
library(tidyverse)
library(ggplot2)
library(plotly)
library(bslib)

# Define UI for data upload app ----
ui <- fluidPage(

	# App title ----
	titlePanel("Uploading Files"),

	# Sidebar layout with input and output definitions ----
	sidebarLayout(

		# Sidebar panel for inputs ----
		sidebarPanel(

			# Input: Select a file ----
			fileInput("files", "Choose CSV File",
					  multiple = TRUE,
					  accept = c("csv",
								 "comma-separated-values",
								 ".csv")),

			# Horizontal line ----
			tags$hr(),

			# set density error hline
            numericInput("max_density_error", "Maximum density error:", value = 0.0001),

			# Horizontal line ----
			tags$hr(),

			# Input: Select number of rows to display ----
			radioButtons("disp", "Display",
						 choices = c(Head = "head",
									 All = "all"),
						 selected = "head")

		),

		# Main panel for displaying outputs ----
		mainPanel(
			# Output: Data file ----
			card(plotlyOutput("density_error"))
		)
	)
)


# Define server logic to read selected file ----
server <- function(input, output) {

	output$density_error <- renderPlotly({

		# input$files will be NULL initially. After the user selects
		# and uploads a file, head of that data file by default,
		# or all rows if selected, will be shown.

		req(input$files)
		print(input$files$datapath)
		df <- read.csv(input$files$datapath, header = TRUE) %>%
            mutate(converged = ifelse(converged == 1, TRUE, FALSE)) %>%
            mutate(df_name = input$files$name)

        # check header values, to cerify validity of input file
        expected_columns <- c("time", "timestep", "iteration", "converged", "density_error")
        for (column in expected_columns) {
            validate(
                need((column %in% colnames(df)), paste0("Missing column ", column, " in dataframe."))
            )
        }

        #             mutate(df_name = input$files$name)
		# TODO Load multiple files and concatenate them
		# vars <- reactiveValues(df = NULL)
		#     observeEvent(
        #     input$files$datapath,
        #     {
        #         req(input$files)
        #         print(paste0("Reading: ", input$files$datapath))
        #         temp <- read.csv(input$files$datapath, header = TRUE)
        #         # check header values, to cerify validity of input file
        #         expected_columns <- c("time", "timestep", "iteration", "converged", "density_error")
        #         for (column in expected_columns) {
        #             validate(
        #                 need((column %in% colnames(temp)), paste0("Missing column ", column, " in dataframe."))
        #             )
        #         }
        #         temp <- temp %>%
        #             mutate(converged = ifelse(converged == 1, TRUE, FALSE)) %>%
        #             mutate(df_name = input$files$name)
        #         vars$df <- as.data.frame(temp)
        #     }
		# )

		plot <- df %>%
			plot_ly(x = ~time, y = ~density_error, color = ~df_name) %>%
			add_lines()
	})

}

# Create Shiny app ----
shinyApp(ui = ui, server = server)
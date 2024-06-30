# read a file and create some interactive line plots

library(shiny)

# Define UI for data upload app ----
ui <- fluidPage(

  # App title ----
  titlePanel("Uploading Files"),

  # Sidebar layout with input and output definitions ----
  sidebarLayout(

    # Sidebar panel for inputs ----
    sidebarPanel(

      # Input: Select a file ----
      fileInput("file1", "Choose CSV File",
                multiple = TRUE,
                accept = c("csv",
                         "comma-separated-values",
                         ".csv")),

      # Horizontal line ----
      tags$hr(),

      # Input: Checkbox if file has header ----
      checkboxInput("header", "Header", TRUE),


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
      tableOutput("contents")
    )
  )
)


# Define server logic to read selected file ----
server <- function(input, output) {

  output$contents <- renderTable({

    # input$file1 will be NULL initially. After the user selects
    # and uploads a file, head of that data file by default,
    # or all rows if selected, will be shown.

    req(input$file1)
    df <- read.csv(input$file1$datapath, header = input$header)

    # check header values, to cerify validity of input file
    expected_columns <- c("time","timestep","iteration","converged","density_error")
    for(column in expected_columns) {
      if (!(column %in% colnames(df))){
        return (paste0("Missing column ", column, " in dataframe."))
      }
    }

    if(input$disp == "head") {
      return(head(df))
    }
    else {
      return(df)
    }

  })

}

# Create Shiny app ----
shinyApp(ui = ui, server = server)
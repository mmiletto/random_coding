# A sinhy app lives in a folder. This folder must contain a file named app.R
# - It is recommended that each app will live in its own unique directory.
#
# The app.R file has three components:
#
#     1. a user interface object
#       - The user interface (ui) object controls the layout and appearance of your app.
#     2. a server function
#       - The server function contains the instructions that your computer needs to build your app.
#     3. a call to the shinyApp function
#       - Finally the shinyApp function creates Shiny app objects from an explicit UI/server pair.

library(shiny)
library(bslib)

# Define UI for app that draws a histogram ----
ui <- page_sidebar(
  # App title ----
  title = "Hello World!",
  # Sidebar panel for inputs ----
  sidebar = sidebar(
    # Input: Slider for the number of bins ----
    sliderInput(
      inputId = "bins",
      label = "Number of bins:",
      min = 5,
      max = 50,
      value = 25
    )
  ),
  # Output: Histogram ----
  plotOutput(outputId = "distPlot")
)


# Define server logic required to draw a histogram ----
server <- function(input, output) {

  # Histogram of the Old Faithful Geyser Data ----
  # with requested number of bins
  # This expression that generates a histogram is wrapped in a call
  # to renderPlot to indicate that:
  #
  # 1. It is "reactive" and therefore should be automatically
  #    re-executed when inputs (input$bins) change
  # 2. Its output type is a plot
  output$distPlot <- renderPlot({

    x <- faithful$waiting
    bins <- seq(min(x), max(x), length.out = input$bins + 1)

    hist(x, breaks = bins, col = "#007bc2", border = "orange",
         xlab = "Waiting time to next eruption (in mins)",
         main = "Histogram of waiting times")
  })

}

shinyApp(ui = ui, server = server)
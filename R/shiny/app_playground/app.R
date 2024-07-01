ui <- fluidPage(

    selectizeInput(inputId = 'select_input', label = 'Choose your files...', choices = '*', multiple = TRUE),
    verbatimTextOutput('debug')
)

server <- function(input, output, session) {

    observe({
        files <- list.files()

        updateSelectizeInput(session = session, inputId = 'select_input', choices = files)
    })

    output$debug <- renderPrint({input$select_input})
}

shinyApp(ui, server)
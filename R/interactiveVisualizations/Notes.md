# Interactive web-based data visualization with R, plotly, and shiny

## R useful functions

- group_by()
- summarise()
- arrange() -> sort
- top_n() -> filter top n entries
- mutate()
- left_join()
- lapply()
- split()

## 2. Overview

Two different forms
	ggplot2 -> plotly 
	direct call for plotly functions: `plot_ly()/plot_geo()/plot_mapbox()`


Almost every plotly function expects a plotly object as its first argument and returns a modified version of it. So we
have the called "purely functional approach". The `%>%` operator from the `magrittr` plays a fundamental role to keep 
the code simple, by rearranging code in kind of pipeline steps.

""""
A layer can be thought of as a group of graphical elements that can be sufficiently described using only 5 components: 
data, aesthetic mappings (e.g., assigning clarity to color), a geometric representation (e.g. rectangles, circles, etc),
statistical transformations (e.g., sum, mean, etc), and positional adjustments (e.g., dodge, stack, etc).
"""


In many scenarios, it can be useful to combine multiple graphical layers into a single plot. In this case, it becomes 
useful to know a few things about plot_ly():
- Arguments specified in plot_ly() are global, meaning that any downstream add_*() functions inherit these arguments (unless inherit = FALSE).
- Data manipulation verbs from the dplyr package may be used to transform the data underlying a plotly object.3

- `plotly_data()` Useful for debug, get the actual state of the plotly data in downstream.

Alternatively, `ggplotly()` can still be desirable for creating visualizations that aren’t necessarily
straight-forward to achieve without it (using `plot_ly()` instead).


It’s also worth mentioning that ggplotly() conversions are not always perfect and ggplot2 doesn’t provide an API for 
interactive features, so sometimes it’s desirable to modify the return values of ggplotly().


## 3. Scattered foundations

**Semantic mappings**: Alpha, Color, Symbols, Stroke and span, Size

A plotly.js figure contains one (or more) `trace(s)`, and every trace has a type. One trace type example is `scatter`.

Traces can have `markers` -> `add_markers(category, name = "Item Category")`

TIP: consider using `toWebGL()` to render plots using Canvas rather than SVG, example in cases with lots 
(tens of thousands) of data points for scatter plots.

Mapping a discrete variable to color produces one `trace` per `category`, this can impact the performance of the plot
in exchange for more interactivity.

`color` and `stroke` -> fill and outline.

### Lines

TIP: Generally speaking, it’s hard to perceive more than 8 different colors/linetypes/symbols in a given plot, so \
sometimes we have to filter data to use these effectively.

### Other plots

- Dumbell (example km per liter in the city or highway)
- Candlestick
- Density (like an hist)
- Parallel coordinates

### Polygons
Polygons can be use to draw many things, but perhaps the most familiar application where you might want to use 
`add_polygons()` is to draw geo-spatial objects

Atip: Can use `split` to categorise and separate data.

### Ribbons
Use them to show uncertainty



## 4. Maps

-> Not very usefull section for me. But it has lots of good examples on geospatial data representation.\

## 5. Bars and Histograms

`add_bars()`
- require both X and Y variables

`add_histogram()`
- require only one dimensional variable


## 6. Boxplots

`add_boxplot()`
- Receives one value, and can ble split into a categorical variable x. 
- It is usefull to sort boxplots by something meaningfull, like the median.

## 7. 2D Frequencies

`add_heatmap()`
- 2D analog of `add_bars()`

`add_histogram2d()` 
- 2D analog of `add_histogram()`
- `zsmooth` parameter changes the number of bins. nbins are defined in x and y.

## 8. 3D Charts

Adding a z attribute to `plot_ly()` it will know how to render markers, lines, and paths in 
three dimensions.

To add axes label to a 3D plot we must use `scene` and `layout()`

`add_surface()`
- Like a 3D heatmap


## 9. Introduction to publishing views

How to save plotly graphs as HTML documents or embed them into larger HTML documents.
Or export lots of plots at once `orca()` function: Static image exporting via orca.

## 10. Saving and embedding HTML
## 11. Exporting Static Images
Either use `orca` form `plotly` or modify and download the image from your browser.
## 12. Editing views for publishing
Can create a shiny app that listens to the `plotly_relayout`. Adjust tags, for example.

# 13. Arranging views'

`subplot()` allows to merge multiple plotly objects.
	- nrows
	- heights 
	- widths -> can define the number of columns

` subplot()` can be used recursively, as it returns a plotly object.


## 13.1 Scatterplot matrices
Alternative to composing multiple interacting plotly objects.
`splom` See https://plot.ly/r/splom/ for more options related to the splom trace type.


## 13.2 Other ways to compose plots

ggplot2 `facet_wrap` and `facet_grid` or `ggmatrix`


## 13.3 Arranging htmlwidgets

A `plotly` object is a `htmlwidgets`, which are `htmltools`. So any arranging method for the latter will work for 
the others. Such methods are:
### 13.3.1 flexdashboard
### 13.3.2 Bootstrap’s grid layout
### 13.3.3  CSS flexbox

# Interactive web-based data visualization with R, plotly, and shiny

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
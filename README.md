import dash
from dash import dcc, html, Input, Output
import networkx as nx
from flask import Flask
import io
import base64
import matplotlib.pyplot as plt

# Initialize Dash app with Flask server
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Flight route map with distances and travel times 
map = {
    "Delhi": {
        "Mumbai": {"distance": 1147, "time": 2.0},
        "Bangalore": {"distance": 1740, "time": 2.8},
        "Chennai": {"distance": 2180, "time": 3.2},
        "Kolkata": {"distance": 1300, "time": 2.2},
        "Hyderabad": {"distance": 1500, "time": 2.5},
        "Pune": {"distance": 1415, "time": 2.4},
        "Jaipur": {"distance": 270, "time": 0.7},
        "Ahmedabad": {"distance": 770, "time": 1.4},
        "Lucknow": {"distance": 490, "time": 1.0},
    },
    "Mumbai": {
        "Delhi": {"distance": 1147, "time": 2.0},
        "Bangalore": {"distance": 840, "time": 1.6},
        "Chennai": {"distance": 1030, "time": 1.8},
        "Kolkata": {"distance": 1660, "time": 2.7},
        "Hyderabad": {"distance": 630, "time": 1.3},
        "Pune": {"distance": 150, "time": 0.5},
        "Jaipur": {"distance": 930, "time": 1.7},
        "Ahmedabad": {"distance": 445, "time": 1.0},
        "Lucknow": {"distance": 1240, "time": 2.1},
    },
    "Bangalore": {
        "Delhi": {"distance": 1740, "time": 2.8},
        "Mumbai": {"distance": 840, "time": 1.6},
        "Chennai": {"distance": 300, "time": 0.8},
        "Kolkata": {"distance": 1540, "time": 2.6},
        "Hyderabad": {"distance": 510, "time": 1.2},
        "Pune": {"distance": 740, "time": 1.5},
        "Jaipur": {"distance": 1820, "time": 2.9},
        "Ahmedabad": {"distance": 1300, "time": 2.2},
        "Lucknow": {"distance": 1880, "time": 3.0},
    },
    "Chennai": {
        "Delhi": {"distance": 2180, "time": 3.2},
        "Mumbai": {"distance": 1030, "time": 1.8},
        "Bangalore": {"distance": 300, "time": 0.8},
        "Kolkata": {"distance": 1660, "time": 2.7},
        "Hyderabad": {"distance": 630, "time": 1.3},
        "Pune": {"distance": 930, "time": 1.7},
        "Jaipur": {"distance": 2260, "time": 3.4},
        "Ahmedabad": {"distance": 1450, "time": 2.4},
        "Lucknow": {"distance": 2040, "time": 3.3},
    },
    "Kolkata": {
        "Delhi": {"distance": 1300, "time": 2.2},
        "Mumbai": {"distance": 1660, "time": 2.7},
        "Bangalore": {"distance": 1540, "time": 2.6},
        "Chennai": {"distance": 1660, "time": 2.7},
        "Hyderabad": {"distance": 1500, "time": 2.5},
        "Pune": {"distance": 1650, "time": 2.7},
        "Jaipur": {"distance": 1500, "time": 2.5},
        "Ahmedabad": {"distance": 1750, "time": 2.8},
        "Lucknow": {"distance": 680, "time": 1.4},
    },
    "Hyderabad": {
        "Delhi": {"distance": 1500, "time": 2.5},
        "Mumbai": {"distance": 630, "time": 1.3},
        "Bangalore": {"distance": 510, "time": 1.2},
        "Chennai": {"distance": 630, "time": 1.3},
        "Kolkata": {"distance": 1500, "time": 2.5},
        "Pune": {"distance": 550, "time": 1.3},
        "Jaipur": {"distance": 1400, "time": 2.4},
        "Ahmedabad": {"distance": 700, "time": 1.4},
        "Lucknow": {"distance": 1300, "time": 2.2},
    },
    "Pune": {
        "Delhi": {"distance": 1415, "time": 2.4},
        "Mumbai": {"distance": 150, "time": 0.5},
        "Bangalore": {"distance": 740, "time": 1.5},
        "Chennai": {"distance": 930, "time": 1.7},
        "Kolkata": {"distance": 1650, "time": 2.7},
        "Hyderabad": {"distance": 550, "time": 1.3},
        "Jaipur": {"distance": 880, "time": 1.6},
        "Ahmedabad": {"distance": 660, "time": 1.4},
        "Lucknow": {"distance": 1200, "time": 2.0},
    },
    "Jaipur": {
        "Delhi": {"distance": 270, "time": 0.7},
        "Mumbai": {"distance": 930, "time": 1.7},
        "Bangalore": {"distance": 1820, "time": 2.9},
        "Chennai": {"distance": 2260, "time": 3.4},
        "Kolkata": {"distance": 1500, "time": 2.5},
        "Hyderabad": {"distance": 1400, "time": 2.4},
        "Pune": {"distance": 880, "time": 1.6},
        "Ahmedabad": {"distance": 630, "time": 1.3},
        "Lucknow": {"distance": 500, "time": 1.1},
    },
    "Ahmedabad": {
        "Delhi": {"distance": 770, "time": 1.4},
        "Mumbai": {"distance": 445, "time": 1.0},
        "Bangalore": {"distance": 1300, "time": 2.2},
        "Chennai": {"distance": 1450, "time": 2.4},
        "Kolkata": {"distance": 1750, "time": 2.8},
        "Hyderabad": {"distance": 700, "time": 1.4},
        "Pune": {"distance": 660, "time": 1.4},
        "Jaipur": {"distance": 630, "time": 1.3},
        "Lucknow": {"distance": 1000, "time": 1.8},
    },
    "Lucknow": {
        "Delhi": {"distance": 490, "time": 1.0},
        "Mumbai": {"distance": 1240, "time": 2.1},
        "Bangalore": {"distance": 1880, "time": 3.0},
        "Chennai": {"distance": 2040, "time": 3.3},
        "Kolkata": {"distance": 680, "time": 1.4},
        "Hyderabad": {"distance": 1300, "time": 2.2},
        "Pune": {"distance": 1200, "time": 2.0},
        "Jaipur": {"distance": 500, "time": 1.1},
        "Ahmedabad": {"distance": 1000, "time": 1.8},
    },
}

def bellman_ford(graph, start, end):
    # ... (rest of the bellman_ford function remains the same)
    distances = {node: float('inf') for node in graph}
    times = {node: float('inf') for node in graph}  # To track the time
    distances[start] = 0
    times[start] = 0  # Start time is 0
    predecessors = {node: None for node in graph}

    # Relax edges |V| - 1 times
    for _ in range(len(graph) - 1):
        for node, neighbors in graph.items():
            for neighbor, info in neighbors.items():
                weight, time = info["distance"], info["time"]
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    times[neighbor] = times[node] + time  # Accumulate the time
                    predecessors[neighbor] = node

    # Check for negative weight cycles (not necessary here since no negative weights)
    path = []
    current_node = end
    while current_node is not None:
        path.insert(0, current_node)
        current_node = predecessors[current_node]

    if distances[end] == float('inf'):
        return None, None, None  # No path found

    return path, distances[end], times[end]

# ... (rest of the fig_to_base64 and create_graph_plot functions remain the same)
def fig_to_base64(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
    return img_b64

def create_graph_plot(graph, route=None):
    G = nx.Graph()
    for airport, connections in graph.items():
        for connection, info in connections.items():
            G.add_edge(airport, connection, weight=info["distance"])

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold', edge_color='gray', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    if route:
        path_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        for edge in path_edges:
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], [pos[edge[0]][1], pos[edge[1]][1]], 'r-', lw=3)

    # Convert to base64 and return
    return fig_to_base64(fig)

# Dash Layout
app.layout = html.Div(
    children=[
        html.H1("India Air Flight Route Planner", style={'text-align': 'center', 'color': '#007bff', 'font-family': 'Arial, sans-serif', 'text-shadow': '2px 2px 4px rgba(0, 0, 0, 0.2)', 'font-size': '36px', 'padding-bottom': '20px'}),
        
        html.Div(
            children=[
                html.Label("Select Airport From:", style={'font-size': '18px', 'color': '#007bff'}),
                dcc.Dropdown(
                    id="airport-from",
                    options=[{"label": airport, "value": airport} for airport in map.keys()],
                    value="Enter Value",
                    style={'width': '80%', 'padding': '10px', 'border-radius': '10px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin-bottom': '20px'}
                ),
                html.Label("Select Airport To:", style={'font-size': '18px', 'color': '#007bff'}),
                dcc.Dropdown(
                    id="airport-to",
                    options=[{"label": airport, "value": airport} for airport in map.keys()],
                    value="Enter Value",
                    style={'width': '80%', 'padding': '10px', 'border-radius': '10px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin-bottom': '20px'}
                ),
            ],
            style={'width': '80%', 'padding': '20px', 'background-color': '#f1f1f1', 'border-radius': '15px', 'box-shadow': '0 6px 12px rgba(0, 0, 0, 0.1)', 'margin-bottom': '30px', 'display': 'inline-block'}
        ),
        
        html.Button("Plot Route", id="plot-button", n_clicks=0, style={'margin-top': '20px', 'font-size': '20px', 'background-color': '#007bff', 'color': '#fff', 'border': 'none', 'padding': '15px 30px', 'border-radius': '10px', 'cursor': 'pointer', 'box-shadow': '0 6px 15px rgba(0, 0, 0, 0.1)', 'transition': 'background-color 0.3s'}),
        
        html.Div(id="route-output", style={'padding-top': '20px', 'text-align': 'center', 'font-size': '20px', 'color': '#007bff', 'font-weight': 'bold', 'line-height': '1.6'}),
        
        html.Img(id="route-graph", style={'height': '600px', 'margin-top': '30px', 'border-radius': '15px', 'box-shadow': '0 6px 20px rgba(0, 0, 0, 0.2)'})
    ],
    style={'background-color': '#f4f7fc', 'padding': '50px', 'text-align': 'center'}
)

# Callbacks
@app.callback(
    [Output("route-output", "children"),
     Output("route-graph", "src")],
    [Input("plot-button", "n_clicks")],
    [Input("airport-from", "value"),
     Input("airport-to", "value")]
)
def update_route(n_clicks, airport_from, airport_to):
    if n_clicks == 0:
        return "", None
        
    # Check if both selections are made
    if not airport_from or not airport_to:
        return "Please select both departure and destination airports.", None

    if airport_from == airport_to:
        return "Departure and destination airports cannot be the same.", None

    route, total_distance, total_time = bellman_ford(map, airport_from, airport_to)
    
    if not route:
        return f"No route found from {airport_from} to {airport_to}.", None

    path_with_distances = [
        f"{route[i]} -> {route[i + 1]} ({map[route[i]].get(route[i + 1], {'distance': 0})['distance']} km, {map[route[i]].get(route[i + 1], {'time': 0})['time']} hrs)"
        for i in range(len(route) - 1)
    ]

    route_str = f"The shortest flight route from {airport_from} to {airport_to} is:\n" + "\n".join(path_with_distances)
    route_str += f"\nTotal distance: {total_distance} km\nTotal time: {total_time} hours"
    
    # Generate graph image with the shortest path highlighted
    graph_image = create_graph_plot(map, route)
    
    return route_str, "data:image/png;base64," + graph_image

# Run the Flask app
if __name__ == "__main__":
    app.run_server(debug=True)
    

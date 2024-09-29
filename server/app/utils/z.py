import networkx as nx
from pyvis.network import Network
import webbrowser
import os

def create_knowledge_graph(files, G):
    for filename, classes, functions, methods, relations in files:
        # Extract directory information
        directories = filename.split('\\')[:-1]
        file_name_only = filename.split('\\')[-1]
        
        # Add directory nodes and edges
        path = ''
        for directory in directories:
            if path:
                parent_path = path
                path = os.path.join(path, directory)
            else:
                parent_path = directory
                path = directory
            if path not in G:
                G.add_node(path, label=directory, title=f'Directory: {directory}', color='#FFD700')  # Gold
                if parent_path != directory:  # Avoid self-loop for root directory
                    G.add_edge(parent_path, path, title='contains')
        
        # Add file node
        full_path = os.path.join(path, file_name_only)
        G.add_node(full_path, label=file_name_only, title=f'File: {file_name_only}', color='#FFA500')
        G.add_edge(path, full_path, title='contains')
        
        # Add nodes and edges for classes, functions, and methods
        for cls in classes:
            G.add_node(cls, label=cls, title=f'Class: {cls}', color='#FF6666')  # Red 
            G.add_edge(full_path, cls, title='contains')
        for func in functions:
            if func not in [method[1] for method in methods]:  
                G.add_node(func, label=func, title=f'Function: {func}', color='#66FF66')  # Green
                G.add_edge(full_path, func, title='contains')
        for cls, method in methods:
            G.add_node(method, label=method, title=f'Method: {method}', color='#6666FF')  # Blue 
            G.add_edge(cls, method, title='contains')
        for rel in relations:
            G.add_edge(rel[0], rel[1], title=rel[2])

    return G

def visualize_knowledge_graph(G, output_file='knowledge_graph.html'):
    net = Network(notebook=False, directed=True)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    
    # Adjust repulsion settings for better spacing
    net.repulsion(node_distance=200, spring_length=200, spring_strength=0.05, damping=0.09)
    
    net.write_html(output_file)
    webbrowser.open(output_file)

# Example usage
files = [
    ('files\\data_preprocessing_template.py', ['MyClass', 'AnotherClass'], ['import_data', 'method1', 'function1', 'method2'], [('MyClass', 'method1'), ('AnotherClass', 'method2')], [('MyClass', 'method1', 'contains'), ('AnotherClass', 'method2', 'contains')]),
    ('files\\data_preprocessing_tools.py', ['YourMom', 'YourSister'], ['preprocess', 'steps', 'fat', 'smash'], [('YourMom', 'fat'), ('YourSister', 'smash')], [('YourMom', 'fat', 'contains'), ('YourSister', 'smash', 'contains')]),
    ('files\\sl\\simple_linear_regression.py', [], ['sl', 'thankyou'], [], []),
    ('files\\sl\\try\\trial.py', ['CodeEntityExtractor'], ['__init__', 'visit_ClassDef', 'visit_FunctionDef', 'extract_code_entities_and_relations', 'create_knowledge_graph', 'visualize_knowledge_graph'], [('CodeEntityExtractor', '__init__'), ('CodeEntityExtractor', 'visit_ClassDef'), ('CodeEntityExtractor', 'visit_FunctionDef')], [('CodeEntityExtractor', '__init__', 'contains'), ('CodeEntityExtractor', 'visit_ClassDef', 'contains'), ('CodeEntityExtractor', 'visit_FunctionDef', 'contains')])
]

G = nx.DiGraph()
G = create_knowledge_graph(files, G)
visualize_knowledge_graph(G, output_file='knowledge_graph.html')

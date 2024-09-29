import ast
import networkx as nx
from pyvis.network import Network
import webbrowser
import os


class CodeEntityExtractor(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []
        self.methods = []
        self.relations = []

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self.methods.append((node.name, item.name))
                self.relations.append((node.name, item.name, "contains"))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not hasattr(node, 'classname'):  # To differentiate between standalone functions and methods
            self.functions.append(node.name)
        self.generic_visit(node)

def extract_code_entities_and_relations(code, filename):
    tree = ast.parse(code)
    extractor = CodeEntityExtractor()
    extractor.visit(tree)
    classes, functions, methods, relations = extractor.classes, extractor.functions, extractor.methods, extractor.relations
    return filename, classes, functions, methods, relations

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
    net.repulsion(node_distance=235, spring_length=130, spring_strength=0.5,)
    net.write_html(f'docs/{output_file}')
    webbrowser.open(f'docs/{output_file}')

if __name__ == "__main__":
    sample_code = """
class MyClass:
    def method1(self):
        pass

def function1():
    pass

class AnotherClass:
    def method2(self):
        pass
    """
    filename = "sample_code.py"
    # classes, functions, methods, relations = extract_code_entities_and_relations(sample_code)
    
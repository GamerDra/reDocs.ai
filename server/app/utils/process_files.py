from transformers import BertTokenizer, BertModel

from logic.create_embeddings import embedding

from logic.infinite_gpt import process_chunks

import ast
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token
from radon.visitors import ComplexityVisitor

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

embedding_List = []

for_gpt = []



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

# def extract_code_elements(code: str):
#     """Extract functions, classes, and methods from code using Radon."""
#     visitor = ComplexityVisitor.from_code(code)
#     functions = []
#     classes = []
#     methods = []

#     for block in visitor.functions:
#         if block.classname:  # Check if it is a method
#             methods.append((block.classname, block.name))
#         else:
#             functions.append(block.name)

#     for block in visitor.classes:
#         classes.append(block.name)
#         # Extract methods from the class
#         for method in block.methods:
#             methods.append((block.name, method.name))

#     return {
#         'functions': functions,
#         'classes': classes,
#         'methods': methods
#     }

def process_file(file_path, G, analyze=False,):
    '''Processes a file to generate a prompt, its embedding, and code analytics.'''
    with open(file_path, 'r', encoding='utf-8') as file:
        filename = file_path.split("/")[-1]
        # G.add_node(filename, label=filename, title=f'File: {filename}', color='#FFA500')
        prompt_text = file.read()
        for_gpt.append(prompt_text)
        # Process the file here
        print(prompt_text)
        result = embedding(prompt_text)

        # embedding_List.append(result)
        # print(result)
        # Process the file here
        # print(prompt_text)
    if (not analyze):
        return (prompt_text, result)
    
    analytics = extract_code_entities_and_relations(prompt_text, filename)
    
    return (prompt_text, result, analytics)

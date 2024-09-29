## Table of Contents

1. [**Understanding the Codebase**](#1-understanding-the-codebase)
   - [Server Side](#server-side)
     - [Logic and API](#logic-and-api)
     - [Utils](#utils)
   - [Frontend](#frontend)

2. [**Machine Learning Standpoint**](#2-machine-learning-standpoint)
   - [Codebase Traversal](#codebase-traversal)
   - [Code Embeddings](#code-embeddings)
   - [Handling Large Code Files](#handling-large-code-files)
   - [Maintaining Context with Agglomerative Clustering](#maintaining-context-with-agglomerative-clustering)
   - [Efficient Documentation Generation](#efficient-documentation-generation)
   - [Novel Techniques Used](#novel-techniques-used)

3. [**Tasks**](#3-tasks)
   - [Dendrogram Feature Addition](#dendrogram-feature-addition)
   - [Code Analytics](#code-analytics)
   - [Knowledge Graphs Addition](#knowledge-graphs-addition)

4. [**Limitations and Core Changes**](#4-limitations-and-core-changes)
   - [Dependency on Pre-trained Models](#dependency-on-pre-trained-models)
   - [Static Code Analysis](#static-code-analysis)
   - [No Logging/Tracing](#no-loggingtracing)
   - [Limited User Customization](#limited-user-customization)
   - [Memory Management](#memory-management)
   - [Limited Error Handling](#limited-error-handling)
   - [Scalability and Memory Management](#scalability-and-memory-management)

5. [**Incomplete Work**](#incomplete-work)
     - [Investigating Embedding Mechanism](#investigating-embedding-mechanism)
     - [Investigating Context Specific Mechanisms](#investigating-context-specific-mechanisms)



## 1. Understanding the Codebase

### Server Side

- #### Logic and API:
  -  We have several functions under `logic`, including:
    - `create_embedding` which leverages a window function for context and cosine similarity.
    - `convert_embedding` reshapes the embeddings to be passed to the clustering function which utilizes Agglomerative Clustering.
    - `infinite_gpt` selects prompts depending on the scenario.
    - `main.py` hosts the web server
    - `routes.py` we have defined the routing of different addresses.
  #### Utils 
    - There are few helper modules
    - In [process_file.py](server/app/utils/process_files.py) there is `process_file` function which reads through file and gives embeddings of the input text.
    - Traversal: We [traverse](server/app/utils/traverse_file.py) all folders and files, passing a single clustered embedding to the AI, hosted on a local server using [FastAPI.](https://fastapi.tiangolo.com/)

### Frontend
  - Manages what should be uploaded and what should not.

## 2.   Machine Learning standpoint 
### Codebase Traversal

- **Process**: We use a 0breadth-first search (BFS)(https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/) to systematically explore all directories and files, ensuring all files are processed in a structured manner.
- **Techniques**:
  - **BFS Traversal**: Efficiently adapted to traverse file directories. [traverse.py](server/app/utils/traverse_file.py)
  - **File Filtering**: Excludes non-relevant files based on extensions and names, so our model is only given the files containing text
  - **Use of Queue**: we used queue to make sure all directories are corvered 
  - **Next step** Now we after we have gathered all the embeddings, we reshape it and then now we pad it with zeros to ensure uniform dimensions so we can pass it to the clustering algorithm, now we have the indices list.
Now with the indices list we can send prompts to the AI in chunks decided by the indices list

### Code Embeddings
- **Meaning**: Embeddings are easier for computers to understand as the words are represented as multi-dimensional arrays or matrices We generate embeddings for each code file using a pre-trained language model, such as BERT (Bidirectional Encoder Representations from Transformers). This step converts the code into a numerical representation that captures its semantic meaning.
- **Process**: Generate embeddings for each code file using a pre-trained BERT model.
- **Techniques**:
  - **Transformer Models**: BERT encapsulates both syntactic and semantic properties of code.
  - **Windowed Embedding Generation**: For large files, the code is split into overlapping windows to maintain contextual integrity. Codebert has token limit of 512 [tokens](https://accubits.com/open-source-program-synthesis-models-leaderboard/codebert/)
  
  ![window](https://github.com/user-attachments/assets/c5ecd095-6627-496c-aab5-55d0de8d97d4)


### Handling Large Code Files

- **Process**: We handle large code files by dividing them into smaller, manageable chunks. Each chunk is processed independently, and their embeddings are aggregated to represent the entire file.
- **Techniques**:
  - **Sliding Window Approach**: Ensures complete coverage of the code file. Overlapping windows help preserve the context across boundaries, which is crucial for understanding the code's overall structure. 
  - **Embedding Aggregation**: Techniques like averaging or concatenation are used to retain comprehensive context.

### Maintaining Context with Agglomerative Clustering

- **Process**: Similar code embeddings are grouped using agglomerative clustering.
- **Techniques**:
  - **Agglomerative Clustering**:
1. Computing distance between every pair of objects.
2. Using linkage to group objects into hierarchical cluster tree, based on the distance. Objects/clusters that are in close proximity are linked together using the linkage function.
3. Determining where to cut the hierarchical tree into clusters. This creates a partition of the data.
  - **Cosine Similarity**: We use cosine similarity as the distance metric for clustering, which is effective in high-dimensional spaces and helps in accurately measuring the similarity between code embeddings.
  - ![cosine](https://github.com/user-attachments/assets/22270ceb-69db-420c-9601-f9351defb34c)


### Efficient Documentation Generation

- **Process**: Once clustered, documentation is generated by sending code snippets to a language model like GPT-3.5.
- **Techniques**:
  - **Prompt Engineering**: Guides the language model to generate structured documentation.
  - **Batch Processing**: Processes clusters in single prompts to improve efficiency.
  - **Mock Responses**: Tests the system without real API costs.

### Novel Techniques used 
According to me the techinques used are

1. **Windowed Embedding Generation**: Preserves context across text for accurate embeddings and for the token limit of 512.
2. **Hierarchical Clustering for Context Maintenance**: Maintains hierarchical relationships within the codebase.
3. **Prompt Engineering for Structured Documentation**: Allows customization of documentation needs.

## 3. Tasks 

### **Dendrogram Feature Addition**: 
- We take the clustered data with distances, find the total number of nodes, and then using that we create the linkage matrix. The code is directly driven from documentation given on [scikit](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py)
- The Dendrograms are saved in the zip folder with documentation, when the document generation is done.

![dendrogram](https://github.com/user-attachments/assets/da43e91c-0a79-4611-aa32-09e80a651b90)


 - Reason for choosing: Since dendrograms are generally associated with hierarchical data, therefore Agglomerative clustering is a good choice for clustering
- **Limitations** The dendrograms only give files clustered, improvements would include dendrogram nodes of the file content according to functions or different code sections. -  Improvements could have been done by allowing dendrogram generation as a separate module to avoid document generation when not required.


### **Code Analytics**: 
- Code analytics: I have used Ast(abstract syntax trees) to derive the detailed code structure, functions classes methods; functions and methods are distinguished, and then these relations are in knowledge graph. [code analytics](server/app/utils/process_files.py)

![code_analytics](https://github.com/user-attachments/assets/d9651c98-5863-4502-8777-1455f6ac5c2a)





- There were few more ways to do this, a simple prompt would have made easy but with the help of modules we can save ourselves some tokens in the Api, and we can reuse this for the creation of knowledge graphs.
- **Limitations** this doesnt handle error (a simple try and except before calling the function) if the code in the file contains any error prone code, it will break for now.


![knowledgegraph _ss](https://github.com/user-attachments/assets/6b78401e-d585-4226-9789-67c7bfc03275)

  | Node Color | Meaning                   |
  |------------|---------------------------|
  | Yellow     | Folder                    |
  | Orange     | File                      |
  | Red        | Class                     |
  | Blue       | Method part of a Class    |
  | Green      | Function                  |

### **Knowledge Graphs Addition**:
  - For the [knowledge graph addition](server/app/utils/knowledge_graph.py), I used the non-clustered data as it was easier to plot, I used the pyvis module to [plot](server/app/knowledge_graph.html) the map using the analytics obtained from the previous section. It will also be saved with the output docs zip file available during the document generation.
  The knowledge graphs are interactive and give a proper insight into the code structure, they are fun to play around with.
  As mentioned in the assignment the next thing that could be done is to add code retrieval which enhances the ability to find code because we the structure is well defined.

**Improvements** could be using the clustering distance to use instead of fix length, but there were some formatting issues that need to be worked on, not many tutorials availaible for py network.

## 4. Limitations and Core changes

Limitations And Potential Solutions
1. **Dependency on Pre-trained Models**:
    - The application relies heavily on pre-trained models like BERT for embeddings and GPT-3.5 for documentation generation.
    - This dependency may limit the customization and accuracy of embeddings and documentation due to      model biases and lack of domain-specific knowledge.
    - As the problem statement suggested exploring more mutli modal approach can help.
2. **Static Code Analysis**
-  Static analysis does not capture dynamic behavior, such as runtime dependencies and interactions between modules during execution.
-  Dynamic and Static Analysis Integration: Integrate dynamic analysis techniques, such as profiling, to capture runtime behavior in addition to static analysis. This provides comprehensive insights into the codebase and enhances documentation with runtime information.
This also allows to improve performance of the code, 
3. **No logging/ tracing**
  - Tracing involves recording the sequence of executed instructions or function calls in a program. It provides a detailed log of the program’s execution flow, including the order of executed functions, their entry and   exit points, and the values of variables at different points in time.
  - logger module in python can help.
4. **Limited User customisation**
  - We can add a checkbox type UI to select what the user wants, want generate documentation using this.
5. **Memory Management**
- The current approach of storing and processing embeddings for numerous files consumes significant memory, particularly due to padding embeddings to the same size.
- Efficient Memory Management and Streaming Processing: Use efficient data structures (e.g., sparse matrices) and techniques like dimensionality reduction (e.g., PCA )
6. **Limited Error Handling**
- The current system may not handle edge cases well, such as corrupted files, unsupported languages, or incomplete code snippets, potentially leading to failures or suboptimal documentation.
- Enhanced Error Handling and Robustness: Implement comprehensive validation and error handling mechanisms

7. **Scalability and Memory Management**
- The current approach of storing and processing embeddings for numerous files  consumes significant memory
- Techniques like dimensionality reduction(PCA, KCA) to  store embeddings, reducing memory usage.
- Implement streaming processing for embedding generation and processing to minimize memory usage by processing data in a [streaming](https://hazelcast.com/glossary/stream-processing/) fashion.
- Instead of BFS, we can maybe use some parallel processing techniques. 


## **Incomplete work **
#### Investigating Embedding Mechanism
- BERT Variants: Explore models like RoBERTa, ALBERT, and DistilBERT which are variations of BERT as mentioned in the [paper](https://arxiv.org/pdf/2002.08155)
- Codex: Specifically designed for coding tasks, Codex powers GitHub Copilot and can provide context-aware code suggestions and documentation.
- GraphCodeBERT: Extends CodeBERT by incorporating data-flow information, which can improve understanding and generation of code structures.

#### Investigating Context Specific Mechanisms

##### Memory network
-  Memory Networks (MemNets) are neural networks that have a memory component which they can read from and write to
- allows long term memorization, as told in the sequence modelling [video](https://www.youtube.com/watch?v=S7oA5C43Rbc), helps the model to know when to use is / are "the dogs ... (long sentence)" 








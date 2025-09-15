## 1) Data Ingestion with Document Loaders (LangChain)

This introductory step demonstrates basic **data ingestion techniques**: loading raw datasets from different sources and converting them into Document objects that LangChain can process.

- #### Text files
Use TextLoader to load a .txt file into a list of Document objects. Each document stores the file content in the attribute page_content.

- #### PDF files
Use PyPDFLoader to load a PDF. Each page of the PDF is parsed into a separate Document, making it easier to chunk and process.

- #### Web pages
Use WebBaseLoader to fetch and parse a web page into Document objects. You can also pass BeautifulSoup filters (bs_kwargs) to restrict which parts of the HTML are parsed, for example only titles, headers, or article content.

- #### Output
Every loader returns a list of Document objects with two main components:
- *page_content*: the extracted text
- *metadata*: additional information such as page number (PDF), source URL (web), or file path (text)

<br><br>

## 2) Data Transformation

**Text splitting** is a fundamental step in many Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) pipelines. Large documents (such as PDFs, long articles, or JSON schemas) are often too big to be processed directly by language models. Splitting them into smaller ***chunks*** makes it possible to:
- feed the text into models that have context length limits,
- preserve semantic meaning while keeping inputs compact,
- allow efficient storage in vector databases for semantic search.
A good splitter also provides overlap between chunks to maintain continuity of meaning across sections.

#### RecursiveCharacterTextSplitter
This splitter divides text into chunks of a defined maximum length (*chunk_size*). Each chunk can overlap with the previous one (*chunk_overlap*) to preserve context. 
Example: a chunk_size of 500 with overlap of 50 ensures each new segment carries 50 characters from the previous one. This approach is robust because it tries to respect logical text boundaries (sentences, paragraphs) when splitting.
Demonstrated on:
- a PDF file (“Attention is all you need”)
- a TXT file (“AI_and_society.txt”) where overlap is shown explicitly.
<br><br>
#### CharacterTextSplitter
This splitter uses a specific separator (for example “\n\n” for paragraph breaks). If the separator is missing, chunks may exceed the expected size, because the splitter prefers not to cut in the middle of a block. Two modes:
- **Variant 1**: split_documents takes LangChain Document objects (from TextLoader).
- **Variant 2**: create_documents works on raw strings, converting them into Documents afterwards.
<br><br>
#### RecursiveJsonSplitter
When working with nested JSON (for instance, API schemas), this splitter traverses the hierarchy and creates chunks respecting the structure. Options include:
- *split_json*: low-level structural chunks,
- *create_documents*: returns LangChain Document objects with page_content and metadata,
- *split_text*: outputs plain text chunks without metadata.
This is particularly useful for large schemas or deeply nested JSON documents.
<br><br>
#### Output format
Every splitter that produces LangChain Documents returns objects with two main attributes:
- page_content: the actual text content,
- metadata: context information such as file path, page number, or JSON keys.

<br><br>

## 3) Embedding Techniques

**Embeddings** are numerical vector representations of text. They map words, sentences, or entire documents into high-dimensional spaces where semantic similarity can be measured. In RAG pipelines, embeddings are essential because they allow you to store documents in a vector database and later retrieve the most relevant chunks given a query.

- **OllamaEmbeddings** is a wrapper around local embedding models available via Ollama. By default it uses llama2, but you can specify another model (e.g., gemma:2b, mxbai-embed-large) if it is already downloaded locally.

- **HuggingFaceEmbeddings** : we can generate vector representations of text using pre-trained models hosted on Hugging Face. In the example, the model all-MiniLM-L6-v2 converts a short text into a dense embedding vector. These embeddings can be used for semantic search, clustering, or as input for a RAG pipeline.

<br><br>

## 4) Vector Store Techniques
Vector store allow us to store and query embeddinga efficiently. Two backends are demonstrated:

- **FAISS**: after loading and splitting documents, embeddings are generated with *mxbai-embed-large*. These embeddings are indexed in a FAISS database, which supports fast similarity search. Queries can be made directly with *similarity_search*, by converting the DB into a retriever, or with *similarity_search_with_score* to get similarity scores. FAISS also supports saving and reloading the index.

- **CHROMA DB**: Similar workflow: documents are loaded, split and embedded with *mxbai-embed-large*. Chroma creates a persistent vector store that supports similarity search queries. It can be saved and reloaded for later use.
# GraphRAG Python package - From PDF to Q&A 
# This code:
# - Ingest PDFs into neo4j,
# - chunk PDFs into smaller pieces,
# - save each of the chunks in neo4j,
# - perform entity and relation extraction for each chunk and save them in the graph

# !pip install python-dotenv neo4j-graphrag openai

## Setup
# Define our variables:
# - Neo4j credentials,
# - List of files to be processed,
# - List of entities and relationships we are interested in and we will ask the LLM to find for us,
# - The LLM and embedder we want to use: OpenAI for this demo, but others are supported (VertexAI, MistralAI, Anthropic...),
# (note: OPENAI_API_KEY must be defined in the env vars),
# - We also decide to use a custom prompt for entity and relation extraction (instead of the default one), so it is also defined below.

import os
from dotenv import load_dotenv

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
import neo4j
# from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# load neo4j credentials (and openai api key in background)
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

FILE_PATHS = []

# define node labels
basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]
academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal", "InternalStudy"]

forensic_node_labels = ["Anatomy", "BiologicalProcess", "Cell", "CellularComponent", "CellType",
                        "Condition", "Disease", "Drug","EffectOrPhenotype", "Exposure", "GeneOrProtein", "Molecule",
                        "MolecularFunction", "Pathway", "ExtractionKit", "AmplificationKit", "CollectionMethod"]

node_labels = basic_node_labels + academic_node_labels + forensic_node_labels

# define relationship types
rel_types = ["ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED", "BIOMARKER_FOR", 
             "CAUSES", "CITES", "CONTRIBUTES_TO", "DESCRIBES", "EXPRESSES", "EXTRACTED_WITH", "AMPLIFIED_WITH",
             "HAS_REACTION", "HAS_SYMPTOM", "INCLUDES", "INTERACTS_WITH", "PRESCRIBED",
             "PRODUCES", "RECEIVED", "RESULTS_IN", "TREATS", "USED_FOR"]



# create text embedder (for chunk text)
embedder = OpenAIEmbeddings()

# create a llm object (for entity and relation extraction)
llm = OpenAILLM(
    model_name="gpt-4o-mini",\
    model_params={
        "response_format": {"type": "json_object"}, # use json_object formatting for best results
        "temperature": 0 # turning temperature down for more deterministic results
    }
    )


# optional: define your own prompt template for entity/relation extraction
# it must have 'text' placeholder and can use the 'schema' key

prompt_template = '''
    "You are a medical researcher tasks with extracting information from papers \n",
    "and structuring it in a property graph to inform further medical and research Q&A.\n",
    "\n",
    "Extract the entities (nodes) and specify their type from the following Input text.\n",
    "Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. \n",
    "\n",
    "\n",
    "Return result as JSON using the following format:\n",
    "{{\"nodes\": [ {{\"id\": \"0\", \"label\": \"the type of entity\", \"properties\": {{\"name\": \"name of entity\" }} }}],\n",
    "  \"relationships\": [{{\"type\": \"TYPE_OF_RELATIONSHIP\", \"start_node_id\": \"0\", \"end_node_id\": \"1\", \"properties\": {{\"details\": \"Description of the relationship\"}} }}] }}\n",
    "\n",
    "- Use only the information from the Input text.  Do not add any additional information.  \n",
    "- If the input text is empty, return empty Json. \n",
    "- Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.\n",
    "- An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. \n",
    "- Multiple documents will be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general. \n",
    "\n",
    "Use only fhe following nodes and relationships (if provided):\n",
    "{schema}\n",
    "\n",
    "Assign a unique ID (string) to each node, and reuse it to define relationships.\n",
    "Do respect the source and target node types for relationship and\n",
    "the relationship direction.\n",
    "\n",
    "Do not return any additional information other than the JSON in it.\n",
    "\n",
    "Input text:\n",
    "\n",
    "{text}\n",
'''


## Knowledge Graph Building - create our Neo4j driver and `SimpleKGPipeline` and run the pipeline on the list of documents
driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE)

kg_builder_pdf = SimpleKGPipeline(
    driver=driver,
    llm=llm,
    # text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
    embedder=embedder,
    entities=node_labels,
    relations=rel_types,
    prompt_template=prompt_template,
    from_pdf=True
    )
       
for path in FILE_PATHS:
    print(f"Processing : {path}")
    pdf_result = await kg_builder_pdf.run_async(file_path=path)
    print(f"PDF Processing Result: {pdf_result}")
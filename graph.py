from langchain.schema import Document
import cassio
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
import os
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from typing import Optional, List
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START, MessagesState
from pydantic import BaseModel, Field
from astrapy import DataAPIClient
from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt
from uipath import UiPath
from dotenv import load_dotenv

load_dotenv()
sdk = UiPath()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_URL =os.getenv("ASTRA_DB_URL")
groq_api_key =os.getenv("groq_api_key")

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)
print('connected to cassio')

# Get an existing collection
client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(ASTRA_DB_URL)
collection = database.get_collection("ragdemo")

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )
os.environ["GROQ_API_KEY"]=groq_api_key
llm=ChatGroq(groq_api_key=groq_api_key,model_name="meta-llama/llama-4-scout-17b-16e-instruct")
structured_llm_router = llm.with_structured_output(RouteQuery)
print('llm connected')


### Router
# Prompt
system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)



## Graph
class GraphState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation (optional)
        documents: list of documents (optional)
    """

    question: str
    generation: Optional[str] = None
    documents: Optional[str] = None 

def retrieve(state:GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state.question

    # Retrieval
    documents = collection.find_one({}, sort={"$vectorize": question})
    # documents = collection.invoke(question)
    return Command(goto="llm_invoke", update={"documents": str(documents), "question": question})

def wiki_search(state:GraphState):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    question = state.question

    # Wiki search
    docs = wiki.invoke({"query": question})
    #print(docs["summary"])
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return  Command(goto="llm_invoke", update={"documents": str(wiki_results), "question": question})


### Edges ###


def route_question(state:GraphState):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state.question
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return  'wiki_search'
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return 'retrieve'
    
llm=ChatGroq(groq_api_key=groq_api_key,model_name="meta-llama/llama-4-scout-17b-16e-instruct")
system = """Given quetion and context give answer of the question from context, if answer is not in context say - answer is not in document"""
llm_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Context: {context}\n question: {question}\n"),
    ]
)
llm_chain = llm_prompt | llm | StrOutputParser()

def llm_invoke(state:GraphState):
    question = state.question 
    context = state.documents
    output = llm_chain.invoke({"question": question, 'context': context})
    # state['generation'] = output
    return GraphOutput(answer=output)

class GraphInput(BaseModel):
    question: str

class GraphOutput(BaseModel):
    answer: str

def input(state: GraphInput):
    return GraphState( question= state.question)


# workflow = StateGraph(GraphState)
workflow = StateGraph(GraphState,input=GraphInput, output=GraphOutput)
# Define the nodes
workflow.add_node("input", input)
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node('llm_invoke',llm_invoke)
# Build graph

workflow.add_edge(START, "input")
workflow.add_conditional_edges("input",route_question,{
        "wiki_search": "wiki_search",
        "retrieve": "retrieve",
    },)
workflow.add_edge( "retrieve", 'llm_invoke')
workflow.add_edge( "wiki_search", 'llm_invoke')
workflow.add_edge( "llm_invoke", END)
# Compile
app = workflow.compile()
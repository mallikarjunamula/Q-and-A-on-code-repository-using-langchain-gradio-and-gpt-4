import gradio as gr
from git import Repo
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("openai_api_key")

# Clone
repo_path = "F:/Upwork_Projects/QA_on_codebase/test_repo"
# repo = Repo.clone_from("https://github.com/gventuri/pandas-ai", to_path=repo_path)
# Load
loader = GenericLoader.from_filesystem(
    repo_path + "/pandasai",
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
len(documents)
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
# len(texts)
db = Chroma.from_documents(texts, OpenAIEmbeddings(model="text-embedding-3-large",disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key)
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

def qa_code(question):
  result = qa(question)
  return result["answer"]

qacode = gr.Interface(
    qa_code,
    [
      gr.Textbox(label="Query", value="How to load excel/.xlsx files to pandas?"),
    ],
    "textbox",
    title="Q&A on Codebase(Pandas github repository) using Langchain and OpenAI's GPT-4",
    theme = "gradio/monochrome"
)
qacode.launch()
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
import os

# persist index
from app.loaders.file import load_local_data

def get_storage_context(persist_dir: str) -> StorageContext:
    return StorageContext.from_defaults(persist_dir=persist_dir)

def indexing():
    documents = load_local_data()

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="app/storage")

    return index

# from llama_index.core import StorageContext, load_index_from_storage

# # rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")

# # load index
# index = load_index_from_storage(storage_context)
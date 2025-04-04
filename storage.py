import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.retrievers import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class AsyncAPIFAISSRetriever:
    def __init__(self, knowledge_base_path: str):
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        
        # Initialize vector stores
        self.vectorstore, self.docstore = self._create_vector_stores()
        
        # Create retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key="doc_id"
        )
    
    def _load_knowledge_base(self, path: str) -> List[Dict]:
        """Load preprocessed knowledge base from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_vector_stores(self):
        """Create FAISS vector store and document store with support for YAML and images"""
        # Prepare documents
        documents = []
        metadatas = []
        ids = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        for doc in self.knowledge_base:
            # Split document content
            splits = text_splitter.split_text(doc['content'])
            for i, split in enumerate(splits):
                doc_id = f"{doc['id']}_{i}"
                documents.append(split)
                metadatas.append({
                    'type': 'documentation',
                    'title': doc['title'],
                    'source': doc['path'],
                    'doc_id': doc['id'],
                    'chunk_id': i,
                    'has_images': len(doc['images']) > 0
                })
                ids.append(doc_id)
            
            # Add code blocks
            for i, code_block in enumerate(doc['code_blocks']):
                code_id = f"code_{doc['id']}_{i}"
                documents.append(code_block)
                metadatas.append({
                    'type': 'code',
                    'language': 'javascript',  # AsyncAPI docs primarily use JS examples
                    'title': doc['title'],
                    'source': doc['path'],
                    'doc_id': doc['id'],
                    'code_block_id': i
                })
                ids.append(code_id)
            
            # Add YAML blocks
            for i, yaml_block in enumerate(doc['yaml_blocks']):
                yaml_id = f"yaml_{doc['id']}_{i}"
                documents.append(yaml_block)
                metadatas.append({
                    'type': 'code',
                    'language': 'yaml',
                    'title': doc['title'],
                    'source': doc['path'],
                    'doc_id': doc['id'],
                    'yaml_block_id': i
                })
                ids.append(yaml_id)
            
            # Add image references (we don't embed images, but store their metadata)
            for i, image in enumerate(doc['images']):
                image_id = f"image_{doc['id']}_{i}"
                # Create a text representation of the image for embedding
                image_description = f"Image from {doc['title']}: {image['alt'] or 'Diagram or screenshot'}"
                documents.append(image_description)
                metadatas.append({
                    'type': 'image',
                    'title': doc['title'],
                    'source': doc['path'],
                    'doc_id': doc['id'],
                    'image_url': image['src'],
                    'alt_text': image['alt'],
                    'image_id': i
                })
                ids.append(image_id)
        
        # Create FAISS vector store
        vectorstore = FAISS.from_texts(
            texts=documents,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Create document store
        docstore = InMemoryStore()
        docstore.mset([(doc['id'], doc) for doc in self.knowledge_base])
        
        return vectorstore, docstore
    
    def query(
        self,
        question: str,
        result_types: Optional[List[str]] = None,
        top_k: int = 3,
        include_source_docs: bool = True
    ):
        """
        Execute a query with flexible filtering options
        
        Args:
            question: The query string
            result_types: List of types to include ('documentation', 'code', 'yaml', 'image')
                        If None, includes all types
            top_k: Number of results to return per type if result_types is specified,
                   or total results if result_types is None
            include_source_docs: Whether to include full source documents in response
        """
        if result_types is None:
            # Return mixed results with no filtering
            docs = self.vectorstore.similarity_search(question, k=top_k)
        else:
            # Get results for each specified type
            docs = []
            for result_type in result_types:
                type_docs = self.vectorstore.similarity_search(
                    question,
                    k=top_k,
                    filter=lambda meta: meta.get('type') == result_type or 
                                     (result_type == 'code' and meta.get('language') in ['yaml', 'javascript'])
                )
                docs.extend(type_docs)
        
        # Convert to LangChain Document objects
        lc_docs = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in docs
        ]
        
        # Get full documents for context if requested
        full_docs = []
        if include_source_docs:
            doc_ids = list(set([doc.metadata['doc_id'] for doc in docs]))
            full_docs = [doc for doc in self.docstore.mget(doc_ids) if doc is not None]
        
        return {
            "relevant_chunks": lc_docs,
            "source_documents": full_docs if include_source_docs else []
        }
    
    def get_code_examples(self, doc_id: str) -> Dict:
        """Get all code examples from a specific document"""
        doc = self.docstore.get(doc_id)
        if not doc:
            return {}
        
        return {
            'javascript': doc.get('code_blocks', []),
            'yaml': doc.get('yaml_blocks', [])
        }
    
    def get_images(self, doc_id: str) -> List[Dict]:
        """Get all images from a specific document"""
        doc = self.docstore.get(doc_id)
        if not doc:
            return []
        
        return doc.get('images', [])
    
    def save_index(self, path: str):
        """Save FAISS index to disk"""
        self.vectorstore.save_local(path)
    
    @classmethod
    def load_index(cls, path: str, knowledge_base_path: str):
        """Load FAISS index from disk"""
        retriever = cls(knowledge_base_path)
        retriever.vectorstore = FAISS.load_local(
            path,
            retriever.embeddings,
            allow_dangerous_deserialization=True
        )
        return retriever

def main():
    # Initialize retriever
    knowledge_base_path = 'processed_asyncapi_docs/asyncapi_knowledge_base.json'
    index_path = "asyncapi_faiss_index"
    
    print("Creating AsyncAPI FAISS retriever...")
    retriever = AsyncAPIFAISSRetriever(knowledge_base_path)
    
    print(f"Saving index to {index_path}...")
    retriever.save_index(index_path)
    
    print("Retriever created and index saved successfully!")
    return retriever

if __name__ == "__main__":
    retriever = main()
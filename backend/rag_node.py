import os
import re
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports - these help us work with AI and documents
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import


# Load API keys from .env file
load_dotenv()


def sanitize_filename(name):
    """Sanitize filename for use as directory name (Windows-safe)"""
    # Replace special characters with underscores
    sanitized = re.sub(r'[–—]', '-', name)  # Replace en-dash/em-dash with hyphen
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)  # Replace invalid Windows chars
    sanitized = re.sub(r'\s+', '_', sanitized)  # Replace whitespace with underscore
    sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
    sanitized = sanitized.strip('_')  # Remove leading/trailing underscores
    return sanitized


class SimpleRAG:

    def __init__(self, docs_folder="./docs", index_folder="./faiss_indexes"):
        """
        Initialize the RAG system with separate indexes for each document

        Args:
            docs_folder: Folder containing your document files
            index_folder: Folder to save/load FAISS indexes
        """
        self.docs_folder = Path(docs_folder)
        self.index_folder = Path(index_folder)
        self.it_docs_folder = self.docs_folder / "IT Support"  # IT Support subfolder
        self.vector_stores = {}  # Dictionary to store separate indexes for each document
        self.embeddings = None

        # Define document categories
        self.hr_documents = ["Leave Policy.pdf", "HR_Policy_Art_Technology.pdf"]
        # IT documents are now DOCX files in the IT Support subfolder
        self.it_documents = [
            "Teams Login Errors 1.docx",
            "Hardware issues Keyboard and Mouse Touchpad 1.docx",
            "Hardware issues -Camera, Mic and Headset 1.docx",
            "Screenshare issue on ubuntu  1.docx",
            "URL not working  1 (1).docx",
            "SharePoint – Permission, File Sharing & Access Issues.docx",
            "OneDrive sync issues  1.docx",
            "System freezing issue .docx",
            "Outlook Application or mail sync issue.docx",
            "Windows VM connectivity issues 1.docx",
        ]

        # Create index folder if it doesn't exist
        self.index_folder.mkdir(exist_ok=True)

    def setup(self, verbose=True, force_rebuild=False):
        """
        Setup the RAG system - load existing indexes or create new ones

        Args:
            verbose: Print progress messages
            force_rebuild: Force rebuild indexes even if they exist
        """
        if verbose:
            print("\n" + "="*60)
            print("STEP 1: Checking for Documents")
            print("="*60)

        # Check if folder exists
        if not self.docs_folder.exists():
            raise ValueError(f"Folder '{self.docs_folder}' not found!")

        # Find PDF files in main docs folder (for HR)
        pdf_files = list(self.docs_folder.glob("*.pdf"))

        # Find DOCX files in IT Support subfolder
        docx_files = []
        if self.it_docs_folder.exists():
            docx_files = list(self.it_docs_folder.glob("*.docx"))

        # Combine all document files
        all_doc_files = pdf_files + docx_files

        if not all_doc_files:
            raise ValueError(f"No document files found!")

        if verbose:
            print(f"Found {len(pdf_files)} PDF files in docs folder:")
            for pdf in pdf_files:
                print(f"  - {pdf.name}")
            print(f"\nFound {len(docx_files)} DOCX files in IT Support folder:")
            for docx in docx_files:
                print(f"  - {docx.name}")

        # Initialize embeddings (shared across all indexes)
        if verbose:
            print("\n" + "="*60)
            print("STEP 2: Initializing Embeddings Model")
            print("="*60)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if verbose:
            print("[OK] Embeddings model loaded")

        # Check if we can load existing indexes for all documents
        all_indexes_exist = True
        for doc_file in all_doc_files:
            index_name = sanitize_filename(doc_file.stem)  # Sanitized filename
            index_path = self.index_folder / index_name
            if not index_path.exists():
                all_indexes_exist = False
                break

        # Load existing indexes if available and not forcing rebuild
        if all_indexes_exist and not force_rebuild:
            if verbose:
                print("\n" + "="*60)
                print("STEP 3: Loading Existing Indexes")
                print("="*60)

            for doc_file in all_doc_files:
                index_name = sanitize_filename(doc_file.stem)
                index_path = self.index_folder / index_name

                if verbose:
                    print(f"\nLoading: {doc_file.name}")

                # Load FAISS index from disk
                vector_store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # Store with document name as key
                self.vector_stores[doc_file.name] = vector_store

                if verbose:
                    print(f"  [OK] Index loaded for {doc_file.name}")

            if verbose:
                print("\n" + "="*60)
                print(f"[OK] Loaded {len(self.vector_stores)} existing indexes")
                print("="*60)

        else:
            # Create new indexes
            if verbose:
                print("\n" + "="*60)
                print("STEP 3: Creating New Indexes")
                print("="*60)

            for doc_file in all_doc_files:
                if verbose:
                    print(f"\nProcessing: {doc_file.name}")

                # Use appropriate loader based on file type
                if doc_file.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(doc_file))
                elif doc_file.suffix.lower() == ".docx":
                    loader = Docx2txtLoader(str(doc_file))
                else:
                    if verbose:
                        print(f"  [SKIP] Unsupported file type: {doc_file.suffix}")
                    continue

                pages = loader.load()

                # Add filename to each page/document
                for page in pages:
                    page.metadata["source"] = doc_file.name

                if verbose:
                    print(f"  - Loaded {len(pages)} pages/sections")

                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                )
                chunks = text_splitter.split_documents(pages)

                if verbose:
                    print(f"  - Created {len(chunks)} chunks")

                # Create FAISS vector store for this document
                vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )

                # Save index to disk
                index_name = sanitize_filename(doc_file.stem)
                index_path = self.index_folder / index_name
                index_path.mkdir(exist_ok=True)  # Create directory for index
                vector_store.save_local(str(index_path))

                # Store with document name as key
                self.vector_stores[doc_file.name] = vector_store

                if verbose:
                    print(f"  [OK] Index created and saved for {doc_file.name}")

            if verbose:
                print("\n" + "="*60)
                print(f"[OK] Created and saved {len(self.vector_stores)} indexes")
                print("="*60)

    def find_matching_documents(self, query: str, doc_list: list) -> list:
        """
        Find documents whose names match keywords in the query.
        Uses keyword overlap to filter documents before semantic search.

        Args:
            query: User's search query
            doc_list: List of document filenames to filter

        Returns:
            List of matching document names, or all docs if no match found
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        matched_docs = []
        match_scores = []

        for doc_name in doc_list:
            # Extract keywords from document name
            # Remove extension, numbers, and special characters
            doc_name_clean = re.sub(r'\.(pdf|docx|doc)$', '', doc_name.lower())
            doc_name_clean = re.sub(r'[0-9\.\(\)\-\_]', ' ', doc_name_clean)
            doc_keywords = set(doc_name_clean.split())
            doc_keywords.discard('')  # Remove empty strings

            # Check for keyword overlap
            overlap = query_words.intersection(doc_keywords)

            # Also check if query contains significant part of document name
            # This handles cases like "screenshare issue" matching "screenshare issue on ubuntu"
            doc_name_in_query = doc_name_clean.strip() in query_lower

            if len(overlap) >= 2 or doc_name_in_query:
                matched_docs.append(doc_name)
                match_scores.append(len(overlap))

        if matched_docs:
            # Sort by match score (more overlapping keywords = better match)
            sorted_docs = [doc for _, doc in sorted(zip(match_scores, matched_docs), reverse=True)]
            print(f"[RAG Filter] Query: '{query}'")
            print(f"[RAG Filter] Matched documents: {sorted_docs}")
            return sorted_docs

        # No matches found - return empty list to let semantic search handle it
        # This prevents falling back to irrelevant documents
        print(f"[RAG Filter] Query: '{query}' - No keyword matches, returning empty")
        return []

    def search(self, question, num_results=3, doc_names=None):
        """
        Search for relevant document chunks across specified documents

        Args:
            question: Your question
            num_results: How many relevant chunks to find from EACH document
            doc_names: List of document filenames to search (if None, searches all)

        Returns:
            List of relevant text chunks with their sources
        """
        if not self.vector_stores:
            raise ValueError("Please run setup() first!")

        # Determine which documents to search
        if doc_names is None:
            search_docs = list(self.vector_stores.keys())
        else:
            search_docs = [doc for doc in doc_names if doc in self.vector_stores]

        if not search_docs:
            return []

        # Search each specified document and collect results
        all_results = []
        for doc_name in search_docs:
            vector_store = self.vector_stores[doc_name]
            results = vector_store.similarity_search(question, k=num_results)

            for doc in results:
                all_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown")
                })

        # Rank all results
        formatted_results = []
        for i, result in enumerate(all_results, 1):
            result["rank"] = i
            formatted_results.append(result)

        return formatted_results

    def search_with_scores(self, question, num_results=3, doc_names=None, score_threshold=1.0):
        """
        Search with similarity scores - returns only results above threshold.
        Uses FAISS similarity_search_with_score which returns (doc, score) tuples.
        Lower score = more similar in FAISS (L2 distance).

        Args:
            question: Your question
            num_results: How many relevant chunks to find from EACH document
            doc_names: List of document filenames to search (if None, searches all)
            score_threshold: Maximum L2 distance to consider relevant (lower = stricter)
                            Typical values: 0.5 (very strict), 1.0 (moderate), 1.5 (lenient)

        Returns:
            List of relevant text chunks with their sources and similarity scores
        """
        if not self.vector_stores:
            raise ValueError("Please run setup() first!")

        # Determine which documents to search
        if doc_names is None:
            search_docs = list(self.vector_stores.keys())
        else:
            search_docs = [doc for doc in doc_names if doc in self.vector_stores]

        if not search_docs:
            return []

        # Search each specified document and collect results with scores
        all_results = []
        for doc_name in search_docs:
            vector_store = self.vector_stores[doc_name]
            # similarity_search_with_score returns list of (Document, score) tuples
            results_with_scores = vector_store.similarity_search_with_score(question, k=num_results)

            for doc, score in results_with_scores:
                # Only include results below threshold (lower score = more similar)
                if score <= score_threshold:
                    all_results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "similarity_score": float(score)  # Lower is better
                    })

        # Sort by similarity score (ascending - lower is better)
        all_results.sort(key=lambda x: x['similarity_score'])

        # Rank all results
        formatted_results = []
        for i, result in enumerate(all_results, 1):
            result["rank"] = i
            formatted_results.append(result)

        return formatted_results

    def search_it_with_scores(self, question, num_results=3, score_threshold=1.0):
        """
        Search IT Support documents with similarity scores.
        Uses keyword pre-filtering first, then returns scored results.

        Args:
            question: Your question
            num_results: How many relevant chunks to find from each IT document
            score_threshold: Maximum L2 distance to consider relevant

        Returns:
            List of relevant text chunks from IT Support documents with scores
        """
        # First: Find documents matching query keywords
        matching_docs = self.find_matching_documents(question, self.it_documents)

        # If no keyword matches, search ALL IT documents with scoring
        # (but the score threshold will filter out irrelevant results)
        if not matching_docs:
            matching_docs = self.it_documents

        # Then: Search with scores
        return self.search_with_scores(question, num_results, doc_names=matching_docs, score_threshold=score_threshold)

    def search_hr_policies(self, question, num_results=3):
        """
        Search only HR policy documents (Leave Policy and HR_Policy_Art_Technology)
        Uses keyword pre-filtering to match relevant documents first.

        Args:
            question: Your question
            num_results: How many relevant chunks to find from each HR document

        Returns:
            List of relevant text chunks from HR documents
        """
        # First: Find documents matching query keywords
        matching_docs = self.find_matching_documents(question, self.hr_documents)

        # Then: Search only in matched documents
        return self.search(question, num_results, doc_names=matching_docs)

    def search_it_policies(self, question, num_results=3):
        """
        Search only IT Support documents (troubleshooting guides)
        Uses keyword pre-filtering to match relevant documents first.

        Args:
            question: Your question
            num_results: How many relevant chunks to find from each IT document

        Returns:
            List of relevant text chunks from IT Support documents
        """
        # First: Find documents matching query keywords
        matching_docs = self.find_matching_documents(question, self.it_documents)

        # Then: Search only in matched documents
        return self.search(question, num_results, doc_names=matching_docs)
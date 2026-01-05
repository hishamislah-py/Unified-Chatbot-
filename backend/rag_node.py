import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports - these help us work with AI and documents
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import


# Load API keys from .env file
load_dotenv()


class SimpleRAG:

    def __init__(self, docs_folder="./docs", index_folder="./faiss_indexes"):
        """
        Initialize the RAG system with separate indexes for each PDF

        Args:
            docs_folder: Folder containing your PDF files
            index_folder: Folder to save/load FAISS indexes
        """
        self.docs_folder = Path(docs_folder)
        self.index_folder = Path(index_folder)
        self.vector_stores = {}  # Dictionary to store separate indexes for each PDF
        self.embeddings = None

        # Define document categories
        self.hr_documents = ["Leave Policy.pdf", "HR_Policy_Art_Technology.pdf"]
        self.it_documents = ["IT_Security_Policy_AI_Usage.pdf", "Compliance Handbook.pdf"]

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
            print("STEP 1: Checking for Existing Indexes")
            print("="*60)

        # Check if folder exists
        if not self.docs_folder.exists():
            raise ValueError(f"Folder '{self.docs_folder}' not found!")

        # Find all PDF files
        pdf_files = list(self.docs_folder.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in '{self.docs_folder}'")

        if verbose:
            print(f"Found {len(pdf_files)} PDF files:")
            for pdf in pdf_files:
                print(f"  - {pdf.name}")

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

        # Check if we can load existing indexes
        all_indexes_exist = True
        for pdf_file in pdf_files:
            index_name = pdf_file.stem  # Filename without extension
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

            for pdf_file in pdf_files:
                index_name = pdf_file.stem
                index_path = self.index_folder / index_name

                if verbose:
                    print(f"\nLoading: {pdf_file.name}")

                # Load FAISS index from disk
                vector_store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # Store with PDF name as key
                self.vector_stores[pdf_file.name] = vector_store

                if verbose:
                    print(f"  [OK] Index loaded for {pdf_file.name}")

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

            for pdf_file in pdf_files:
                if verbose:
                    print(f"\nProcessing: {pdf_file.name}")

                # Load single PDF
                loader = PyPDFLoader(str(pdf_file))
                pages = loader.load()

                # Add filename to each page
                for page in pages:
                    page.metadata["source"] = pdf_file.name

                if verbose:
                    print(f"  - Loaded {len(pages)} pages")

                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                )
                chunks = text_splitter.split_documents(pages)

                if verbose:
                    print(f"  - Created {len(chunks)} chunks")

                # Create FAISS vector store for this PDF
                vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )

                # Save index to disk
                index_name = pdf_file.stem
                index_path = self.index_folder / index_name
                vector_store.save_local(str(index_path))

                # Store with PDF name as key
                self.vector_stores[pdf_file.name] = vector_store

                if verbose:
                    print(f"  [OK] Index created and saved for {pdf_file.name}")

            if verbose:
                print("\n" + "="*60)
                print(f"[OK] Created and saved {len(self.vector_stores)} indexes")
                print("="*60)

    def search(self, question, num_results=3, pdf_names=None):
        """
        Search for relevant document chunks across specified PDFs

        Args:
            question: Your question
            num_results: How many relevant chunks to find from EACH PDF
            pdf_names: List of PDF filenames to search (if None, searches all)

        Returns:
            List of relevant text chunks with their sources
        """
        if not self.vector_stores:
            raise ValueError("Please run setup() first!")

        # Determine which PDFs to search
        if pdf_names is None:
            search_pdfs = list(self.vector_stores.keys())
        else:
            search_pdfs = [pdf for pdf in pdf_names if pdf in self.vector_stores]

        if not search_pdfs:
            return []

        # Search each specified PDF and collect results
        all_results = []
        for pdf_name in search_pdfs:
            vector_store = self.vector_stores[pdf_name]
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

    def search_hr_policies(self, question, num_results=3):
        """
        Search only HR policy documents (Leave Policy and HR_Policy_Art_Technology)

        Args:
            question: Your question
            num_results: How many relevant chunks to find from each HR document

        Returns:
            List of relevant text chunks from HR documents
        """
        return self.search(question, num_results, pdf_names=self.hr_documents)

    def search_it_policies(self, question, num_results=3):
        """
        Search only IT policy documents (IT_Security_Policy_AI_Usage and Compliance Handbook)

        Args:
            question: Your question
            num_results: How many relevant chunks to find from each IT document

        Returns:
            List of relevant text chunks from IT documents
        """
        return self.search(question, num_results, pdf_names=self.it_documents)
# ====== IMPORT ======
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI
from dotenv import load_dotenv
import os


# ====== LOAD ENV ======
load_dotenv()

BASE_URL = os.getenv("OPENAI_ENDPOINT")
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")
GPT_KEY = os.getenv("GPT_KEY")


# ====== LOAD PDF ======
loader = PyPDFLoader("data.pdf")
documents = loader.load()


# ====== SPLIT TEXT ======
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)


# ====== EMBEDDING ======
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=EMBEDDING_KEY,
    openai_api_base=BASE_URL
)


# ====== VECTOR DB ======
if os.path.exists("chroma_db"):
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
else:
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db"
    )


# ====== GPT CLIENT ======
client = OpenAI(
    base_url=BASE_URL,
    api_key=GPT_KEY
)


# ====== TRANSLATE VN → EN ======
def translate_to_en(text):
    res = client.chat.completions.create(
        model="GPT-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Translate to English accurately (keep legal meaning): {text}"
            }
        ]
    )
    return res.choices[0].message.content.strip()


# ====== SEARCH ======
def search_context(query):
    query_en = translate_to_en(query)
    docs = db.similarity_search(query_en, k=12)
    return "\n\n".join([d.page_content for d in docs])


# ====== GPT ANSWER (FULL CONTENT MODE) ======
def ask_gpt(context, question):
    prompt = f"""
Bạn là chatbot.

Nhiệm vụ:
- Viết lại TOÀN BỘ nội dung trong context
- KHÔNG được tóm tắt
- KHÔNG được bỏ bất kỳ ý nào
- KHÔNG được suy diễn
- Phải giữ nguyên đầy đủ nội dung và logic
- Nếu là điều khoản dài → phải trình bày đầy đủ

Nếu có số liệu → PHẢI giữ chính xác

KHÔNG được trả lời "không biết" nếu context có dữ liệu

Context:
{context}

Câu hỏi: {question}
"""

    res = client.chat.completions.create(
        model="GPT-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return res.choices[0].message.content.strip()


# ====== MAIN ======
if __name__ == "__main__":
    while True:
        q = input("Nhập câu hỏi: ")

        context = search_context(q)
        answer = ask_gpt(context, q)

        print("\n--- Answer ---")
        print(answer)
        print()
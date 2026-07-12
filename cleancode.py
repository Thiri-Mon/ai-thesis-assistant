import os
import gc
import pandas as pd
import streamlit as st
from collections import Counter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# =========================================================================
# 🔴 PAGE SETUP (Must be the very first Streamlit command)
# =========================================================================
st.set_page_config(page_title="AI-Powered Thesis Assistant", layout="wide")
st.title("🌐 AI-Powered Thesis Assistant (SBERT + ChromaDB + Ollama)")

DB_DIR = "./chroma_db_storage"

# 🌟 SBERT Embedding Model Initialization
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}  # Change to False if model needs online download first
    )

embeddings = load_embedding_model()

# =========================================================================
# ၁။ Session State Initializer & Database Safe Loader
# =========================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "မင်္ဂလာပါ အစ်မ! Thesis ခေါင်းစဉ် Dataset (Excel/CSV/PDF) တင်ပြီး ဆင်တူရိုးမှား စစ်ဆေးနိုင်ပါပြီ။"
    }]

if "vector_store" not in st.session_state:
    if os.path.exists(DB_DIR):
        try:
            st.session_state.vector_store = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
        except Exception:
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

def is_database_empty():
    if st.session_state.vector_store is None:
        if os.path.exists(DB_DIR):
            try:
                st.session_state.vector_store = Chroma(
                    persist_directory=DB_DIR,
                    embedding_function=embeddings
                )
            except Exception:
                return True
        else:
            return True
            
    try:
        db_data = st.session_state.vector_store.get()
        if db_data and 'ids' in db_data and len(db_data['ids']) > 0:
            return False
        return True
    except Exception:
        return True

# =========================================================================
# ၂။ Sidebar Configurations & Robust Multi-Format Data Ingestion
# =========================================================================
st.sidebar.header("📁 Thesis Data Upload")

uploaded_files = st.sidebar.file_uploader(
    "Excel/CSV/PDF ဖိုင်များ တင်ရန်", 
    type=["xlsx", "xls", "csv", "pdf"], 
    accept_multiple_files=True
)

column_name_input = st.sidebar.text_input(
    "Thesis ခေါင်းစဉ်ပါသော Column နာမည်ကို ရိုက်ထည့်ပါ", 
    value="Thesis Title"
)

if st.sidebar.button("Database ထဲသို့ ထည့်သွင်းမည်"):
    if uploaded_files:
        with st.sidebar.spinner("ဒေတာများကို Vector Database ထဲသို့ ထည့်သွင်းနေပါသည်..."):
            documents = []
            
            for uploaded_file in uploaded_files:
                try:
                    # Case A: Handling Excel Workbooks
                    if uploaded_file.name.endswith(('.xlsx', '.xls')):
                        uploaded_file.seek(0)
                        xl = pd.ExcelFile(uploaded_file)
                        for sheet in xl.sheet_names:
                            uploaded_file.seek(0)
                            df = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)
                            
                            target_col_index = None
                            start_row_index = 1
                            found_header = False
                            search_term = column_name_input.strip().lower()
                            
                            for r_idx in range(min(len(df), 15)):
                                row_values = [str(val).strip().lower() for val in df.iloc[r_idx]]
                                if search_term and (search_term in row_values):
                                    target_col_index = row_values.index(search_term)
                                    start_row_index = r_idx + 1
                                    found_header = True
                                    break
                                for c_idx, val in enumerate(row_values):
                                    if any(kw in val for kw in ['title', 'thesis', 'ခေါင်းစဉ်', 'topic']):
                                        target_col_index = c_idx
                                        start_row_index = r_idx + 1
                                        found_header = True
                                        break
                                if found_header:
                                    break
                            
                            if target_col_index is None:
                                target_col_index = 3 if len(df.columns) > 3 else 0
                                start_row_index = 1
                            
                            if target_col_index < len(df.columns):
                                for index in range(start_row_index, len(df)):
                                    text_content = str(df.iloc[index, target_col_index]).strip()
                                    if text_content and text_content.lower() != "nan" and len(text_content) > 8:
                                        if text_content.lower() in ["thesis title", "title", "ခေါင်းစဉ်", "topic", "-", "no", "name", "roll no"]:
                                            continue
                                        meta_data = {"source": f"{uploaded_file.name} ({sheet})", "row": index}
                                        documents.append(Document(page_content=text_content, metadata=meta_data))
                    
                    # Case B: Handling Raw CSV Files
                    elif uploaded_file.name.endswith('.csv'):
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, header=None)
                        # Basic index assignment strategy for CSV fallback
                        target_col_index = 0
                        for index in range(0, len(df)):
                            text_content = str(df.iloc[index, target_col_index]).strip()
                            if text_content and text_content.lower() != "nan" and len(text_content) > 8:
                                meta_data = {"source": uploaded_file.name, "row": index}
                                documents.append(Document(page_content=text_content, metadata=meta_data))

                except Exception as e:
                    st.sidebar.error(f"Error parsing {uploaded_file.name}: {e}")
                    continue
                            
            if documents:
                if "vector_store" in st.session_state:
                    st.session_state.vector_store = None
                gc.collect()
                
                try:
                    import chromadb
                    persistent_client = chromadb.PersistentClient(path=DB_DIR)
                    for collection in persistent_client.list_collections():
                        persistent_client.delete_collection(collection.name)
                except Exception:
                    pass
                
                st.session_state.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=DB_DIR,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                st.sidebar.success(f"🎉 ဒေတာစုစုပေါင်း ({len(documents)}) ခု သိမ်းဆည်းပြီးပါပြီ!")
                st.rerun()
            else:
                st.sidebar.error("❌ ဖတ်ရှုရန် သင့်လျော်သော စာသားဒေတာ မတွေ့ရှိပါ။")
    else:
        st.sidebar.warning("⚠️ ကျေးဇူးပြု၍ ဖိုင်အရင် တင်ပေးပါဦး။")

# Sidebar Live Metrics Panel
if not is_database_empty():
    st.sidebar.success("✅ Knowledge Base Ready!")
    try:
        db_data = st.session_state.vector_store.get()
        total_records = len(db_data['ids'])
        st.sidebar.metric(label="စုစုပေါင်း Title အရေအတွက်", value=total_records)
        
        if 'metas' in db_data and db_data['metas']:
            sources = [meta.get('source', 'Unknown') for meta in db_data['metas'] if meta]
            file_counts = Counter(sources)
            summary_text = "".join([f"• {file_name}: **{count}** ခု\n" for file_name, count in file_counts.items()])
            st.sidebar.info(f"📊 **ဖိုင်အလိုက် ခွဲခြမ်းစိတ်ဖြာမှု:**\n{summary_text}")
    except Exception:
        pass

# =========================================================================
# ၃။ Persistent Chat Window Render (Fixes the Chat Disappearance Bug)
# =========================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =========================================================================
# ၄။ Runtime Analytical Engine (Similarity Verification & Generation)
# =========================================================================
if user_query := st.chat_input("စစ်ဆေးချင်သော Thesis ခေါင်းစဉ် ရိုက်ထည့်ပါ..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        query_lower = user_query.lower()
        is_thesis_check = any(kw in query_lower for kw in ["thesis", "title", "ခေါင်းစဉ်", "စစ်", "similar", "တူ", "တူလား", "ဆင်တူ"])
        
    if is_thesis_check and is_database_empty():
            warning_response = "⚠️ စနစ်ထဲတွင် နှိုင်းယှဉ်စစ်ဆေးစရာ Thesis Dataset (Knowledge Base) မရှိသေးပါ။ ကျေးဇူးပြု၍ ဘယ်ဘက် Sidebar တွင် Excel/CSV/PDF ဖိုင်တစ်ခုခု အရင် တင်ပေးပါဗျာ။"
            response_placeholder.write(warning_response)
            st.session_state.messages.append({"role": "assistant", "content": warning_response})
        
    else:
            context = ""
            is_duplicate_detected = False
            context_list = []
            
            # DB ရှိမှသာ Context ဆွဲထုတ်မယ်
            if not is_database_empty():
                # 💡 ကွက်တိ ရာခိုင်နှုန်း score တွက်ချက်ပြီး ရှာဖွေခြင်း
                docs_and_scores = st.session_state.vector_store.similarity_search_with_score(user_query, k=4)
                
                for doc, score in docs_and_scores:
                    # distance score က သုညနားနီးလေ တူလေမို့လို့ 0.4 ထက်ငယ်ရင် Duplicate လို့ သတ်မှတ်မယ်
                    if score < 0.4: 
                        is_duplicate_detected = True
                    
                    # ဖိုင်နာမည်ကနေ မေဂျာနာမည်ကို ယူခြင်း (.xlsx ဖြတ်ခြင်း)
                    raw_source = doc.metadata.get("source", "Unknown.xlsx")
                    # Clean string processing to split extension wrappers smoothly
                    clean_file = raw_source.split(" (")[0] if " (" in raw_source else raw_source
                    major_name = clean_file.replace(".xlsx", "").replace(".xls", "").upper()
                    
                    context_list.append(f"- Title: {doc.page_content} [Department/Major: {major_name}](Distance: {score:.3f})")
                
                context = "\n".join(context_list)
            
            # Chat History (Memory) တည်ဆောက်ခြင်း
            chat_history_str = ""
            for msg in st.session_state.messages[:-1]:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                chat_history_str += f"{role_label}: {msg['content']}\n"
            
            # System Prompt စည်းကမ်းသတ်မှတ်ချက်
            system_prompt = (
                f"You are an expert academic advisor in Computer Science and Information Technology.\n"
                f"Your primary task is to evaluate whether the user's proposed thesis title already exists in the database or if it is completely unique.\n\n"
                f"STATUS FROM DATABASE SEARCH:\n"
                f"- Is exact or near-exact title found?: {'YES, IT ALREADY EXISTS (DUPLICATE)' if is_duplicate_detected else 'NO, THIS TITLE IS UNIQUE / NEW.'}\n\n"
                f"CRITICAL RULES:\n"
                f"1. Read the 'STATUS FROM DATABASE SEARCH' above. If it says 'NO', congratulate the user and confirm that it is unique.\n"
                f"2. If it says 'YES', politely inform them, and MUST show the matching title(s) along with their [Department/Major] exactly as listed in the Context below so the user knows which major it belongs to.\n"
                f"3. Do NOT just repeat the major name. You must show both full 'Title' and 'Department/Major'.\n\n"
                f"Conversation History:\n{chat_history_str if chat_history_str else 'No previous conversation.'}\n\n"
                f"Context of closest titles from database:\n{context if context else 'EMPTY (No database uploaded or no matching data)'}"
            )
            
            messages_input = [
                ("system", system_prompt),
                ("user", user_query)
            ]
            
            llm = ChatOllama(model="llama3.2", temperature=0.3, streaming=True)

            ai_response = ""
            try:
                for chunk in llm.stream(messages_input):
                    ai_response += chunk.content
                    response_placeholder.write(ai_response + "▌")
            except Exception as e:
                st.error(f"Ollama Run ရာတွင် အမှားရှိနေပါသည်။ Error: {e}")
                st.stop()
            
            response_placeholder.write(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})


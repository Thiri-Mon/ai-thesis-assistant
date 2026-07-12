import streamlit as st
import tempfile
import pandas as pd  # Excel နှင့် CSV ဖတ်ရန်အတွက်
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="AI-Powered Thesis Assistant", layout="wide")
st.title("🌐 AI-Powered Thesis Assistant (SBERT + ChromaDB + Ollama)")


# 🌟 SBERT Embedding Model ကို Initialize လုပ်ခြင်း
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only":True} #False to be downloaded,if needed
    )
embeddings = load_embedding_model()

# =========================================================================
# ၁။ ChromaDB အလွတ် ဟုတ်/မဟုတ် စစ်ဆေးမည့် Function (အစ်မ ပို့ပေးတဲ့ စနစ်အသစ်)
# =========================================================================
def is_database_empty():
    # ၁။ session_state ထဲမှာ မရှိသေးရင် Folder ကို တိုက်ရိုက် သွားစစ်မယ်
    import os
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        if os.path.exists("./chroma_db_storage"):
            try:
                # Folder ရှိရင် ချက်ချင်း လှမ်းဖတ်ပြီး session ထဲ ထည့်လိုက်မယ်
                from langchain_community.vectorstores import Chroma
                st.session_state.vector_store = Chroma(
                    persist_directory="./chroma_db_storage",
                    embedding_function=embeddings
                )
            except Exception:
                return True
        else:
            return True
            
    # ၂။ ဒေတာ တကယ် ရှိ/မရှိ ChromaDB ထဲအထိ လှမ်းစစ်ခြင်း
    try:
        db_data = st.session_state.vector_store.get()
        if db_data and 'ids' in db_data and len(db_data['ids']) > 0:
            return False  # ဒေတာ တကယ်ရှိတယ်!
        return True       # ဒေတာ မရှိဘူး အလွတ်ကြီးပဲ
    except Exception:
        return True

# =========================================================================
# ၂။ Session State ခေါ်ယူခြင်း နှင့် ChromaDB Auto-Load စနစ်
# =========================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "မင်္ဂလာပါ အစ်မ! Thesis ခေါင်းစဉ် Dataset (Excel/CSV/PDF) တင်ပြီး ဆင်တူရိုးမှား စစ်ဆေးနိုင်ပါပြီ။"}]

# Page စပွင့်ကတည်းက စက်ထဲမှာ သိမ်းထားတဲ့ ChromaDB ရှိရင် အလိုအလျောက် ဆွဲတင်ခြင်း
if "vector_store" not in st.session_state:
    import os
    if os.path.exists("./chroma_db_storage"):
        try:
            st.session_state.vector_store = Chroma(
                persist_directory="./chroma_db_storage",
                embedding_function=embeddings
            )
        except Exception:
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None


# =========================================================================
# =========================================================================
# ၃။ Sidebar Configuration (ဖိုင်တင်ခြင်း နှင့် Database အသစ်သွင်းခြင်းအပိုင်း)
# =========================================================================
st.sidebar.header("📁 Thesis Data Upload")

# Multiple files တင်နိုင်အောင် သတ်မှတ်ထားခြင်း
uploaded_files = st.sidebar.file_uploader(
    "Excel/CSV/PDF ဖိုင်များ တင်ရန်", 
    type=["xlsx", "xls", "csv", "pdf"], 
    accept_multiple_files=True
)

# Column Name Input ယူခြင်း
column_name_input = st.sidebar.text_input(
    "Thesis ခေါင်းစဉ်ပါသော Column နာမည်ကို ရိုက်ထည့်ပါ", 
    value="Thesis Title"
)

# Database အသစ်ဆောက်မည့် ခလုတ်
if st.sidebar.button("Database ထဲသို့ ထည့်သွင်းမည်"):
    if uploaded_files:
        with st.sidebar.spinner("ဒေတာများကို Vector Database ထဲသို့ ထည့်သွင်းနေပါသည်..."):
            import pandas as pd
            from langchain_core.documents import Document
            from langchain_community.vectorstores import Chroma
            import os
            import gc
            
            documents = []
            
            # 💡 ဖိုင်တစ်ခုချင်းစီကို အမှားအယွင်းမရှိ Data အကုန်ဖတ်မည့် စနစ်အသစ်
            # 💡 Excel ဖိုင်တစ်ခုချင်းစီရဲ့ Sheet အားလုံးကို အပြည့်အဝ ပတ်ဖတ်မည့် စနစ်အသစ်
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.name.endswith(('.xlsx', '.xls')):
                        # sheet_name=None ထည့်လိုက်ခြင်းဖြင့် Sheet အားလုံးကို dict အနေနဲ့ ဖတ်ပါမယ်
                        xl = pd.ExcelFile(uploaded_file)
                        sheet_names = xl.sheet_names
                        
                        search_term = column_name_input.strip().lower()
                        
                        # Sheet တစ်ခုချင်းစီကို လိုက်ဖတ်မယ်
                        for sheet in sheet_names:
                            df = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)
                            
                            target_col_index = None
                            start_row_index = 1
                            found_header = False
                            
                            # ၁။ ပထမဆုံး ၁၅ ကြောင်းအတွင်း Column Header ပါ/မပါ ရှာမယ်
                            for r_idx in range(min(len(df), 15)):
                                row_values = [str(val).strip().lower() for val in df.iloc[r_idx]]
                                
                                # UI က ရိုက်ထည့်တဲ့ နာမည်နဲ့ တူရင် ယူမယ်
                                if search_term and (search_term in row_values):
                                    target_col_index = row_values.index(search_term)
                                    start_row_index = r_idx + 1
                                    found_header = True
                                    break
                                    
                                # Keyword တွေ ပါသလား ရှာမယ်
                                for c_idx, val in enumerate(row_values):
                                    if 'title' in val or 'thesis' in val or 'ခေါင်းစဉ်' in val or 'topic' in val:
                                        target_col_index = c_idx
                                        start_row_index = r_idx + 1
                                        found_header = True
                                        break
                                if found_header:
                                    break
                            
                            # ၂။ Header ရှာမတွေ့ရင် default အနေနဲ့ 'Thesis Title' Column က Index 3 မှာ အမြဲရှိတတ်လို့ 3 ကို ယူမယ်၊ မရှိရင် 0 ယူမယ်
                            if target_col_index is None:
                                target_col_index = 3 if len(df.columns) > 3 else 0
                                start_row_index = 1
                            
                            # ၃။ ဒေတာတွေကို ဆွဲထုတ်ပြီး Documents List ထဲ ထည့်မယ်
                          
                            if target_col_index < len(df.columns):
                                for index in range(start_row_index, len(df)):
                                    text_content = str(df.iloc[index, target_col_index]).strip()
                                    
                                    # 💡 ဒေတာအလွတ် မဟုတ်မှ ဆက်လုပ်မယ်
                                    if text_content and text_content.lower() != "nan" and len(text_content)>8:
                                        
                                        # (က) Header စာသားတွေ၊ နံပါတ်စဉ်တွေ သို့မဟုတ် "No", "-" စတာတွေကို ဖယ်ထုတ်မယ်
                                        if text_content.lower() in ["thesis title", "title", "ခေါင်းစဉ်", "topic", "-", "no", "name", "roll no"]:
                                            continue
                                            
                                        # (ခ) စာလုံးရေ ၈ လုံးထက်ကျော်တဲ့ တကယ့် Thesis ခေါင်းစဉ်တွေကိုပဲ ယူမယ်
                                        if len(text_content) > 8:
                                            meta_data = {"source": f"{uploaded_file.name} ({sheet})", "row": index}
                                            doc = Document(page_content=text_content, metadata=meta_data)
                                            documents.append(doc)
                                        
                except Exception as e:
                    print(f"Error reading file {uploaded_file.name}: {e}")
                    continue
                            
            if documents:
                # 🚨 Windows WinError 32 ကို ကျော်လွှားရန် Memory ထဲက DB connection တွေကို အရင်ရှင်းထုတ်ခြင်း
                if "vector_store" in st.session_state:
                    st.session_state.vector_store = None
                gc.collect()
                
                # 💡 Folder ကို ဖျက်မယ့်Chroma ရဲ့ အထဲက client သုံးပြီး ဒေတာကို Reset ချပစ်နည်း
                # 💡 Folder ကို ဖျက်မယ့်အစား Chroma ရဲ့ အထဲက client သုံးပြီး ဒေတာကို Reset ချပစ်နည်း
                try:
                    import chromadb
                    persistent_client = chromadb.PersistentClient(path="./chroma_db_storage")
                    # လက်ရှိ ရှိနေတဲ့ collection အဟောင်းတွေကို အကုန်ရှင်းပစ်မယ်
                    for collection in persistent_client.list_collections():
                        persistent_client.delete_collection(collection.name)
                except Exception:
                    pass # ပထမဆုံးအကြိမ် ဆောက်တာဆိုရင် ကျော်သွားမယ်
                
                # ဒေတာအသစ်များဖြင့် Database အသစ်ပြန်ဆောက်ခြင်း
                st.session_state.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory="./chroma_db_storage",
                    collection_metadata={"hnsw:space": "cosine"}
                )
                st.sidebar.success(f"🎉 ဌာနအသီးသီးမှ ဒေတာစုစုပေါင်း ({len(documents)}) ခုကို အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
                st.rerun()
            else:
                st.sidebar.error("❌ ဖတ်လို့ရတဲ့ Thesis ဒေတာ စာသားများကို Excel ထဲတွင် လုံးဝ ရှာမတွေ့ပါရှင်။")
    else:
        st.sidebar.warning("⚠️ ကျေးဇူးပြု၍ ဖိုင်အရင် တင်ပေးပါဦးဗျာ။")

# စက်ထဲမှာ ChromaDB ရှိနေပြီးသားဆိုရင် စိမ်းရောင် Success Box ပြမယ်
# 💡 Sidebar အောက်ခြေနားမှာ ဒေတာ အရေအတွက် ပြရန် ကုဒ်တိုးပေးခြင်း
if not is_database_empty():
    st.sidebar.success("✅ Database (Knowledge Base) Ready ဖြစ်နေပါပြီ။")
    
    try:
        # ၁။ စုစုပေါင်း အရေအတွက်ကို ယူမယ်
        db_data = st.session_state.vector_store.get()
        total_records = len(db_data['ids'])
        st.sidebar.metric(label="စုစုပေါင်း သိမ်းဆည်းထားသော Title အရေအတွက်", value=total_records)
        
        # ၂။ ဘယ်ဖိုင်ကနေ Data ဘယ်နှစ်ခုစီ ပါဝင်နေလဲဆိုတာ Summary တွက်ပြီး ပြမယ်
        if 'metas' in db_data and db_data['metas']:
            from collections import Counter
            # ဖိုင်နာမည်တွေကို လိုက်ရေတွက်ခြင်း
            sources = [meta.get('source', 'Unknown') for meta in db_data['metas'] if meta]
            file_counts = Counter(sources)
            
            summary_text = ""
            for file_name, count in file_counts.items():
                summary_text += f"• {file_name}: **{count}** ခု ပါဝင်သည်။\n"
                
            st.sidebar.info(f"📊 **ဖိုင်အလိုက် သိမ်းဆည်းထားမှု:**\n{summary_text}")
    except Exception as e:
        pass

# =========================================================================
# ၄။ Chat Engine အလုပ်လုပ်ပုံအပိုင်း
# =========================================================================
if user_query := st.chat_input("စစ်ဆေးချင်သော Thesis ခေါင်းစဉ် သို့မဟုတ် မေးခွန်း ရိုက်ထည့်ပါ..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # User က သာမန် နှုတ်ဆက်တာလား သို့မဟုတ် Thesis စစ်ခိုင်းတာလားဆိုတာ စစ်ဆေးခြင်း
        query_lower = user_query.lower()
        is_thesis_check = any(kw in query_lower for kw in ["thesis", "title", "ခေါင်းစဉ်", "စစ်", "similar", "တူ", "တူလား", "တူညီမှု", "ဆင်တူ"])
        
        # 🚨 အကယ်၍ Thesis စစ်ခိုင်းတာ ဖြစ်ပြီး၊ DB ကလည်း အလွတ်ဖြစ်နေမှသာ Warning ပြပါမယ်
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
                    file_name = doc.metadata.get("source", "Unknown.xlsx")
                    major_name = file_name.replace(".xlsx", "").replace(".xls", "").upper()
                    
                    context_list.append(f"- Title: {doc.page_content} [Department/Major: {major_name}]")
                
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
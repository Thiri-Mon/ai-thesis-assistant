import os
import gc           
import streamlit as st
import tempfile
import pandas as pd  # For Excel နှင့် CSV 
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


st.set_page_config(page_title="AI-Powered Thesis Assistant", layout="wide")
st.title("🌐 Welcome to AI-Powered Thesis Assistant (SBERT + ChromaDB + Ollama)")


# 🌟 SBERT Embedding Model ကို Initialize လုပ်ခြင်း
DB_DIR = "./chroma_db_storage"

# 🌟 SBERT Embedding Model ကို Initialize လုပ်ခြင်း
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True} # စက်ထဲမှာ model ရှိပြီးသားဖြစ်ရပါမယ်။ မရှိရင် False ပြောင်းပါ။
    )

embeddings = load_embedding_model()

# =========================================================================
# ၁။ Session State စတင်ခြင်း နှင့် ChromaDB Auto-Load စနစ် (တစ်နေရာတည်းမှာပဲ လုပ်သည်)
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

# ChromaDB အလွတ် ဟုတ်/မဟုတ် စစ်ဆေးမည့် Function
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
            return False  # ဒေတာ တကယ်ရှိတယ်!
        return True       # ဒေတာ မရှိဘူး အလွတ်ကြီးပဲ
    except Exception:
        return True




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

# Dynamic Distance Threshold Slider
threshold_input = st.sidebar.slider(
    "Duplicate Sensitivity Threshold (Distance)",
    min_value=0.20,
    max_value=0.60,
    value=0.45,  # Adjusted to your new preferred strict value
    step=0.01,
    help="Lower values are more lenient (require higher similarity to flag). Higher values are stricter (flag even moderately similar titles as duplicates)."
)


# Database အသစ်ဆောက်မည့် ခလုတ်
if st.sidebar.button("Database ထဲသို့ ထည့်သွင်းမည်"):
    if uploaded_files:
        with st.sidebar.spinner("ဒေတာများကို Vector Database ထဲသို့ ထည့်သွင်းနေပါသည်..."):
            
            documents = []
            
            # 💡 ဖိုင်တစ်ခုချင်းစီကို အမှားအယွင်းမရှိ Data အကုန်ဖတ်မည့် စနစ်အသစ်
            # 💡 Excel ဖိုင်တစ်ခုချင်းစီရဲ့ Sheet အားလုံးကို အပြည့်အဝ ပတ်ဖတ်မည့် စနစ်အသစ်
            for uploaded_file in uploaded_files:
                try:
                    # Case A: Handling Excel Workbooks
                    if uploaded_file.name.endswith(('.xlsx', '.xls')):
                        uploaded_file.seek(0) # 💡 Reset file pointer for pandas
                        # sheet_name=None ထည့်လိုက်ခြင်းဖြင့် Sheet အားလုံးကို dict အနေနဲ့ ဖတ်ပါမယ်
                        xl = pd.ExcelFile(uploaded_file)
                        sheet_names = xl.sheet_names
                        search_term = column_name_input.strip().lower()
                        
                        # Sheet တစ်ခုချင်းစီကို လိုက်ဖတ်မယ်
                        for sheet in sheet_names:
                            uploaded_file.seek(0) # 💡 Reset pointer per sheet read
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
                          
                            # ✨ ဒီကုဒ်သစ်ကြီးတစ်ခုလုံးကို အစားထိုး ထည့်ပေးလိုက်ပါရှင်
                            if target_col_index < len(df.columns):
                                for index in range(start_row_index, len(df)):
                                    text_content = str(df.iloc[index, target_col_index]).strip()
                                    
                                    # 💡 ဒေတာအလွတ် မဟုတ်မှ ဆက်လုပ်မယ်
                                    if text_content and text_content.lower() != "nan" and len(text_content) > 8:
                                        
                                        # (က) Header စာသားတွေ၊ နံပါတ်စဉ်တွေ သို့မဟုတ် "No", "-" စတာတွေကို ဖယ်ထုတ်မယ်
                                        if text_content.lower() in ["thesis title", "title", "ခေါင်းစဉ်", "topic", "-", "no", "name", "roll no"]:
                                            continue
                                            
                                        # 💡 ကျောင်းသားနာမည် နှင့် ရိုးလ်နံပါတ်ကိုပါ Excel Column ထဲက ဆွဲထုတ်မယ်
                                        student_name = "Unknown"
                                        roll_no = "Unknown"
                                        
                                        try:
                                            if len(df.columns) > 1:
                                                student_name = str(df.iloc[index, 1]).strip()
                                            if len(df.columns) > 2:
                                                roll_no = str(df.iloc[index, 2]).strip()
                                        except Exception:
                                            pass

                                        # 💡 Excel ရဲ့ မျက်မြင် Row နံပါတ် အစစ်အမှန်ဖြစ်အောင် + 1 လုပ်ပေးပါတယ်
                                        actual_excel_row = index + 1
                                        
                                        # Metadata ထဲမှာ နာမည်၊ ရိုးလ်နံပါတ်နဲ့ Row အစစ်ကို သေချာတွဲသိမ်းမယ်
                                        meta_data = {
                                            "source": f"{uploaded_file.name} ({sheet})", 
                                            "row": actual_excel_row,
                                            "student_name": student_name,
                                            "roll_no": roll_no
                                        }
                                        
                                        doc = Document(page_content=text_content, metadata=meta_data)
                                        documents.append(doc)


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
                    st.sidebar.error(f"Error reading file {uploaded_file.name}: {e}")
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
                
               # ✨ ဒီကုဒ်သစ်လေးကို အစားထိုး ထည့်ပေးလိုက်ပါရှင်
                # 💡 ဒေတာအသစ်များဖြင့် Database အသစ်ပြန်ဆောက်ခြင်း (IDs များကို စနစ်တကျ ထည့်သွင်းခြင်း)
                computed_ids = [f"doc_{idx}_{doc.metadata.get('row', index)}" for idx, doc in enumerate(documents)]
                
                st.session_state.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    ids=computed_ids,  # ✨ ဒေတာတွေ ရောထွေးမသွားအောင် ထိန်းပေးမယ့် လိုင်းလေးပါ
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
        st.sidebar.metric(label="စုစုပေါင်း Title အရေအတွက်", value=total_records)
        
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
# 🧹 Clear Chat History Mechanism (Fixes Memory Contamination)
# =========================================================================
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "မင်္ဂလာပါ အစ်မ! Thesis ခေါင်းစဉ် Memory ကို ရှင်းလင်းပြီးပါပြီ။ ခေါင်းစဉ်အသစ်များ ထပ်မံစစ်ဆေးနိုင်ပါပြီ။"
    }]
    st.rerun()

# =========================================================================
# ၄။ Chat Engine အလုပ်လုပ်ပုံအပိုင်း
# =========================================================================
# ပြန်ပွင့်လာတိုင်း ယခင် Chat သမိုင်းကြောင်းကို ပတ်ပြပေးခြင်း
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_query := st.chat_input("စစ်ဆေးချင်သော Thesis ခေါင်းစဉ် သို့မဟုတ် မေးခွန်း ရိုက်ထည့်ပါ..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # User က သာမန် နှုတ်ဆက်တာလား သို့မဟုတ် Thesis စစ်ခိုင်းတာလားဆိုတာ စစ်ဆေးခြင်း
                # =========================================================================
        # 🔴 FIXED ROUTER ENGINE: SEPARATES GREETINGS FROM RAW PROPOSALS
        # =========================================================================
        query_lower = user_query.lower()
        
        # 1. First, check if they are just greeting or making a short conversational statement
        is_greeting = any(g_kw in query_lower for g_kw in ["hello", "hi", "hey", "မင်္ဂလာပါ", "good morning", "good afternoon"])
        
        # 2. Check if they used checking keywords
        has_keywords = any(kw in query_lower for kw in ["thesis", "title", "ခေါင်းစဉ်", "စစ်", "similar", "duplicate", "တူ"])
        
         # 💡 NEW: Catch engineering keywords commonly found in short titles
        has_engineering_terms = any(eng_kw in query_lower for eng_kw in ["system", "detection", "analysis", "monitoring", "framework", "using", "control"])
        # 3. Only run a database check if it's NOT a basic greeting, has keywords, and is a substantial sentence length
        is_thesis_check = False
        #if not is_greeting and (has_keywords or len(user_query) > 35):
                # Ensure short transitional sentences do not force lookups
                #if len(user_query) > 35 or not has_keywords:
        # Smart Gate: Triggers database lookup if it's NOT a greeting AND:
        # (Has thesis keywords OR is a long title OR is a short engineering title)
        
        if not is_greeting and (has_keywords or len(user_query) > 35 or has_engineering_terms):
                is_thesis_check = True
                 

        
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
            if is_thesis_check and not is_database_empty():
                # 💡 ကွက်တိ ရာခိုင်နှုန်း score တွက်ချက်ပြီး ရှာဖွေခြင်း
                docs_and_scores = st.session_state.vector_store.similarity_search_with_score(user_query, k=4)
                # 📍 ADD THESE TWO DEBUG LINES RIGHT HERE:
                print("\n🔍 --- DEBUGGING CHROMADB LOOKUP ---")
                print(f"User Query Submitted: '{user_query}'")

                for doc, score in docs_and_scores:
                    # distance score က သုညနားနီးလေ တူလေမို့လို့ 0.4 ထက်ငယ်ရင် Duplicate လို့ သတ်မှတ်မယ်
                    if score < threshold_input: 
                        # 📍 ADD THIS LINE TO PRINT THE REAL MATH SCORES:
                        print(f"Matched Title in DB: '{doc.page_content}' | Calculated Distance: {score:.4f}")
                        is_duplicate_detected = True
                    
                    # =========================================================================
                    # 🔴 FIXED DEPT MAPPING: FORCES FULL OFFICIAL UNIVERSITY TITLES
                    # =========================================================================
                    raw_source = doc.metadata.get("source", "Unknown.xlsx")
                    
                    # Force Python to take only the first element [0] if a split occurs
                    if " (" in raw_source:
                        clean_file = raw_source.split(" (")[0]
                    else:
                        clean_file = raw_source
                        
                    # Now clean_file is guaranteed to be a string, so .replace() will work perfectly!
                    dept_code = clean_file.replace(".xlsx", "").replace(".xls", "").upper().strip()
                    # Official University Department Mapping Dictionary
                    dept_mapping = {
                        "CE": "Computer Engineering",
                        "IS": "Information And Science",
                        "ECE": "Electronics  Engineering",
                        "PRE": "Precision Engineering",
                        "AME": "Advanced  Materials Engineering"
                    }

                    # Lookup the full name, default to the raw code if not found in the dictionary
                    major_name = dept_mapping.get(dept_code, dept_code)

                    
                    context_list.append(f"- Title: {doc.page_content} [Department/Major: {major_name}](Distance: {score:.3f})")
                
                context = "\n".join(context_list)

             # Direct Streamlit Output Bypass (Bypasses LLM Guesswork Entirely)
            if is_thesis_check:
                if is_duplicate_detected:
                    st.warning("⚠️ **System Alert: Similar/Duplicate Records Located in Datastore:**")
                    st.code(context, language="text") # Prints the raw, unaltered database strings in a clean box
                else:
                    # Case B: If the title is unique (>= 0.45), print matches as recommended literature references &
                    # 💡 ONLY SHOW THE BLUE CONTAINER IF THE CLOSEST MATCH IS RELEVANT (Distance < 0.520)
                    # This cleanly hides the references window during conversational turns!Smart controlling for normal conversation
                    #if len(docs_and_scores) > 0 and docs_and_scores[0][1] < 0.520:
                    #if len(docs_and_scores) > 0 and any(item[1] < 0.520 for item in docs_and_scores):
                    #if len(docs_and_scores) > 0 and any(score < 0.520 for _, score in docs_and_scores):
                    if len(docs_and_scores) > 0 and any(score < 0.520 for _, score in docs_and_scores):
                        st.info("📊 **Literature Review References (Closest Historical Research Found):**")
                        st.code(context, language="text")
                    
            # Chat History (Memory) တည်ဆောက်ခြင်း
                        # =========================================================================
            # 🔄 FIXED AREA 3: SLIDING WINDOW CHAT MEMORY (Perfect Long-Turn Solution)
            # =========================================================================
            chat_history_str = ""
            
            # Slice the history to only remember the last 4 items (2 turns of user/assistant exchange)
            # This perfectly prevents old topics (like Food or Malarial Parasites) from bleeding into new turns
            recent_messages = st.session_state.messages[-4:] if len(st.session_state.messages) > 4 else st.session_state.messages
            
            for msg in recent_messages:
                # Skip the system prompt configurations from accumulating in the text
                if msg["role"] == "system":
                    continue
                role_label = "User" if msg["role"] == "user" else "Assistant"
                
                # Clean out the raw datastore match text from memory so it doesn't pollute the next check
                clean_content = msg['content'].split("### 🔍 Real Datastore Matches")[0].strip()
                chat_history_str += f"{role_label}: {clean_content}\n"



            # =========================================================================
            # 🔴 BULLETPROOF LOGIC ROUTER (Separates Unique from Duplicate Channels)
            # =========================================================================
                        # =========================================================================
            # 🔴 OPTIMIZED HYBRID LOGIC ROUTER (The Best Combination)
            # =========================================================================
                       # =========================================================================
            # 🔴 ULTRA-STRICT HYBRID PROMPT ROUTER (Zero Conversation on Duplicates)
            # =========================================================================
            if is_duplicate_detected:
                # Mode A: High-Security Duplicate Warning (Forces absolute raw text use)
                system_prompt = (
                    f"You are a strict academic verification engine. A dangerous duplicate has been detected.\n\n"
                    f"DATABASE SEARCH STATUS: YES, IT ALREADY EXISTS (DUPLICATE)\n\n"
                    f"CRITICAL COMPLIANCE RULES:\n"
                    f"1. State clearly that a duplicate or highly similar title exists.\n"
                    f"2. You are FORBIDDEN from writing your own title names, creating distances, or inventing departments like BIOL. You do not know any biology departments.\n"
                    f"3. Look at the 'Context' block below. You MUST copy and paste the text lines inside the Context block exactly as they are written, word-for-word.\n\n"
                    f"Context of closest titles from database (COPY AND PASTE THESE ONLY):\n{context}"
                )
            else:
                # Mode B: Creative Unique Innovation (Now handles general conversation vs raw titles)
                system_prompt = (
                    f"You are an expert academic advisor in Computer Science and Engineering.\n"
                    f"DATABASE SEARCH STATUS: COMPLETELY UNIQUE / NEW TITLE APPROVED.\n\n"
                    f"CRITICAL RULES:\n"
                    f"1. If the user query is just a greeting, introduction, or asking how to start (e.g., 'Hello', 'Hi', 'I want to check my title'), do NOT suggest specific technical extensions yet. Instead, politely welcome them, tell them the system is ready, and ask them to paste their exact, raw thesis title.\n"
                    f"2. If the user query is an actual, unique thesis title description, congratulate them warmly and provide exactly 3 highly innovative research extensions or advanced methodology variations specifically tailored to expand their exact topic idea.\n"
                    f"3. Do NOT mention distance scores, copy-paste templates, or database rows, because no duplicate exists.\n\n"
                    f"Conversation History:\n{chat_history_str if chat_history_str else 'No previous conversation.'}\n\n"
                    f"Context of closest titles from database:\nEMPTY (No duplicates found below threshold)"
                )



                
            
            # System Prompt စည်းကမ်းသတ်မှတ်ချက်
                #system_prompt = (
                #f"You are an expert academic advisor in Computer Science and Engineering.\n"
                #f"Your primary task is to evaluate whether the user's proposed thesis title already exists in the database or if it is completely unique.\n\n"
                #f"STATUS FROM DATABASE SEARCH:\n"
                #f"- Is exact or near-exact title found?: {'YES, IT ALREADY EXISTS (DUPLICATE)' if is_duplicate_detected else 'NO, THIS TITLE IS UNIQUE / NEW.'}\n\n"
                #f"CRITICAL RULES:\n"
                #f"1. If the status is 'NO', congratulate the user and suggest 3 new research directions extending their topic ideas.\n"
                #f"2. If the status is 'YES', you are STRICLY FORBIDDEN from creating, inventing, or paraphrasing title names. You MUST copy and paste the EXACT literal words provided in the 'Context' block below word-for-word, along with their correct [Department/Major] and (Distance) score.\n"
                #f"3. Do not summarize or alter the names of the matches. Present them exactly as they are stored.\n\n"
                #f"Conversation History:\n{chat_history_str if chat_history_str else 'No previous conversation.'}\n\n"
                #f"Context of closest titles from database (USE THESE ONLY):\n{context if is_duplicate_detected else 'EMPTY (No duplicates found below threshold)'}"
            #)

            

            
            messages_input = [
                ("system", system_prompt),
                ("user", user_query)
            ]
            
        
            

            llm = ChatOllama(model="llama3.2", temperature=0.3, repeat_penalty=1.2,streaming=True)
            #  UPDATED FIXED ENGINE BLOCK:
            #llm = ChatOllama(
            #    model="llama3.2", 
            #   temperature=0.5, # Slightly raised to avoid repetitive phrase trapping
            #   streaming=True,
            #   num_predict=256, # Safety maximum token cutoff for response length
            #   repeat_penalty=1.2, # Direct instruction to punish repetitive sentences
            #   top_k=20,
            #   top_p=0.8
           # )


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
                    

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

# ၁။ Session State ခေါ်ယူခြင်း
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "မင်္ဂလာပါ အစ်မ! Thesis ခေါင်းစဉ် Dataset (Excel/CSV/PDF) တင်ပြီး ဆင်တူယိုးမှား စစ်ဆေးနိုင်ပါပြီ။"}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 🚨 ဖြည့်စွက်ချက် ၁ - ChromaDB ထဲမှာ တကယ် Data ရှိမရှိ စစ်ဆေးပေးမည့် Function
def is_database_empty():
    if st.session_state.vector_store is None:
        return True
    try:
        # Chroma DB ထဲက stored IDs တွေကို ဆွဲထုတ်ပြီး စစ်တာပါ
        db_data = st.session_state.vector_store.get()
        if not db_data or len(db_data.get('ids', [])) == 0:
            return True
        return False
    except Exception:
        return True

# 🌟 SBERT Embedding Model ကို Initialize လုပ်ခြင်း
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embedding_model()

# ၂။ Sidebar Configuration
with st.sidebar:
    st.header("၁။ AI Model ရွေးချယ်ရန်")
    available_models = ["qwen2:0.5b","llama3.2", "mistral"]
    selected_model = st.selectbox("Active Ollama Model:", available_models)

    st.write("---")
    st.header("၂။ Knowledge Base / Dataset Upload")
    
    uploaded_files = st.file_uploader(
        "ယခင် Thesis ခေါင်းစဉ်ဟောင်းများ Dataset တင်ရန် (ဖိုင်အများကြီးတွဲတင်နိုင်ပါသည်)", 
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True
    )
    
    column_name = "Thesis Title"
    if uploaded_files:
        has_tabular = any(f.name.endswith('.xlsx') or f.name.endswith('.csv') for f in uploaded_files)
        if has_tabular:
            column_name = st.text_input("Thesis ခေါင်းစဉ်ပါသော Column နာမည်ကို ရိုက်ထည့်ပါ (ဥပမာ- Title)", value="Thesis Title")

    if uploaded_files and st.button("Database ထဲသို့ ထည့်သွင်းမည်"):
        with st.spinner("SBERT ဖြင့် အချက်အလက်များကို Vector ပြောင်း၍ Chroma DB ထဲ သိမ်းနေပါသည်..."):
            
            all_titles = []  
            pdf_chunks = []  
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file, header=3)
                    if column_name in df.columns:
                        titles = df[column_name].dropna().astype(str).tolist()
                        all_titles.extend(titles)  
                    else:
                        st.error(f"❌ {uploaded_file.name} ထဲတွင် '{column_name}' ဆိုသော Column ရှာမတွေ့ပါ။")

                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if column_name in df.columns:
                        titles = df[column_name].dropna().astype(str).tolist()
                        all_titles.extend(titles)  
                    else:
                        st.error(f"❌ {uploaded_file.name} ထဲတွင် '{column_name}' ဆိုသော Column ရှာမတွေ့ပါ။")

                elif uploaded_file.name.endswith('.pdf'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = text_splitter.split_documents(docs)
                    pdf_chunks.extend(chunks)

            if all_titles:
                st.session_state.vector_store = Chroma.from_texts(
                    texts=all_titles, embedding=embeddings, persist_directory="./chroma_db_storage"
                )
                st.success(f"✅ Excel/CSV ဖိုင်များထဲမှ ခေါင်းစဉ်စုစုပေါင်း ({len(all_titles)}) ခုကို ပေါင်းစည်းသိမ်းဆည်းပြီးပါပြီ။")
                
            if pdf_chunks:
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=pdf_chunks, embedding=embeddings, persist_directory="./chroma_db_storage"
                    )
                else:
                    st.session_state.vector_store.add_documents(documents=pdf_chunks)
                st.success(f"✅ PDF ဖိုင်များထဲမှ စာသားများကိုလည်း Database ထဲသို့ ထည့်သွင်းပြီးပါပြီ။")

# ၃။ Conversation History ပြသခြင်း
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ၄။ Chat Engine အလုပ်လုပ်ပုံအပိုင်း
if user_query := st.chat_input("စစ်ဆေးချင်သော Thesis ခေါင်းစဉ် သို့မဟုတ် မေးခွန်း ရိုက်ထည့်ပါ..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # 🚨 ပြင်ဆင်ချက် - User က သာမန် နှုတ်ဆက်တာလား သို့မဟုတ် Thesis စစ်ခိုင်းတာလားဆိုတာ စစ်ဆေးခြင်း
        # (စာလုံးအသေးပြောင်းပြီး thesis, title, စစ် စတဲ့ keyword ပါမပါ ကြည့်တာပါ)
        query_lower = user_query.lower()
        is_thesis_check = any(kw in query_lower for kw in ["thesis", "title", "ခေါင်းစဉ်", "စစ်", "similar"])
        
        # 🚨 အကယ်၍ Thesis စစ်ခိုင်းတာ ဖြစ်ပြီး၊ DB ကလည်း အလွတ်ဖြစ်နေမှသာ Warning ပြပါမယ်
        if is_thesis_check and is_database_empty():
            warning_response = "⚠️ စနစ်ထဲတွင် နှိုင်းယှဉ်စစ်ဆေးစရာ Thesis Dataset (Knowledge Base) မရှိသေးပါ။ ကျေးဇူးပြု၍ ဘယ်ဘက် Sidebar တွင် Excel/CSV/PDF ဖိုင်တစ်ခုခု အရင် တင်ပေးပါဗျာ။"
            response_placeholder.write(warning_response)
            st.session_state.messages.append({"role": "assistant", "content": warning_response})
        
        else:
            # သာမန်နှုတ်ဆက်တာပဲဖြစ်ဖြစ် (သို့) DB ထဲမှာ Data ရှိနေရင် ပုံမှန်အတိုင်း အလုပ်လုပ်မည်
            context = ""
            
            # DB ရှိမှသာ Context ဆွဲထုတ်မယ်၊ မရှိရင် context = "" အတိုင်းပဲ LLM ဆီသွားမယ်
            if not is_database_empty():
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                relevant_docs = retriever.invoke(user_query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # System Prompt ကို လက်ရှိ အခြေအနေနဲ့ ကိုက်ညီအောင် ညှိလိုက်ပါတယ်
            system_prompt = (
                f"You are an expert academic advisor in Computer Science and Information Technology.\n"
                f"Your task is to evaluate the user's input or proposed thesis title.\n\n"
                f"CRITICAL RULES:\n"
                f"1. If the user is just saying greeting words (like Hello, Hi, Mingalaba), just greet them back nicely and tell them you are ready to help once they upload the dataset and share a title.\n"
                f"2. If the user provides a thesis title but the Context below is EMPTY, politely tell them that you cannot check for duplication yet because no database is uploaded, but give general feedback on their title structure.\n"
                f"3. Do NOT invent or fabricate fake past thesis titles from your own imagination.\n\n"
                f"Context of past titles:\n{context}"
            )
            
            messages_input = [
                ("system", system_prompt),
                ("user", user_query)
            ]
            
            llm = ChatOllama(model=selected_model, temperature=0.3, streaming=True)

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
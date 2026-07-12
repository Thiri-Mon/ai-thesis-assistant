import pandas as pd
import os

# အစ်မရဲ့ ဖိုင် ၅ ဖိုင် နာမည်စာရင်း
files = ["AME.xlsx", "CE.xlsx", "EcE.xlsx", "IS.xlsx", "PrE.xlsx"]
total_titles_in_excel = 0

print("🔍 --- Excel ဖိုင်အလိုက် မူရင်း Data အရေအတွက် စစ်ဆေးခြင်း ---")

for file in files:
    if os.path.exists(file):
        try:
            xl = pd.ExcelFile(file)
            file_total = 0
            
            # Sheet အားလုံးကို လိုက်ပတ်မယ်
            for sheet in xl.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet, header=None)
                
                # အစ်မတို့ရဲ့ Thesis Title က Column D (Index 3) မှာ အမြဲရှိတာမို့လို့
                if len(df.columns) > 3:
                    # Header row တွေကို ကျော်ပြီး အောက်က Data တွေကိုပဲ ရေတွက်မယ်
                    # Row 4 ကနေစဖတ်တာမို့လို့ Index 4 ကနေ စရေပါမယ်
                    titles_in_sheet = df.iloc[4:, 3].dropna().astype(str).str.strip()
                    
                    # စာသားအလွတ်တွေနဲ့ Header တွေကို ဖယ်ထုတ်ပစ်မယ်
                    clean_titles = [t for t in titles_in_sheet if t and t.lower() != "nan" and len(t) > 5 and "title" not in t.lower() and "ခေါင်းစဉ်" not in t]
                    
                    file_total += len(clean_titles)
            
            print(f"• {file}: Original Total Titles**{file_total}** ခု ရှိသည်။")
            total_titles_in_excel += file_total
            
        except Exception as e:
            print(f"❌ {file} is found in error: {e}")
    else:
        print(f"⚠️ {file}Not found, check in folder")

print("--------------------------------------------------")
print(f"📊 Total titles of 5 Major:  **{total_titles_in_excel}** ခု")
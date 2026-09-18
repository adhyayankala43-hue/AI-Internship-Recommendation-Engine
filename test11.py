from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
import traceback
import os
import csv         
import time
import datetime 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder='.')
@app.get("/ping")
def ping():
    return "ok"
# --- DATA ENTRY Consent---
def log_to_timeline(student_id, reg_no, student_name, company, interaction_type, score):
    file_path = 'user_interactions_timeline.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Create headers if the file is brand new
        if not file_exists:
            writer.writerow(['Timestamp', 'Student_ID', 'Registration_No', 'Student_Name', 'Company_Selected', 'Interaction_Type', 'Interaction_Score'])
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([student_id, reg_no, student_name, company, score, timestamp])

# --- Company Data---
@app.route('/log-company-interaction', methods=['POST'])
def log_company_interaction():
    try:
        data = request.json
        company = data.get('company', 'Unknown')
        student_id = data.get('student_id', 'N/A')
        reg_no = data.get('registration_no', 'N/A')
        student_name = data.get('name', 'Anonymous')
        consent = data.get('consent_given', False)
        
        if consent and student_id != 'N/A':
            # Log the explicit company selection with a score of 3.0
            log_to_timeline(student_id, reg_no, student_name, company, 'company_selection', 3.0)
            print(f"🎯 Company Interaction Logged: {company} for {student_name} ({student_id})")
            return jsonify({"status": "success", "message": "Thank you for your response and feedback"})
        else:
            return jsonify({"status": "skipped", "message": "Consent not given or invalid ID"})
            
    except Exception as e:
        print("❌ ERROR LOGGING COMPANY INTERACTION:", e)
        return jsonify({"error": str(e)}), 500
        # --- DEEP LEARNING MODEL ARCHITECTURE ---
class RecommendationNet(nn.Module):
    def __init__(self, num_companies, embedding_dim=16):
        super(RecommendationNet, self).__init__()
        self.company_embedding = nn.Embedding(num_companies, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, company_idx):
        embed = self.company_embedding(company_idx)
        x = self.relu(self.fc1(embed))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

print("Loading Tri-Engine databases and Deep Learning weights...")
try:
    df_internships = pd.read_csv(os.path.join(BASE_DIR, "internship_database_2.csv"), encoding='latin-1')
    df_ratings = pd.read_csv(os.path.join(BASE_DIR, "company_ratings.csv"), encoding='latin-1')
    df_interactions = pd.read_csv(os.path.join(BASE_DIR, "user_interactions_timeline.csv"), on_bad_lines='skip')
    df_timeline = df_interactions
    df_internships.columns = df_internships.columns.str.strip()
    df_ratings.columns = df_ratings.columns.str.strip()
    df_interactions.columns = df_interactions.columns.str.strip()
    df_interactions['Interaction_Score'] = pd.to_numeric(df_interactions['Interaction_Score'], errors='coerce')
    
    if 'Company Name' in df_internships.columns: df_internships.rename(columns={'Company Name': 'Company_Name'}, inplace=True)
    if 'Company Name' in df_ratings.columns: df_ratings.rename(columns={'Company Name': 'Company_Name'}, inplace=True)
    if 'Company Name' in df_interactions.columns: df_interactions.rename(columns={'Company Name': 'Company_Name'}, inplace=True)

    df_merged = pd.merge(df_internships, df_ratings, on="Company_Name").fillna("")
    df_merged['Internship_Profile'] = (
        df_merged['Skill_1'].astype(str) + " " + 
        df_merged['Skill_2'].astype(str) + " " + 
        df_merged['Skill_3'].astype(str) + " " + 
        df_merged['Recommended_Degree'].astype(str)
    )

    peer_scores = df_interactions.groupby('Company_Name')['Interaction_Score'].mean(numeric_only=True).reset_index()
    peer_scores.rename(columns={'Interaction_Score': 'Raw_Collab_Score'}, inplace=True)
    scaler = MinMaxScaler()
    peer_scores['Collab_Score'] = scaler.fit_transform(peer_scores[['Raw_Collab_Score']])
    df_merged = pd.merge(df_merged, peer_scores, on="Company_Name", how="left")
    df_merged['Collab_Score'] = df_merged['Collab_Score'].fillna(0) 

    company_classes = np.load(os.path.join(BASE_DIR, "company_classes.npy"), allow_pickle=True)
    num_companies = len(company_classes)
    
    dl_model = RecommendationNet(num_companies)
    dl_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "dl_model.pth"), weights_only=True, map_location=torch.device('cpu')))
    dl_model.eval() 

    with torch.no_grad():
        all_company_indices = torch.arange(num_companies, dtype=torch.long)
        dl_scores = dl_model(all_company_indices).flatten().numpy()
        
    dl_score_dict = dict(zip(company_classes, dl_scores))
    df_merged['DL_Score'] = df_merged['Company_Name'].map(dl_score_dict).fillna(0)

    print("✅ Databases and PyTorch Model loaded successfully!")
except Exception as e:
    print("❌ ERROR LOADING DATABASES OR DL MODEL:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_cgpa = float(data.get('cgpa', 0))
        user_skills = data.get('skills', [])
        user_degree = data.get('degree', '')
        user_location = data.get('location', 'Any')
        interested_company = data.get('interested_company', None)
        consent = data.get('consent_given', False)
        student_id = data.get('student_id', 'N/A')
        reg_no = data.get('registration_no', 'N/A')
        student_name = data.get('name', 'Anonymous')
        interested_company = data.get('interested_company')

        if consent:
            if interested_company:
                # User clicked "Focus Similar" (High Interaction Score: 3.0)
                log_to_timeline(student_id, reg_no, student_name, interested_company, 'focus_similar_search', 3.0)
            else:
                # User ran a general profile search (Base Interaction Score: 1.0)
                log_to_timeline(student_id, reg_no, student_name, 'General Search', 'profile_execution', 1.0)
        
        if not user_skills or len(user_skills) == 0:
            return jsonify({"error": "zero_skills"}), 400
        
        df_merged['Minimum_CGPA'] = pd.to_numeric(df_merged['Minimum_CGPA'], errors='coerce').fillna(0)
        df_filtered = df_merged[df_merged['Minimum_CGPA'] <= user_cgpa].copy()
        
        if df_filtered.empty:
            return jsonify({"content_based": [], "hybrid": [], "dl_based": [], "overall": []})

        user_profile_text = " ".join(user_skills) + " " + user_degree
        vectorizer = TfidfVectorizer()
        all_profiles = [user_profile_text] + df_filtered['Internship_Profile'].tolist()
        tfidf_matrix = vectorizer.fit_transform(all_profiles)
        
        content_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        df_filtered['Content_Score'] = content_sim
        
        CONTENT_WEIGHT = 0.7
        COLLAB_WEIGHT = 0.3
        DL_WEIGHT = 0.3
        
        df_filtered['Final_Hybrid_Score'] = (df_filtered['Content_Score'] * CONTENT_WEIGHT) + (df_filtered['Collab_Score'] * COLLAB_WEIGHT)
        df_filtered['Final_DL_Score'] = (df_filtered['Content_Score'] * CONTENT_WEIGHT) + (df_filtered['DL_Score'] * DL_WEIGHT)
        
        # --- NEW: CALCULATE 4TH ENGINE (OVERALL CONSENSUS) ---
        df_filtered['Final_Overall_Score'] = (df_filtered['Content_Score'] + df_filtered['Final_Hybrid_Score'] + df_filtered['Final_DL_Score']) / 3
        
        if interested_company:
            target_row = df_filtered[df_filtered['Company_Name'] == interested_company]
            if not target_row.empty:
                target_domain = target_row.iloc[0]['Domain']
                domain_mask = df_filtered['Domain'] == target_domain
                df_filtered.loc[domain_mask, 'Content_Score'] += 0.15 
                df_filtered.loc[domain_mask, 'Final_Hybrid_Score'] += 0.15
                df_filtered.loc[domain_mask, 'Final_DL_Score'] += 0.15
                df_filtered.loc[domain_mask, 'Final_Overall_Score'] += 0.15
                
                df_filtered['Content_Score'] = df_filtered['Content_Score'].clip(upper=0.99)
                df_filtered['Final_Hybrid_Score'] = df_filtered['Final_Hybrid_Score'].clip(upper=0.99)
                df_filtered['Final_DL_Score'] = df_filtered['Final_DL_Score'].clip(upper=0.99)
                df_filtered['Final_Overall_Score'] = df_filtered['Final_Overall_Score'].clip(upper=0.99)

        if user_location and user_location != "Any":
            df_filtered['Is_Exact_Location'] = (df_filtered['Location'] == user_location).astype(int)
        else:
            df_filtered['Is_Exact_Location'] = 0 
            
        top_content = df_filtered.sort_values(by=['Is_Exact_Location', 'Content_Score'], ascending=[False, False]).head(5)
        top_hybrid = df_filtered.sort_values(by=['Is_Exact_Location', 'Final_Hybrid_Score'], ascending=[False, False]).head(5)
        top_dl = df_filtered.sort_values(by=['Is_Exact_Location', 'Final_DL_Score'], ascending=[False, False]).head(5)
        top_overall = df_filtered.sort_values(by=['Is_Exact_Location', 'Final_Overall_Score'], ascending=[False, False]).head(5)
        
        # Updated to optionally include raw scores for the line graph
        def format_results(dataframe, score_col, default_type, include_raw=False):
            res = []
            for _, row in dataframe.iterrows():
                match_percentage = int(row[score_col] * 100)
                if match_percentage <= 0: continue
                    
                display_type = "📍 Location Priority" if row['Is_Exact_Location'] == 1 else default_type
                
                raw_skills = f"{row['Skill_1']}, {row['Skill_2']}, {row['Skill_3']}".split(',')
                unique_skills = []
                for s in raw_skills:
                    cleaned_skill = s.strip()
                    if cleaned_skill and cleaned_skill not in unique_skills:
                        unique_skills.append(cleaned_skill)
                clean_skills_str = ", ".join(unique_skills)
                
                data_dict = {
                    'title': f"{row['Domain']} Intern",
                    'company': str(row['Company_Name']),
                    'location': str(row['Location']),
                    'score': match_percentage,
                    'type': display_type,
                    'rating': f"{row.get('Overall_Rating', 'N/A')} ⭐",
                    'req_skills': clean_skills_str,
                    'duration': str(row.get('Duration', 'N/A'))
                }
                
                if include_raw:
                    data_dict['content_raw'] = int(row['Content_Score'] * 100)
                    data_dict['hybrid_raw'] = int(row['Final_Hybrid_Score'] * 100)
                    data_dict['dl_raw'] = int(row['Final_DL_Score'] * 100)
                    
                res.append(data_dict)
            return res

        return jsonify({
            "content_based": format_results(top_content, 'Content_Score', 'Strict Content Match'),
            "hybrid": format_results(top_hybrid, 'Final_Hybrid_Score', 'Statistical Hybrid Match'),
            "dl_based": format_results(top_dl, 'Final_DL_Score', 'PyTorch DL Match'),
            "overall": format_results(top_overall, 'Final_Overall_Score', 'Consensus Match', True)
        })

    except Exception as e:
        print("\n" + "="*40)
        print("❌ CRASH IN RECOMMENDATION ENGINE:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        company = data.get('company', 'Unknown')
        title = data.get('title', 'Unknown')
        feedback_val = data.get('feedback', 'Unknown')
        engine_type = data.get('engine_type', 'Unknown')
        
        file_path = os.path.join(BASE_DIR, "feedback_logs.csv")
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Company', 'Title', 'Engine_Type', 'Feedback'])
            
            writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), company, title, engine_type, feedback_val])
            
        print(f"📝 Feedback Logged [{engine_type}]: {company} -> {feedback_val}")
        return jsonify({"status": "success"})
        
    except Exception as e:
        print("❌ ERROR LOGGING FEEDBACK:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/insights', methods=['GET'])
def get_insights():
    try:
        # 1. Catch the search query from the frontend
        search_query = request.args.get('search', '').lower().strip()
        
        file_path = os.path.join(BASE_DIR, "user_interactions_timeline.csv")
        if not os.path.exists(file_path):
            file_path = 'user_interactions_timeline.csv'
            
        if not os.path.exists(file_path):
            return jsonify({"status": "empty", "feed": [], "graph": {}})
        
        df = pd.read_csv(file_path, on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        col_map = {}
        for c in df.columns:
            clean_c = c.strip().lower().replace('_', ' ')
            if 'company' in clean_c: col_map[c] = 'Company_Selected'
            elif 'student id' in clean_c: col_map[c] = 'Student_ID'
            elif 'reg' in clean_c: col_map[c] = 'Registration_No'
            elif 'name' in clean_c: col_map[c] = 'Student_Name'
            elif 'time' in clean_c or 'date' in clean_c: col_map[c] = 'Timestamp'

        df.rename(columns=col_map, inplace=True)

        required_cols = ['Student_ID', 'Registration_No', 'Student_Name', 'Company_Selected', 'Timestamp']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'Legacy Data'

        df = df[required_cols].copy()
        df = df.dropna(subset=['Company_Selected'])

        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        junk_words = ['', 'nan', 'n/a', 'unknown', 'none', 'general search', 'undefined', 'null']
        df_valid = df[~df['Company_Selected'].str.lower().isin(junk_words)].copy()

        if df_valid.empty:
            return jsonify({"status": "empty", "feed": [], "graph": {}})

        # Build Graph Data
        raw_counts = df_valid['Company_Selected'].value_counts().head(10).to_dict()
        company_counts = {str(k): int(v) for k, v in raw_counts.items()}

        for col in ['Student_ID', 'Registration_No', 'Student_Name', 'Timestamp']:
            df_valid.loc[df_valid[col].str.lower().isin(junk_words), col] = 'Legacy Record'

        # 🚨 SERVER-SIDE SEARCH: Filter by Student Name if the user is typing
        if search_query:
            df_valid = df_valid[df_valid['Student_Name'].str.lower().str.contains(search_query, na=False)]

        # 🚨 CRASH PREVENTION: Limit feed to the 10 most recent matches!
        raw_feed = df_valid.tail(10).iloc[::-1].to_dict(orient='records')
        
        clean_feed = []
        for row in raw_feed:
            clean_row = {}
            for key, val in row.items():
                val_str = str(val).strip()
                if val_str.endswith('.0'):
                    val_str = val_str[:-2]
                if pd.isna(val) or val_str.lower() == 'nan':
                    clean_row[key] = "N/A"
                else:
                    clean_row[key] = val_str
            clean_feed.append(clean_row)

        return jsonify({
            "status": "success",
            "feed": clean_feed, 
            "graph": company_counts
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)


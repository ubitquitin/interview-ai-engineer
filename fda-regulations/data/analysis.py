import json
import os
import pandas as pd
import re
import html
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. Data Acquisition & Engineering
def load_and_process(file_path):
    """Loads and processes raw FDA warning letter data for analysis.

    Performs data cleaning, feature engineering, and datetime parsing to prepare
    data for downstream analysis and visualization.

    Args:
        file_path (str): Path to JSONL file containing raw warning letter data

    Returns:
        pd.DataFrame: Processed DataFrame with cleaned metadata, engineered features
                     (letter_length, word_count, violation_count, year), and structured columns
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            
            # Clean HTML entities from metadata and content (handles &quot;, &amp;, etc.)
            clean_metadata = {k: (html.unescape(v) if isinstance(v, str) else v) 
                             for k, v in record['metadata'].items()}
            clean_content = html.unescape(record['content'])
            
            entry = {
                **clean_metadata,
                'content': clean_content,
                'letter_length': len(clean_content),
                'word_count': len(clean_content.split())
            }
            
            # Feature Engineering: Count violations (numbered items like "\n1. ")
            violations = re.findall(r'\n\d+\.\s', clean_content)
            entry['violation_count'] = len(violations) if violations else 0
            
            # Extract Year
            entry['year'] = pd.to_datetime(entry['issue_date']).year
            data.append(entry)
            
    return pd.DataFrame(data)

def get_main_department(office_name):
    """Normalizes and categorizes FDA issuing office names into main departments.

    Args:
        office_name (str): Raw issuing office name from warning letter metadata

    Returns:
        str: Standardized department name (e.g., "Center for Drug Evaluation and Research")
             or "Uncategorized" if office name is empty/unknown
    """
    # Standardize to title case and remove extra spaces
    name = office_name.strip()

    if not office_name or pd.isna(office_name):
        return "Uncategorized"
    
    # 1. Handle Center-level offices
    if "Tobacco" in name:
        return "Center for Tobacco Products"
    if "Veterinary" in name:
        return "Center for Veterinary Medicine"
    if "Pharmaceutical" in name:
        return "Div. of Pharmaceutical Quality Operations"
    if "Biologic" in name:
        return "Div. of Biological Products Operations"
    if "Drug" in name:
        return "Center for Drug Evaluation and Research"
    if "Food" in name:
        return "Center for Food Safety and Applied Nutrition"
    if "Human" in name or "HUMAN" in name:
        return "Div. of Human and Animal Food Operations"
    if "Imports" in name:
        return "Division of Imports"
    if "Medical Device" in name:
        return "Div. of Medical Device and Radiological Health"
    
    return name # Fallback for anything else


# Load the data
df = load_and_process('warning_letters_raw.jsonl')

# Create analysis directory
analysis_dir = "analysis"
os.makedirs(analysis_dir, exist_ok=True)

# 2. Summary Statistics
print("--- Summary Statistics ---")
stats = df[['letter_length', 'word_count', 'violation_count']].describe()
print(stats)

print("\n--- Violations by Issuing Office ---")
office_totals = df.groupby('issuing_office')['violation_count'].sum().sort_values(ascending=False)
print(office_totals)

# 3. Visualizations

# A. Static Visual: Distribution of Letter Lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['letter_length'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Warning Letter Lengths (Characters)')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.savefig(f'{analysis_dir}/letter_length_dist.png')
plt.close()

# B. Static Visual: Top 15 Offenders
top_offenders = df.sort_values('violation_count', ascending=False).head(15)
plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_offenders, 
    x='violation_count', 
    y='company_name', 
    palette='viridis'
)
plt.title('Top 15 Companies by FDA Violation Count', fontsize=14)
plt.xlabel('Number of Violations', fontsize=12)
plt.ylabel('') 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(f'{analysis_dir}/top_violations_per_company.png')
plt.close()

# C. Updated Interactive Component: Issuing Office Clusters
# Aggregating data by office to see who is most active/strict
# Create a new column for the cleaned department names
df['main_department'] = df['issuing_office'].apply(get_main_department)

# Aggregate by the new CLEANED department
office_df = df.groupby('main_department').agg(
    num_letters=('company_name', 'count'),
    total_violations=('violation_count', 'sum'),
    avg_letter_length=('letter_length', 'mean')
).reset_index()

# Update the scatter plot to use 'main_department'
fig = px.scatter(
    office_df, 
    x="num_letters", 
    y="total_violations", 
    size="avg_letter_length",
    color="main_department",
    hover_name="main_department",
    text="main_department",
    title="FDA Department Clusters: Volume vs. Violations",
    labels={
        "num_letters": "Total Warning Letters Sent", 
        "total_violations": "Total Violations Identified",
        "issuing_office": "Issuing Office",
        "avg_letter_length": "Avg Letter Length"
    },
    template="plotly_white"
)

# Adjust label position so they don't overlap bubbles
fig.update_traces(textposition='top center')

fig.write_html(f"{analysis_dir}/office_clusters_analysis.html")

print(f"\nProcessing Complete. Created files in '{analysis_dir}/':")
print("- letter_length_dist.png")
print("- top_violations_per_company.png")
print("- office_clusters_analysis.html")
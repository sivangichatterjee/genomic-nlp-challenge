import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


#------------------------------------A. Dataset--------------------------------------
####Read the csv####
df = pd.read_csv("texts.csv")

nlp=spacy.load('en_core_web_sm')

print(f"Loaded {len(df)} documents.")
print("Ready to start extraction!")

#----------------------B. Entity & Relation Extraction----------------------------
####Entity Extraction####

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    #variants and genes
    variants = re.findall(r'rs\d+', text)
    
    genes = re.findall(r'\b[A-Z0-9]{3,8}\b', text)
    noise_genes = {'DNA', 'RNA', 'SNP', 'GWAS', 'MRI', 'IGAP', 'UK', 'USA', 'CI', 'OR', 'AD', 'SD'} 
    genes = [g for g in genes if g not in noise_genes]
    diseases = []

    disease_terms = [
        "Stroke", "Depression", "Dementia", "Schizophrenia",
        "Diabetes", "Obesity", "Hypertension", "Anxiety", "Autism"
    ]
    
    # words that separate two distinct diseases (even if capitalized)
    splitters = ["Between", "With", "And", "In", "Of", "To", "For", "Vs"]
    
    # words that indicate not a disease
    bad_suffixes = ["Overlap", "Interaction", "Study", "Analysis", "Polymorphism", 
                    "Variation", "Consortium", "Biobank", "Group", "Data", "Cohort", "Risk", "Integrity"]

    doc = nlp(text)

    for chunk in doc.noun_chunks: 
        # Word + keyword
        rule_keyword = re.findall(r'\b[A-Z][a-zA-Z\']+\s+(?:disease|syndrome|disorder|cancer|tumor|diabetes|stroke)\b', chunk.text, flags=re.IGNORECASE)    
        diseases.extend(rule_keyword)

        pattern = r"\b(" + "|".join(disease_terms) + r")\b"
        matches = re.findall(pattern, chunk.text, flags=re.IGNORECASE)
        diseases.extend(matches)

        # capitalized chains
        rule_caps = re.findall(r'\b[A-Z][a-zA-Z\']+(?:\s+[A-Z][a-zA-Z\']+)+\b', chunk.text)
        
        for d in rule_caps:
            # 1. SPLIT long phrases like "Overlap Between ALS"
            temp_string = d
            for splitter in splitters:
                # Add pipe around splitters
                temp_string = temp_string.replace(f" {splitter} ", " | ")
            
            pieces = temp_string.split(" | ")

            for piece in pieces:
                piece = piece.strip()
                # skip if too short
                if len(piece) < 3: continue
                
                # skip if it ends with a bad suffix 
                if any(piece.lower().endswith(bad.lower()) for bad in bad_suffixes):
                    continue 
                
                # if just a generic word, skip
                if piece in splitters or piece in bad_suffixes:
                    continue

                diseases.append(piece)

    return {
        "variants": list(set(variants)),
        "genes": list(set(genes)),
        "diseases": list(set(diseases))
    }

# Print to check
df['entities'] = df['Text'].apply(extract_entities)
print(df[['Text', 'entities']])

####Relation extraction####
def extract_relations(row):
  text=row['Text']
  entities=row['entities']

  relation_keywords = [
        "associated with","disease-associated","related", "linked to", "increases risk", 
        "decreases risk", "affects", "cause", "leads to", "mutation in"
    ]

  matched_relations = [rel for rel in relation_keywords if rel in text]

  return {
      "text-id": row['Id'],
      "variants": entities.get('variants', []),
      "genes": entities.get('genes', []),
      "diseases": entities.get('diseases', []),
      "relations": matched_relations,
      "evidence_span": text
  }

df['relations'] = df.apply(extract_relations, axis=1)
for rel in df['relations']:
    if rel is not None:
        print(rel)

####Save curated results####
df['relations'] = df.apply(extract_relations, axis=1)

relations_list = [rel for rel in df['relations'] if rel is not None]
with open("curated_results.json", "w", encoding="utf-8") as f:
    json.dump(relations_list, f, indent=2, ensure_ascii=False)

#----------------------- C. Topic Grouping / Clustering--------------------------

# Creating the TF-IDF Vectorizer
# max_features=1000 keeps only the top 1000 most important words
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fitting text data
X = vectorizer.fit_transform(df['Text'])

print("Text has been vectorized into a matrix of shape:", X.shape)

###Method 1. KMeans Clustering####
# Defining number of clusters 
num_clusters = 5  

# Creating and running the K-Means algorithm
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Assiging the cluster (topic) ID back to  dataframe
df['topic_cluster'] = kmeans.labels_

print("\nClustering complete. Documents assigned to topics:")
print(df[['Id', 'Text', 'topic_cluster']])


####Plotting####

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['Text'])

# Reducing dimensions from 1000s down to 2 (x and y coordinates)
pca = PCA(n_components=2)
scatter_coords = pca.fit_transform(X.toarray()) # .toarray() for sparse matrices

# Adding  new coordinates to dataframe
df['x_coord'] = scatter_coords[:, 0]
df['y_coord'] = scatter_coords[:, 1]

# Creating the plot
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x='x_coord',
    y='y_coord',
    hue='topic_cluster',  # Color the dots based on the topic ID
    palette='viridis',    # A color-blind friendly color palette
    s=150,                
    alpha=0.8,            
    legend='full'
)

plt.title('2D Visualization of Topic Clusters (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True, alpha=0.3)

# Saving the plot 
plt.savefig('topic_clusters_visualization.png')
print("\nPlot created and saved as 'topic_clusters_visualization.png'")
plt.show()


###Plot 2###
# Checking if 'topic_cluster' exists in dataframe
if 'topic_cluster' not in df.columns:
    print("Error: 'topic_cluster' column not found. Please run K-Means step first.")
else:
    # Create the plot
    plt.figure(figsize=(8, 5))
    
    # Count the number of documents in each cluster
    cluster_counts = df['topic_cluster'].value_counts().sort_index()
    
    sns.barplot(
        x=cluster_counts.index, 
        y=cluster_counts.values, 
        palette='viridis'
    )
    
    plt.title('Document Count per Topic Cluster')
    plt.xlabel('Topic Cluster ID')
    plt.ylabel('Number of Documents')
    
    # Saving the plot 
    plt.savefig('topic_cluster_counts.png')
    print("Bar chart created and saved as 'topic_cluster_counts.png'")
    plt.show()



###BONUS: Method 2. HYBRID extraction function (ML model + Rules) + Sentence Transformer ####
import spacy
import re
import pandas as pd

# Loading the model
nlp_sci = spacy.load("en_core_sci_sm")
print("scispacy model loaded.")


# Defining the HYBRID extraction function (ML model + Rules)
def extract_entities_hybrid(text):
    
    if not isinstance(text, str):
        return {"variants": [], "genes": [], "diseases": []}

    #  Using Regex for variants 
    variants = re.findall(r'rs\d+', text)
    
    genes = []
    diseases = []
    
    # Keywords to check against found entities
    disease_keywords = ['disease', 'syndrome', 'cancer', 'disorder', 'tumor', 'hypertension']
    
    # Using ML model (scispacy) to find ENTITY candidates
    doc = nlp_sci(text)
    
    for ent in doc.ents:
        # Checking if the found entity is a gene or disease

        if any(keyword in ent.text.lower() for keyword in disease_keywords):
            diseases.append(ent.text)
            
        # Checking for gene-like pattern: 3-6 ALL CAPS
        elif re.fullmatch(r'\b[A-Z][A-Z0-9]{2,6}\b', ent.text):
            genes.append(ent.text)
            
    return {
        "variants": list(set(variants)),
        "genes": list(set(genes)),
        "diseases": list(set(diseases))
    }

# Applying the new HYBRID function
print("Running HYBRID extraction")
df['entities'] = df['Text'].apply(extract_entities_hybrid)

# 4. Printing the new results
print("\n--- Extraction Results (Hybrid) ---")
pd.set_option('display.max_colwidth', 200)
print(df[['entities']].head())


####Sentence-Transformer####

model = SentenceTransformer('all-MiniLM-L6-v2')

# Get your list of texts
texts = df['Text'].tolist()

# Creating Embeddings
print("Creating sentence embeddings")
X_embeddings = model.encode(texts)
print("Embeddings created with shape:", X_embeddings.shape)

# K-Means Cluster 
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_embeddings)

# Assigning new clusters back to DF
df['topic_cluster_bonus'] = kmeans.labels_

# 6. Plotting same PCA as before
pca = PCA(n_components=2)
scatter_coords = pca.fit_transform(X_embeddings)

df['x_bonus'] = scatter_coords[:, 0]
df['y_bonus'] = scatter_coords[:, 1]

# Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x='x_bonus',
    y='y_bonus',
    hue='topic_cluster_bonus', 
    palette='viridis',
    s=150
)
plt.title('BONUS: Topic Clusters (Sentence Transformer Embeddings)')
plt.savefig('topic_clusters_bonus.png')
plt.show()

        
#  Genomic Text Curation & Topic Grouping

**A lightweight NLP pipeline that extracts curatable entities (Genes, Variants, Diseases) from unstructured text and groups research papers into thematic clusters.**

## 🚀 Key Features
* **Entity Extraction:** Identifies `rsID` variants and gene symbols using Regex and spaCy.
* **Topic Modeling:** Clusters documents using TF-IDF and K-Means to identify high-level research themes.
* **Curated Output:** Exports structured JSON data ready for human review.

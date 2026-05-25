import requests
import json

url = 'https://query.wikidata.org/sparql'
query = """
SELECT DISTINCT ?concept ?conceptLabel
WHERE {
  ?concept (wdt:P361|wdt:P279)* / (wdt:P31)? wd:Q11660.
  SERVICE wikibase:label { 
    bd:serviceParam wikibase:language "en". 
  }
}
"""

print(f"The path for this query is : {query.splitlines()[3]}")

headers = {
    'User-Agent': 'MyDataFetcherBot/1.0 (rdesboscs@gmail.com)',
    'Accept': 'application/sparql-results+json'
}

response = requests.get(url, params={'query': query}, headers=headers)

if response.status_code == 200:
    data = response.json()
    
    with open('../dict_testing.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent = 4)
        
else:
    print("error")

mit_glossary = [
    "Adaptable AI",
    "Agentic AI",
    "Auditable AI",
    "Autoencoders",
    "Bayesian Networks",
    "Classic Machine Learning",
    "Computer Vision",
    "Continual Learning",
    "Convolutional Neural Networks",
    "CNN",
    "CNNs",
    "Data-Driven AI",
    "Deep Neural Networks",
    "DNN",
    "DNNs"
    "Ethical AI",
    "Expert Systems",
    "Explainable AI",
    "XAI",
    "Fairness",
    "Fine-Tuning",
    "GANS",
    "Generative Adversarial Networks",
    "General AI",
    "AGI",
    "Generative AI",
    "Hybrid AI",
    "Knowledge-Driven AI",
    "Knowledge Graphs",
    "Ontologies",
    "Hallucination",
    "Knowledge Representation and Reasoning (KR&R)",
    "Large Language Models (LLMs)",
    "Large Models",
    "LSTM (Long Short-Term Memory)",
    "Machine Learning (ML)",
    "Multimodal AI",
    "Narrow AI",
    "Natural Language Processing (NLP)",
    "Neural Networks",
    "Neuro-Symbolic AI",
    "Optimization",
    "Planning",
    "Pre-Trained Models",
    "Predictive AI",
    "Prompt Engineering",
    "RAG",
    "Recurrent Neural Networks (RNNs)",
    "Reinforcement Learning",
    "Responsible AI",
    "Robotics",
    "Scalable AI",
    "Scheduling",
    "Structure Learning (Graph Learning)",
    "Sub-Symbolic (Connectionist) AI",
    "Supervised Learning",
    "Symbolic AI",
    "Tokens",
    "Transfer Learning",
    "Transformers",
    "Unsupervised Learning",
    "Zero-shot Learning"
]

iabac_glossary = [
    "Machine Learning",
    "Supervised Learning",
    "Unsupervised Learning",
    "Reinforcement Learning",
    "Semi-supervised Learning",
    "Self-supervised Learning",
    "Natural Language Processing",
    "Text Analysis / Classification",
    "Sentiment Analysis",
    "Machine Translation",
    "Chatbots and Virtual Assistants",
    "Text Summarization",
    "Grammar Checking",
    "Computer Vision",
    "Image Recognition",
    "Object Detection",
    "Image Generation",
    "Image Segmentation",
    "Robotics",
    "Manufacturing Automation",
    "Autonomous Vehicles",
    "Healthcare Robotics",
    "Expert Systems",
    "Knowledge Base",
    "Inference Engine",
    "User Interface",
    "Fuzzy Logic",
    "Rule-Based Decision Making",
    "Approximate Reasoning",
    "Neural Networks and Deep Learning",
    "Speech Recognition",
    "Image Classification",
    "Predictive Analytics",
    "Transformer Models / Large Language Models (LLMs)"
]


llm_dictionary = [
    # --- General AI & Machine Learning ---
    "Machine Learning", "ML",
    "Deep Learning", "DL",
    "Reinforcement Learning", "RL",
    "Artificial Neural Network",
    "Neural Network",
    "Multilayer Perceptron", "MLP",
    "Support Vector Machine", "SVM",
    "K-Nearest Neighbors", "KNN",
    "Principal Component Analysis", "PCA",
    "Stochastic Gradient Descent", "SGD",
    "Rectified Linear Unit", "ReLU",
    "Reinforcement Learning from Human Feedback", "RLHF",
    "Low-Rank Adaptation", "LoRA",
    "Mixture of Experts", "MoE",
    "Backpropagation",
    "Gradient Descent",
    "Hyperparameter",
    "Underfitting",
    "Overfitting",
    "Transfer Learning",
    "Fine-tuning",
    "Zero-shot learning",
    "Few-shot learning",
    "Batch Normalization",
    "Cross-Entropy Loss",

    # --- Computer Vision (CV) ---
    "Computer Vision", "CV",
    "Convolutional Neural Network", "CNN",
    "Region-based Convolutional Neural Network", "R-CNN",
    "You Only Look Once", "YOLO",
    "Single Shot MultiBox Detector", "SSD",
    "Generative Adversarial Network", "GAN",
    "Vision Transformer", "ViT",
    "Optical Character Recognition", "OCR",
    "Intersection over Union", "IoU",
    "Mean Average Precision", "mAP",
    "Scale-Invariant Feature Transform", "SIFT",
    "Histogram of Oriented Gradients",
    "Feature Pyramid Network", "FPN",
    "Neural Radiance Field", "NeRF",
    "Contrastive Language-Image Pretraining", "CLIP",
    "Image Classification",
    "Object Detection",
    "Semantic Segmentation",
    "Instance Segmentation",
    "Image Generation",
    "Bounding Box",
    "Feature Extraction",
    "Edge Detection",
    "Data Augmentation",
    "Super-resolution",
    "Pose Estimation",
    "Facial Recognition",
    "Point Cloud",

    # --- Natural Language Processing (NLP) ---
    "Natural Language Processing", "NLP",
    "Natural Language Understanding", "NLU",
    "Natural Language Generation", "NLG",
    "Large Language Model", "LLM",
    "Recurrent Neural Network", "RNN",
    "Long Short-Term Memory", "LSTM",
    "Gated Recurrent Unit", "GRU",
    "Bidirectional Encoder Representations from Transformers", "BERT",
    "Generative Pre-trained Transformer", "GPT",
    "Named Entity Recognition",
    "Part-of-Speech",
    "Term Frequency-Inverse Document Frequency", "TF-IDF",
    "Machine Translation", "MT",
    "Masked Language Modeling", "MLM",
    "Retrieval-Augmented Generation", "RAG",
    "Bilingual Evaluation Understudy", "BLEU",
    "Recall-Oriented Understudy for Gisting Evaluation", "ROUGE",
    "Sequence-to-Sequence", "Seq2Seq",
    "Transformer",
    "Word Embedding",
    "Tokenization",
    "Lemmatization",
    "Stemming",
    "Sentiment Analysis",
    "Attention Mechanism",
    "Self-Attention",
    "Prompt Engineering",
    "Corpus",
    "Stop Words",
    "Vector Database",
    "Embeddings"
]

ai_dict = {}
with open("../dict_testing.json", "r", encoding = "utf-8") as f:
    ai_dict_data = json.load(f)

count = 0
print(f"\n---------\n")
print(f"DICT SNIPPET\n")
for concept in ai_dict_data["results"]["bindings"]:
    if count <= 25:
        print(f"{concept["conceptLabel"]["value"]}")
        count += 1
    ai_dict[concept["conceptLabel"]["value"].lower()] = 0
print(f"\n---------\n")

# print(json.dumps(ai_dict, ensure_ascii = False, indent = 4))

recall = 0
print(f"RECALLED TERMS\n")
for term in llm_dictionary :
    if term.lower() in ai_dict :
        recall += 1
        print(f"{term}")
print(f"\n")

print(f"This dictionnary contains {len(ai_dict)} and has a recall of {recall/len(llm_dictionary)} on the preselected glossary")
import networkx as nx
from community import community_louvain
import json
import numpy as np
import math
from tqdm import tqdm


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

for word in llm_dictionary:
    ai_dict[word.casefold()] = 0

def tfidf(readme: str)->np.array:
    v_readme = np.zeros(len(ai_dict))
    for i, word in enumerate(ai_dict):
        v_readme[i] = readme.count(word) * ai_dict[word]
    return v_readme

def cos_similarity(v1: np.array, v2: np.array)->float:
    if np.linalg.norm(v1)*np.linalg.norm(v2) == 0:
        return 0.0
    return np.sum(v1*v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def analyse_labels(graph_filepath, json_filepath):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)
    # N is the number of nodes
    N = len(repo_data)
    graph = nx.read_gexf(graph_filepath)
    for _, _, data in graph.edges(data=True):
        if 'weight' in data:
            data['weight'] = float(data['weight'])
    
    communities = community_louvain.best_partition(graph, weight = 'weight', resolution = 1.0)

    list_commu = {}
    for repo, commu_id in communities.items():
        if commu_id not in list_commu:
            list_commu[commu_id] = []
        list_commu[commu_id].append(repo)

    readme_corpus = {}

    for repo in repo_data:
        readme_corpus[repo.get("nameWithOwner")] = repo.get("readme_text")

    for word in tqdm(ai_dict.keys(), desc = "Computing IDF"):
        for readme in readme_corpus.values():
            if word in readme:
                ai_dict[word] += 1
        if ai_dict[word] > 0 :
            ai_dict[word] = math.log10(N/ai_dict[word])

    repo_labels = {}
    commu_labels = {}

    for commu_id, commu in tqdm(list_commu.items(), desc = "Assigning labels to the communities"):
        best_labels = np.zeros(N)
        for repo in commu:
            v_readme = tfidf(readme_corpus[repo])
            repo_labels[repo] = []
            for label in np.argsort(v_readme)[::-1][ : 3]:
                repo_labels[repo].append(label)
                best_labels[label] += 1
        commu_labels[commu_id] = []
        for label in np.argsort(best_labels)[::-1][ : 5]:
            commu_labels[commu_id].append(label)

        translated_labels = [llm_dictionary[label] for label in commu_labels[commu_id]]
        print(f"\n\nCommunity {commu_id}, top 5 labels are : {translated_labels}\n\n")
    
    # Purity calculation
    purity = 0
    for commu_id, commu in list_commu.items():
        ground_truth = set(commu_labels[commu_id][:3])
        for repo in commu:
            if len(set(repo_labels[repo]) & ground_truth) > 0:
                purity += 1
    purity /= N
    print(f"\n\n\nEmpiric purity : {purity}\n\n")




    

if __name__ == "__main__":
    
    analyse_labels("outputs/graphs/repo/clean_fork_network.gexf", "data/raw/repo_raw_data_fork.json")
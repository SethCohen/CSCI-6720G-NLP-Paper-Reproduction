import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os
from dotenv import load_dotenv
import logging
from sentence_transformers import SentenceTransformer

# Constants
TOP_N = 10
GPT_MODEL = "gpt-4o"
USE_SENTENCE_TRANSFORMER = True
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables...")
load_dotenv()
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)
logger.info("Environment variables loaded.")

def load_data(train_path, dev_path):
    """
    Load training and development data from specified file paths.
    
    Args:
        train_path (str): Path to the training data file.
        dev_path (str): Path to the development data file.
    
    Returns:
        tuple: DataFrames containing training and development data.
    """
    logger.info(f"Loading data from {train_path} and {dev_path}...")
    train_data = pd.read_csv(train_path, sep='\t')
    dev_data = pd.read_csv(dev_path, sep='\t')
    logger.info("Data loaded successfully.")
    return train_data, dev_data

def create_combined_data(train_data):
    """
    Create a combined representation of Source, Change, and Target columns.
    
    Args:
        train_data (DataFrame): Training data containing 'Source', 'Change', and 'Target' columns.
    
    Returns:
        list: Combined representation of the data.
    """
    return train_data.apply(lambda row: f"Source: {row['Source']} Change: {row['Change']} Target: {row['Target']}", axis=1).tolist()

def calculate_similarity_tfidf(combined_train_data, combined_source):
    """
    Calculate cosine similarity using TF-IDF between the combined source and training data.
    
    Args:
        combined_train_data (list): Combined representation of the training data.
        combined_source (str): Combined representation of the source and change.
    
    Returns:
        np.ndarray: Similarity scores for each training example.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_train_data)
    source_vec = vectorizer.transform([combined_source])
    return cosine_similarity(source_vec, tfidf_matrix).flatten()

def calculate_similarity_sentence_transformer(combined_train_data, combined_source):
    """
    Calculate cosine similarity using Sentence Transformer between the combined source and training data.
    
    Args:
        combined_train_data (list): Combined representation of the training data.
        combined_source (str): Combined representation of the source and change.
    
    Returns:
        np.ndarray: Similarity scores for each training example.
    """
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    train_embeddings = model.encode(combined_train_data, convert_to_tensor=True)
    source_embedding = model.encode([combined_source], convert_to_tensor=True)
    return cosine_similarity(source_embedding, train_embeddings).flatten()

def find_relevant_examples(train_data, source_sentence, change):
    """
    Find the most relevant examples from the training data based on cosine similarity.
    
    Args:
        train_data (DataFrame): Training data.
        source_sentence (str): The source sentence for which we want relevant examples.
        change (str): The change to be applied to the source sentence.
    
    Returns:
        DataFrame: The most relevant examples from the training data.
    """
    logger.info("Calculating similarities...")
    combined_train_data = create_combined_data(train_data)
    combined_source = f"Source: {source_sentence} Change: {change}"
    
    if USE_SENTENCE_TRANSFORMER:
        similarities = calculate_similarity_sentence_transformer(combined_train_data, combined_source)
    else:
        similarities = calculate_similarity_tfidf(combined_train_data, combined_source)
    
    relevant_indices = similarities.argsort()[-TOP_N:][::-1]
    logger.info(f"Found {TOP_N} most relevant examples.")
    return train_data.iloc[relevant_indices]

def create_prompt(examples, source_sentence, change):
    """
    Generate a prompt based on relevant examples and a new sentence to predict.
    
    Args:
        examples (DataFrame): Relevant examples from training data.
        source_sentence (str): The source sentence for which we need a prediction.
        change (str): The change to be applied to the source sentence.
    
    Returns:
        str: A formatted prompt to send to the OpenAI model.
    """
    logger.info("Creating prompt with relevant examples...")
    prompt = """
    You are a helpful assistant with a strong background in linguistics.
    
    This is a linguistic puzzle. Below are example sentences in a foreign language and sets of changes to apply to them.
    The examples are followed by the problem sentence and desired change.
    
    Your task is to look closely at the example sentences and to change the sentence correctly.
    """
    for idx, row in examples.iterrows():
        prompt += f"""
        Example [{idx}]:
        Sentence: {row['Source']}
        Change(s): {row['Change']}
        Answer: {row['Target']}
        """
    prompt += f"""
    Here is the problem. Give just the answer, and nothing else.
    Sentence: {source_sentence}
    Change(s): {change}
    """
    logger.info("Prompt created successfully.")
    return prompt

def get_prediction(prompt):
    """
    Request a prediction from the OpenAI API based on the generated prompt.
    
    Args:
        prompt (str): The prompt to send to the model.
    
    Returns:
        str: The model's answer.
    """
    logger.info("Requesting prediction from OpenAI API...")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    logger.info("Prediction received from OpenAI API.")
    return response.choices[0].message.content.strip()

def save_predictions(dev_data, predictions, output_path):
    """
    Save predictions to the output file.
    
    Args:
        dev_data (DataFrame): Development data with predictions added.
        predictions (list): List of predictions.
        output_path (str): Path to save the output file.
    """
    logger.info("Saving predictions to output file...")
    dev_data['Predicted Target'] = pd.Series(predictions)
    dev_data.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Predictions saved to {output_path}")

def main(train_path, dev_path, output_path):
    """
    Main function to process development data and create predictions.
    
    Args:
        train_path (str): Path to the training data file.
        dev_path (str): Path to the development data file.
        output_path (str): Path to save the output with predictions.
    """
    logger.info("Starting main process...")
    train_data, dev_data = load_data(train_path, dev_path)
    predictions = []
    
    for idx, row in dev_data.iterrows():
        logger.info(f"Processing row {idx + 1}/{len(dev_data)}...")
        source_sentence = row['Source']
        change = row['Change']
        examples = find_relevant_examples(train_data, source_sentence, change)
        prompt = create_prompt(examples, source_sentence, change)
        prediction = get_prediction(prompt)
        predictions.append(prediction)
    
    save_predictions(dev_data, predictions, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error("Usage: python prompt.py train.tsv dev.tsv results.tsv")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

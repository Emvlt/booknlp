from pathlib import Path

import numpy as np
import csv
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from booknlp.booknlp import BookNLP
from scipy.interpolate import make_interp_spline

from database import add_characters_to_db

nltk.download('vader_lexicon', quiet=True)

def analyze_character_arc(tokens_df, entities_df, target_character, num_segments):
    print(f"\n--- Analyzing Arc for {target_character} ---")
    # 1. Reconstruct all sentences in the book
    # Group tokens by sentence_ID and join them into readable strings
    sentences = tokens_df.groupby('sentence_ID')['word'].apply(lambda x: ' '.join(x.astype(str))).to_dict()
    
    # 2. Find the primary Coref ID for the target character
    # We look for the most frequent coref ID associated with the character's name
    char_matches = entities_df[entities_df['text'].str.contains(target_character, case=False, na=False)]
    
    char_coref_id = char_matches['COREF'].value_counts().idxmax()
    
    # 3. Find all sentence IDs where this character is mentioned
    char_start_tokens = entities_df[entities_df['COREF'] == char_coref_id]['start_token'].tolist()
    char_sentence_ids = tokens_df[tokens_df['token_ID_within_document'].isin(char_start_tokens)]['sentence_ID'].unique()
    
    # 4. Analyze Sentiment over Narrative Time
    sia = SentimentIntensityAnalyzer()
    arc_data = []
    
    for sent_id in char_sentence_ids:
        text = sentences.get(sent_id, "")
        sentiment_score = sia.polarity_scores(text)['compound'] # Score from -1.0 (negative) to 1.0 (positive)
        arc_data.append({'sentence_ID': sent_id, 'sentiment': sentiment_score, 'text': text})
        
    arc_df = pd.DataFrame(arc_data)
    
    # 5. Divide the character's appearances into chronological segments (e.g., 10 "chapters")
    arc_df['narrative_segment'] = pd.qcut(arc_df['sentence_ID'], q=num_segments, labels=False, duplicates='drop') + 1
    
    # Calculate the average sentiment for each segment
    segment_sentiment = arc_df.groupby('narrative_segment')['sentiment'].mean()
    
    return segment_sentiment.values
        


def get_characters(entities_filepath):
    # Read the tab-separated output file
    df = pd.read_csv(entities_filepath, sep='\t')
    
    # Filter strictly for People (PER) who are Proper Nouns (PROP)
    characters = df[(df['cat'] == 'PER') & (df['prop'] == 'PROP')]
    
    top_characters = characters['text'].value_counts().head(5)
    corefs = characters['COREF'].value_counts().head(5)
    return list(top_characters.keys()), list(corefs)
    

current_file_path = Path(__file__).parent.resolve()

bookname = 'pride_and_prejudice'

# Input file to process
input_file = current_file_path.joinpath(f"{bookname}.txt")

# Output directory to store resulting files in
output_directory = current_file_path.joinpath(f"{bookname}")

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id = current_file_path.joinpath(f"{bookname}")

entities_fp = current_file_path.joinpath(f"{bookname}.entities")
tokens_fp   = current_file_path.joinpath(f"{bookname}.tokens")
if not entities_fp.is_file():
    model_params={
        "pipeline":"entity,quote,supersense,event,coref", 
        "model":"big"
    }
    
    booknlp=BookNLP("en", model_params)
    booknlp.process(input_file, output_directory, book_id)

tokens_df = pd.read_csv(tokens_fp, sep='\t', quoting=csv.QUOTE_NONE)
entities_df = pd.read_csv(entities_fp, sep='\t')

# Fill NaN values to prevent errors during string joining
tokens_df['word'] = tokens_df['word'].fillna('')
entities_df['text'] = entities_df['text'].fillna('')

characters, characters_id = get_characters(entities_fp)

for char in characters:
    num_segments = 20
    arc = analyze_character_arc(tokens_df, entities_df, char, num_segments)
    
    x = np.linspace(0, num_segments, num_segments)
    B_spline_coeff  = make_interp_spline(x, arc)
    X_Final = np.linspace(x.min(), x.max(), 500)
    y = B_spline_coeff(X_Final)
    
    
    
    plt.plot(y, label=char)
plt.legend()
plt.savefig(f'test_{bookname}')
    
    
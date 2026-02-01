import pandas as pd
from tqdm import tqdm
from .explaination import generate_explaination
from .judgement import generate_judgement, extract_judgement_prediction
from sklearn.metrics import classification_report
from .utils import get_HF_pipeline
import os
import datetime

# Mapping from dataframe columns to full trait names
TRAIT_MAP = {
    'cEXT': 'Extraversion',
    'cNEU': 'Neuroticism',
    'cAGR': 'Agreeableness',
    'cCON': 'Conscientiousness',
    'cOPN': 'Openness'
}

# The traits to evaluate, corresponding to dataframe columns
BIG_FIVE_TRAITS = list(TRAIT_MAP.keys())

def pado_predict(text: str, trait_short: str, model_name: str, prompt_type='pado', max_new_tokens: int = None, pipeline = None, progress_verbose: bool = True, item_index: int = 0, total_items: int = 0, log_filepath: str = None):
    if trait_short not in TRAIT_MAP:
        raise ValueError(f"Invalid trait_short: {trait_short}. Expected one of {BIG_FIVE_TRAITS}")
    
    trait_full = TRAIT_MAP[trait_short]

    # This function is designed to process one text at a time.
    # The `evaluate_dataframe` function contains its own, separate batching logic.

    # Generate explanations for both high and low induction scenarios.
    if progress_verbose:
        tqdm.write(f"[{item_index + 1}/{total_items}] Trait {trait_full}: High explanation...")
    explanation_high = generate_explaination(trait_full, text, model_name, induce='high', prompt_type=prompt_type, max_new_tokens=max_new_tokens, pipeline=pipeline, log_filepath=log_filepath)
    
    if progress_verbose:
        tqdm.write(f"[{item_index + 1}/{total_items}] Trait {trait_full}: Low explanation...")
    explanation_low = generate_explaination(trait_full, text, model_name, induce='low', prompt_type=prompt_type, max_new_tokens=max_new_tokens, pipeline=pipeline, log_filepath=log_filepath)

    # Generate a judgement based on the two explanations.
    if progress_verbose:
        tqdm.write(f"[{item_index + 1}/{total_items}] Trait {trait_full}: Judgement...")
    # Since this function handles a single text, generate_judgement will return a single string.
    judgement_text = generate_judgement(trait_full, text, explanation_high, explanation_low, model_name, max_new_tokens=max_new_tokens, pipeline=pipeline, log_filepath=log_filepath)

    # Extract the final prediction from the judgement text.
    prediction = extract_judgement_prediction(judgement_text)

    # extract_judgement_prediction already returns 'high', 'low', or 'Unknown'.
    return prediction

def evaluate_dataframe(df: pd.DataFrame, model_name: str, text_column: str = "text", max_new_tokens: int = None, verbose: bool = True, log_llm_output: bool = True) -> pd.DataFrame:
    results_df = df.copy()
    
    log_filepath = None
    if log_llm_output:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filepath = os.path.join(log_dir, f"llm_output_{timestamp}.log")

    # Create result directory and generate report path
    report_dir = "result"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(report_dir, f"CR_{timestamp}.txt")

    # Write header to the report file
    total_records = len(results_df)
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Records: {total_records}\n\n")

    is_hf_pipeline = not (model_name.startswith("gpt") or model_name.startswith("gemini"))

    if is_hf_pipeline:
        pipeline = get_HF_pipeline(model_name, max_new_tokens=max_new_tokens if max_new_tokens is not None else 512)
    else:
        pipeline = None
        
    for trait_short in BIG_FIVE_TRAITS:
        pred_column_name = f'{trait_short}_pred'
        trait_full = TRAIT_MAP[trait_short]
        
        all_texts = results_df[text_column].tolist()
        all_predictions = []

        progress_desc = f"Evaluating {trait_full} ({model_name})"

        for text in tqdm(all_texts, desc=progress_desc):
            # Explanations
            explanations_high = generate_explaination(trait_full, text, model_name, induce='high', prompt_type='pado', max_new_tokens=max_new_tokens, pipeline=pipeline, log_filepath=log_filepath)
            explanations_low = generate_explaination(trait_full, text, model_name, induce='low', prompt_type='pado', max_new_tokens=max_new_tokens, pipeline=pipeline, log_filepath=log_filepath)

            # Judgements
            judgement_text = generate_judgement(trait_full, text, explanations_high, explanations_low, model_name, max_new_tokens=max_new_tokens, pipeline=pipeline, log_filepath=log_filepath)
            
            # Predictions
            prediction = extract_judgement_prediction(str(judgement_text))
            
            if prediction == 'high':
                all_predictions.append('high')
            elif prediction == 'low':
                all_predictions.append('low')
            else:
                all_predictions.append('unknown')

        results_df[pred_column_name] = all_predictions

        # If ground truth exists, calculate and print accuracy for the trait.
        if trait_short in results_df.columns:
            y_true = results_df[trait_short]
            y_pred = results_df[pred_column_name]

            report = classification_report(y_true, y_pred, zero_division=0)
            with open(report_path, 'a') as f:
                f.write(f"Classification Report for {trait_full}:\n")
                f.write(report)
                f.write("\n\n")
            
            correct_predictions = (y_true == y_pred).sum()
            total_predictions = len(results_df)
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                print(f"Accuracy for {trait_full}: {accuracy:.2%}")

    return results_df
import pandas as pd
from tqdm import tqdm
from .explaination import generate_explaination
from .judgement import generate_judgement, extract_judgement_prediction
from sklearn.metrics import classification_report
from .utils import get_HF_pipeline

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

def pado_predict(text: str, trait_short: str, model_name: str, prompt_type='pado', max_new_tokens: int = None, pipeline = None, progress_verbose: bool = False, item_index: int = 0, total_items: int = 0):
    if trait_short not in TRAIT_MAP:
        raise ValueError(f"Invalid trait_short: {trait_short}. Expected one of {BIG_FIVE_TRAITS}")
    
    trait_full = TRAIT_MAP[trait_short]

    # Generate explanations for both high and low induction scenarios.
    if progress_verbose:
        tqdm.write(f"[{item_index + 1}/{total_items}] Trait {trait_full}: High explanation...")
    explanation_high = generate_explaination(trait_full, text, model_name, induce='high', prompt_type=prompt_type, max_new_tokens=max_new_tokens, pipeline=pipeline)
    
    if progress_verbose:
        tqdm.write(f"[{item_index + 1}/{total_items}] Trait {trait_full}: Low explanation...")
    explanation_low = generate_explaination(trait_full, text, model_name, induce='low', prompt_type=prompt_type, max_new_tokens=max_new_tokens, pipeline=pipeline)

    # Generate a judgement based on the two explanations.
    if progress_verbose:
        tqdm.write(f"[{item_index + 1}/{total_items}] Trait {trait_full}: Judgement...")
    judgement_text = generate_judgement(trait_full, text, explanation_high, explanation_low, model_name, max_new_tokens=max_new_tokens, pipeline=pipeline)

    # Extract the final prediction from the judgement text.
    prediction = extract_judgement_prediction(judgement_text)

    # Standardize the output to lowercase 'high' or 'low'.
    if 'High' in prediction:
        return 'high'
    elif 'Low' in prediction:
        return 'low'
    else:
        return 'unknown'

def evaluate_dataframe(df: pd.DataFrame, model_name: str, report_path: str = "classification_report.txt", text_column: str = "text", max_new_tokens: int = None, verbose: bool = False, batch_size: int = 1) -> pd.DataFrame:
    results_df = df.copy()
    
    if report_path:
        with open(report_path, 'w') as f:
            pass  # Create/Clear the file
    
    is_hf_pipeline = not model_name.startswith("gpt")

    if is_hf_pipeline:
        pipeline = get_HF_pipeline(model_name, max_new_tokens=max_new_tokens if max_new_tokens is not None else 512, batch_size=batch_size)
    else:
        pipeline = None
        
    for trait_short in BIG_FIVE_TRAITS:
        pred_column_name = f'{trait_short}_pred'
        trait_full = TRAIT_MAP[trait_short]

        if is_hf_pipeline:
            # --- BATCHED IMPLEMENTATION for HF pipelines ---
            texts = results_df[text_column].tolist()
            
            with tqdm(total=3, desc=f"Evaluating {trait_full} (Batched)") as pbar:
                pbar.set_description(f"Trait {trait_full}: High explanations")
                explanations_high = generate_explaination(trait_full, texts, model_name, induce='high', prompt_type='pado', max_new_tokens=max_new_tokens, pipeline=pipeline)
                pbar.update(1)

                pbar.set_description(f"Trait {trait_full}: Low explanations")
                explanations_low = generate_explaination(trait_full, texts, model_name, induce='low', prompt_type='pado', max_new_tokens=max_new_tokens, pipeline=pipeline)
                pbar.update(1)

                pbar.set_description(f"Trait {trait_full}: Judgements")
                judgement_texts = generate_judgement(trait_full, texts, explanations_high, explanations_low, model_name, max_new_tokens=max_new_tokens, pipeline=pipeline)
                pbar.update(1)

            predictions_raw = [extract_judgement_prediction(jt) for jt in tqdm(judgement_texts, desc="Extracting predictions")]
            
            predictions = []
            for p in predictions_raw:
                if 'High' in p:
                    predictions.append('high')
                elif 'Low' in p:
                    predictions.append('low')
                else:
                    predictions.append('unknown')
            
            results_df[pred_column_name] = predictions
        else:
            # --- SEQUENTIAL IMPLEMENTATION for GPT or other models ---
            if verbose:
                predictions = []
                total_items = len(results_df)
                for i, row in tqdm(results_df.iterrows(), total=total_items, desc=f"Evaluating {trait_full}"):
                    text = row[text_column]
                    pred = pado_predict(
                        text, trait_short, model_name, max_new_tokens=max_new_tokens, pipeline=pipeline, 
                        progress_verbose=True, item_index=i, total_items=total_items
                    )
                    predictions.append(pred)
                results_df[pred_column_name] = predictions
            else:
                tqdm.pandas(desc=f"Evaluating {trait_full}")
                results_df[pred_column_name] = results_df[text_column].progress_apply(
                    lambda text: pado_predict(text, trait_short, model_name, max_new_tokens=max_new_tokens, pipeline=pipeline)
                )

        # If ground truth exists, calculate and print accuracy for the trait.
        if trait_short in results_df.columns:
            y_true = results_df[trait_short]
            y_pred = results_df[pred_column_name]

            if report_path:
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

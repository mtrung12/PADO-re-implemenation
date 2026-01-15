import pandas as pd
from tqdm import tqdm
from .explaination import generate_explaination
from .judgement import generate_judgement, extract_judgement_prediction
from sklearn.metrics import classification_report

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

def pado_predict(text: str, trait_short: str, model_name: str, prompt_type='pado', max_new_tokens: int = None):
    if trait_short not in TRAIT_MAP:
        raise ValueError(f"Invalid trait_short: {trait_short}. Expected one of {BIG_FIVE_TRAITS}")
    
    trait_full = TRAIT_MAP[trait_short]

    # Generate explanations for both high and low induction scenarios.
    explanation_high = generate_explaination(trait_full, text, model_name, induce='high', prompt_type=prompt_type, max_new_tokens=max_new_tokens)
    explanation_low = generate_explaination(trait_full, text, model_name, induce='low', prompt_type=prompt_type, max_new_tokens=max_new_tokens)

    # Generate a judgement based on the two explanations.
    judgement_text = generate_judgement(trait_full, explanation_high, explanation_low, model_name, max_new_tokens=max_new_tokens)

    # Extract the final prediction from the judgement text.
    prediction = extract_judgement_prediction(judgement_text)

    # Standardize the output to lowercase 'high' or 'low'.
    if 'High' in prediction:
        return 'high'
    elif 'Low' in prediction:
        return 'low'
    else:
        return 'unknown'

def evaluate_dataframe(df: pd.DataFrame, model_name, report_path: str = "classification_report.txt", text_column: str = "text", max_new_tokens: int = None) -> pd.DataFrame:
    results_df = df.copy()

    if report_path:
        with open(report_path, 'w') as f:
            pass  # Create/Clear the file
    
    for trait_short in BIG_FIVE_TRAITS:
        tqdm.pandas(desc=f"Evaluating {TRAIT_MAP[trait_short]}")
        
        # Create a new column for the predictions of the current trait.
        pred_column_name = f'{trait_short}_pred'
        
        # Apply the prediction function to the text column.
        results_df[pred_column_name] = results_df[text_column].progress_apply(
            lambda text: pado_predict(text, trait_short, model_name, max_new_tokens=max_new_tokens)
        )
        
        # If ground truth exists, calculate and print accuracy for the trait.
        if trait_short in results_df.columns:
            y_true = results_df[trait_short]
            y_pred = results_df[pred_column_name]

            if report_path:
                report = classification_report(y_true, y_pred, zero_division=0)
                with open(report_path, 'a') as f:
                    f.write(f"Classification Report for {TRAIT_MAP[trait_short]}:\n")
                    f.write(report)
                    f.write("\n\n")
            
            correct_predictions = (y_true == y_pred).sum()
            total_predictions = len(results_df)
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                print(f"Accuracy for {TRAIT_MAP[trait_short]}: {accuracy:.2%}")

    return results_df

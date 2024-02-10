import nltk
import numpy as np
import evaluate

def compute_metrics(eval_pred, tokenizer):
    metric_rouge = evaluate.load("rouge", rouge_types=["rouge1", "rouge2", "rougeL"])
    metric_bertscore = evaluate.load("bertscore")
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    rouge = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bertscore = metric_bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    return {'rouge':rouge, 'bertscore':np.mean(bertscore['f1'])}
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

def load_model():
    base_model = "meta-llama/Llama-3.1-8B"
    peft_model = "llk010502/llama3.1-8B-financial_sentiment"
    
    model_name = 'meta-llama/Llama-3.1-8B'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        trust_remote_code=True,
        device_map='cuda'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = PeftModel.from_pretrained(model, peft_model)
def scorer(prompts, model, tokenizer):

    sentiments = ['positive', 'neutral', 'negative']

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits # shape:[batch_size, output_length, vocab_size]

    print(logits.shape)
    # Get logits for the last token (the next token to be predicted)
    last_token_logits = logits[:, -1, :]
    last_token_logits = last_token_logits.to(torch.float32)
    probabilities = torch.softmax(last_token_logits, dim=-1)  # Shape: [batch_size, vocab_size]
    sentiment_scores = []
    for i in range(len(prompts)):

        sentiments_prob = [probabilities[i, tokenizer.convert_tokens_to_ids(s)].item() for s in sentiments]

        # Standarized Positive - Standarized Negative
        sentiment_score = (sentiments_prob[0] - sentiments_prob[2])/sum(sentiments_prob)
        sentiment_scores.append(sentiment_score)
    return sentiment_scores

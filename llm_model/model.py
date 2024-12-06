from transformers import AutoModelForSequenceClassification, AutoTokenizer
def load_model():
    # Replace this with your logic to load the LLM
    model_name = "your-repo/your-model-name"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return {"model": model, "tokenizer": tokenizer}

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
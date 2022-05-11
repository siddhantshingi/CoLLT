from transformers import BertForSequenceClassification, DistilBertForSequenceClassification
from transformers import BertTokenizerFast, DistilBertTokenizerFast

def get_encoder(num_classes, model='distilbert', device='cpu'):
    if model=='bert':
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_classes, # The number of output labels.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        tokenizer = BertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    elif model=='distilbert':
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_classes, # The number of output labels.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    return model.to(device), tokenizer
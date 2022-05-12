from transformers import LongformerForSequenceClassification, BertForSequenceClassification, DistilBertForSequenceClassification, RobertaForSequenceClassification,RobertaConfig
from transformers import LongformerTokenizerFast, BertTokenizerFast, DistilBertTokenizerFast, RobertaTokenizer


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
    elif model=='longformer':
        model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                           gradient_checkpointing=False,
                                                           attention_window = 512)
        tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 1024)
    elif model=='roberta':
      config = RobertaConfig.from_pretrained('roberta-base')
      tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
      model = RobertaForSequenceClassification(config)


    return model.to(device), tokenizer

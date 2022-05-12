import torch
import os.path as osp
import losses as L
import augmentors as A
import models as M
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm
from torch.optim import Adam, AdamW
import datasets
from contrast_models import WithinEmbedContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
import numpy as np
import pickle
import io

#Use CUDA if available
device_name = 'cuda' if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print(device_name)


#Load data and split into train and test  
#TODO create a validation set  
# train_data, test_data = datasets.load_dataset('imdb', split =['train', 'test'], 
#                                             cache_dir='./data/')
# num_dev = 10
# train_data, test_data_dev = train_data.select(list(np.random.randint(len(train_data), size=num_dev))), test_data.select(list(np.random.randint(len(test_data), size=num_dev)))

with open("data.pickle","rb") as f:
    train_data_dev = pickle.load(f)

with open("data_test.pickle","rb") as f:
    test_data_dev = pickle.load(f)

with open("data_val.pickle","rb") as f:
    val_data_dev = pickle.load(f)

val_data_dev = val_data_dev.select(list(np.random.randint(len(val_data_dev), size=100)))
check_save_data = val_data_dev.select(list(np.random.randint(len(val_data_dev), size=1)))
#Check the data is balanced or not
print(Counter(train_data_dev['label']))
print(Counter(test_data_dev['label']))
print(Counter(val_data_dev['label']))

#Load Bert model
model_name='distilbert'
model, tokenizer = M.get_encoder(num_classes=2, model=model_name, device=device_name)

#Tokenizer 1 for contrastive learning
def tokenization_contrastive(batched_text):
    return tokenizer(batched_text['text'], padding = False, truncation=False)
train_data_cl = train_data_dev.map(tokenization_contrastive, batched = True, batch_size = len(train_data_dev))
check_save_data_cl = check_save_data.map(tokenization_contrastive, batched = True, batch_size = len(check_save_data))

#Tokenizer 2 for classification downstream task
def tokenization_classification(batched_text):
    return tokenizer(batched_text['text'], padding = 'max_length', truncation=True, max_length = 512)
train_data_dev = train_data_dev.map(tokenization_classification, batched = True, batch_size = len(train_data_dev))
test_data_dev = test_data_dev.map(tokenization_classification, batched = True, batch_size = len(test_data_dev))
val_data_dev = val_data_dev.map(tokenization_classification, batched = True, batch_size = len(val_data_dev))

#Define Encoder class
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, input_dim=768, hidden_dim=768, output_dim=1536):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x):
        aug1, aug2 = self.augmentor
        ids1, mask1 = aug1(x, device_name)
        ids2, mask2 = aug2(x, device_name)
        z1 = self.encoder(ids1, mask1).last_hidden_state[:,0,:]
        z2 = self.encoder(ids2, mask2).last_hidden_state[:,0,:]
        return z1, z2
    
    def predict(self, x):
        aug1, aug2 = self.augmentor
        ids1, mask1 = aug1(x, device_name)
        ids2, mask2 = aug2(x, device_name)
        z1 = self.encoder(ids1, mask1).last_hidden_state[:,0,:]
        z2 = self.encoder(ids2, mask2).last_hidden_state[:,0,:]
        return (z1 + z2)/2

#TODO: Generalize for any number of augmentations
aug1 = A.RandomSampling()
aug2 = A.RandomSampling()

encoder_model = Encoder(encoder=getattr(model, model_name), augmentor=(aug1, aug2)).to(device)
contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

optimizer = Adam(encoder_model.parameters(), lr=5e-4)
scheduler = LinearWarmupCosineAnnealingLR(
    optimizer=optimizer,
    warmup_epochs=400,
    max_epochs=4000)

def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2 = encoder_model.forward(data)
    loss = contrast_model(z1, z2)
    if torch.isnan(loss):
      print ('ERROR')
      return
    loss.backward()
    optimizer.step()
    return loss.item()

epoch = 1
batch_size = 5
with tqdm(total=epoch, desc='(T)') as pbar:
    for epoch in range(1, epoch + 1):
        # For each batch of training data...
        num_batches = int(len(train_data_cl)/batch_size) + 1

        for i in range(num_batches):
            end_index = min(batch_size * (i+1), len(train_data_cl))

            batch = train_data_cl[i*batch_size:end_index]

            if len(batch['text']) == 0: continue

            # print (batch)
            loss = train(encoder_model, contrast_model, batch, optimizer)
            scheduler.step()
            # break
        pbar.set_postfix({'loss': loss})
        pbar.update()



# Checkpointing: Save encoder model
torch.save(encoder_model, "cl_encoder.pt")

# Load encoder model
load_model = torch.load("cl_encoder.pt")
load_model.to(device)
load_model.eval()
print(encoder_model.predict(check_save_data_cl))
print(load_model.predict(check_save_data_cl))



# Freeze the model parameters
for param in getattr(model, model_name).parameters():
    param.requires_grad = False


def get_validation_performance(model, val_set, batch_size):
    # Put the model in evaluation mode
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0

    num_batches = int(len(val_set)/batch_size) + 1

    total_correct = 0
    total = 0
    with tqdm(total=epoch, desc='(V)') as pbar:
      for i in range(num_batches):

        end_index = min(batch_size * (i+1), len(val_set))

        batch = val_set[i*batch_size:end_index]
        
        if len(batch['text']) == 0: continue

        input_id_tensors = torch.tensor(batch['input_ids'])
        input_mask_tensors = torch.tensor(batch['attention_mask'])
        label_tensors = torch.tensor(batch['label'])
        
        # Move tensors to the GPU
        b_input_ids = input_id_tensors.to(device)
        b_input_mask = input_mask_tensors.to(device)
        b_labels = label_tensors.to(device)
          
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

          # Forward pass, calculate logit predictions.
          outputs = model(b_input_ids, 
                                  attention_mask=b_input_mask,
                                  labels=b_labels)
          loss = outputs.loss
          logits = outputs.logits
              
          # Accumulate the validation loss.
          total_eval_loss += loss.item()
          
          # Move logits and labels to CPU
          logits = logits.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()

          # Calculate the number of correctly labeled examples in batch
          pred_flat = np.argmax(logits, axis=1).flatten()
          labels_flat = label_ids.flatten()
          # print (labels_flat)
          # print (pred_flat)
          num_correct = np.sum(pred_flat == labels_flat)
          total_correct += num_correct
          total += len(labels_flat)
          
        pbar.set_postfix({'val_accuracy': total_correct / total})
        pbar.update()
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_correct / len(val_set)
    return avg_val_accuracy

batch_size = 50
optimizer = AdamW(model.parameters(),
                lr = 5e-5, # args.learning_rate - default is 5e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
epochs = 10
for epoch_i in range(0, epochs):
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    num_batches = int(len(train_data_dev)/batch_size) + 1

    with tqdm(total=num_batches, desc='(T)') as pbar:
      for i in range(num_batches):
        end_index = min(batch_size * (i+1), len(train_data_dev))

        batch = train_data_dev[i*batch_size:end_index]

        if len(batch['text']) == 0: continue

        input_id_tensors = torch.tensor(batch['input_ids'])
        input_mask_tensors = torch.tensor(batch['attention_mask'])
        label_tensors = torch.tensor(batch['label'])

        # Move tensors to the GPU
        b_input_ids = input_id_tensors.to(device)
        b_input_mask = input_mask_tensors.to(device)
        b_labels = label_tensors.to(device)

        # Clear the previously calculated gradient
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        outputs = model(b_input_ids, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits

        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Update parameters and take a step using the computed gradient.
        optimizer.step()

        pbar.set_postfix({'loss': loss.item()})
        pbar.update()
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set. Implement this function in the cell above.
    print(f"Total loss: {total_train_loss}")
    val_acc = get_validation_performance(model, val_set=val_data_dev, batch_size=batch_size*2)
    print(f"Validation accuracy: {val_acc}")
    
print("")
print("Training complete!")
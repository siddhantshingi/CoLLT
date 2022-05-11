import torch
import os.path as osp
import losses as L
import augmentors as A
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
import datasets
from contrast_models import WithinEmbedContrast
from transformers import BertTokenizer, BertModel
device = torch.device('cpu')


def get_encoder():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    return model, tokenizer

def get_model_inputs(x):
    ids = torch.tensor(x['input_ids']).to(device)
    mask = torch.tensor(x['attention_mask']).to(device)
    return ids, mask

## TODO: identity augmentation
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, input_dim=768, hidden_dim=768, output_dim=1536):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain=1.4)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain=1.4)

    def forward(self, x):
        aug1, aug2 = self.augmentor
        ids1, mask1 = aug1(x)
        ids2, mask2 = aug2(x)
        z1 = self.encoder(ids1, mask1)
        z2 = self.encoder(ids2, mask2)
        return z1, z2

    def project(self, x):
        return self.fc2(F.relu(self.fc1(x[1]))), self.fc2(F.relu(self.fc1(x[2])))


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    train_data, test_data = datasets.load_dataset('imdb', split =['train', 'test'], 
                                             cache_dir='./data/')
    
    encoder, tokenizer = get_encoder()

    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding = 'max_length', truncation=True, max_length = 512)
    train_data_dev = train_data_dev.map(tokenization, batched = True, batch_size = len(train_data_dev))
    test_data_dev = test_data_dev.map(tokenization, batched = True, batch_size = len(test_data_dev))

    # aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    # aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    encoder_model = Encoder(encoder=encoder, augmentor=(aug1, aug2)).to(device)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=5e-4)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=400,
        max_epochs=4000)

    with tqdm(total=4000, desc='(T)') as pbar:
        for epoch in range(1, 4001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
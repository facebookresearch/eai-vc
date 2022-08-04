import torch
import torchvision
from tqdm import tqdm
import hydra

import eaif_models.utils.load_model


def compute_accuracy(dataloader, model, config):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    model = model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            inp, tar = data
            inp, tar = inp.to(config['device']), tar.to(config['device'])
            # calculate outputs by running images through the network
            out = model(inp)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(out.data, 1)
            total += tar.size(0)
            correct += (predicted == tar).sum().item()
    return 100.0 * correct / total


# add linear layer to model
class ClassifierModel(torch.nn.Module):
    def __init__(self, base_model, embedding_dim, out_dim):
        super(ClassifierModel, self).__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim
        self.fc_cls = torch.nn.Linear(embedding_dim, out_dim)
    def forward(self, x):
        return self.fc_cls(self.base_model(x))
    

# pre-loaded embedding dataset
class FrozenEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, size):
        self.embeddings = embeddings
        self.labels = labels
        self.size = size
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
    

def precompute_embeddings(model, dataset, config):
    # data parallel mode to use both GPUs
    model = model.to(config['device'])
    model = torch.nn.DataParallel(model)
    model = model.train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            batch_x, batch_y = data
            z = model(batch_x).detach().to('cpu')
            for idx in range(batch_x.shape[0]):
                embeddings.append(z[idx])
                labels.append(batch_y[idx])
    return embeddings, labels


@hydra.main(config_path="conf", config_name="run_cifar_lin_probe")
def probe_model_eval(config):
    # Get base model, transform, and probing classifier
    model = hydra.utils.instantiate(config["model"])
    embedding_dim = config["embedding_dim"]
    transform = getattr(eaif_models.utils.load_model, config["transform"])

    probe = torch.nn.Sequential(torch.nn.BatchNorm1d(embedding_dim),
                                torch.nn.Linear(embedding_dim, 10))

    raw_tr_data = torchvision.datasets.CIFAR10(root=config["data_root"], train=True, download=True, transform=transform)
    raw_te_data = torchvision.datasets.CIFAR10(root=config["data_root"], train=False, download=True, transform=transform)
    
    tr_z, tr_y  = precompute_embeddings(model, raw_tr_data, config)
    te_z, te_y  = precompute_embeddings(model, raw_te_data, config)
    tr_data = FrozenEmbeddingDataset(tr_z, tr_y, len(tr_z))
    te_data = FrozenEmbeddingDataset(te_z, te_y, len(te_z))

    trainloader = torch.utils.data.DataLoader(tr_data, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(te_data, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    probe = probe.to(config['device'])
    probe = torch.nn.DataParallel(probe)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=config['lr'])
    
    # train loop
    for epoch in range(config['num_epochs']):
        running_loss = 0.0
        probe = probe.train()
        for idx, data in tqdm(enumerate(trainloader)):
            z, y = data
            z, y = z.to(config['device']), y.to(config['device'])

            optimizer.zero_grad()
            out  = probe(z)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # after each epoch compute train and test accuracy
        train_accuracy = compute_accuracy(trainloader, probe, config)
        test_accuracy  = compute_accuracy(testloader,  probe, config)
        print('===========================')
        print(f'epoch : %i | train accuracy : %3.2f | test accuracy : % 3.2f' % (epoch, train_accuracy, test_accuracy))
        print('===========================')



if __name__ == "__main__":
    probe_model_eval()

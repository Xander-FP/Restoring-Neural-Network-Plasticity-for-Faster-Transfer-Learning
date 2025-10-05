from Helpers.general_nn import general_batch_lightning, general_batch_lightning_autoencode
from torch.utils.data.sampler import SubsetRandomSampler
from Helpers.data_prep import DataPrep 
from torchvision.transforms import v2
from torch import optim, nn
import torch.nn.functional as F
import lightning as L
import torch

class LitModel(L.LightningModule):
    def __init__(self, encoder, training_parameters, resnet_size=None, project_name=None):
        super().__init__()
        self.training_parameters = training_parameters
        self.encoder = encoder
        self.resnet_size = resnet_size
        self.save_hyperparameters({'project_name': project_name})

    # training_step defines the train loop.
    # it is independent of forward
    def training_step(self, batch, batch_idx):
        if self.encoder.classify:
            return self.training_step_classification(batch, batch_idx)
        else:
            return self.training_step_autoencoder(batch, batch_idx)
    
    def training_step_classification(self, batch, batch_idx):
        loss, acc, batch_size = general_batch_lightning(self, batch, nn.CrossEntropyLoss())
        self.log("train_accuracy", acc, batch_size=batch_size) 
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def training_step_autoencoder(self, batch, batch_idx):
        loss, acc, batch_size = general_batch_lightning_autoencode(self, batch, nn.MSELoss())
        self.log("train_accuracy", acc, batch_size=batch_size) 
        self.log("train_loss", loss, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.encoder.classify:
            self.validation_step_classification(batch, batch_idx)
        else:
            self.validation_step_autoencoder(batch, batch_idx)

    def validation_step_classification(self, batch, batch_idx):
        loss, acc, batch_size = general_batch_lightning(self, batch, nn.CrossEntropyLoss())
        self.log("val_accuracy", acc, batch_size=batch_size) 
        self.log("val_loss", loss, batch_size=batch_size)

    def validation_step_autoencoder(self, batch, batch_idx):
        loss, acc, batch_size = general_batch_lightning_autoencode(self, batch, nn.MSELoss())
        self.log("val_accuracy", acc, batch_size=batch_size) 
        self.log("val_loss", loss, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        loss, acc, batch_size = general_batch_lightning(self, batch, nn.CrossEntropyLoss())
        # self.log("resnet_size", int(self.resnet_size))
        self.log("test_accuracy", acc, batch_size=batch_size) 
        self.log("test_loss", loss, batch_size=batch_size)

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.training_parameters['learning_rate'], weight_decay=self.training_parameters['weight_decay'])
        optimizer = optim.SGD(self.parameters())
        return optimizer
    
###################################################### MY CUSTOM FUNCTIONS ##############################################################

    def load_data(self, dataset_path, dataset_type, batch_size):
        train_set, valid_set = DataPrep.get_train_datasets(data_dir=dataset_path, dataset_type=dataset_type)
        test_set = DataPrep.get_test_datasets(data_dir=dataset_path, dataset_type=dataset_type)
        self.__prepare_data(train_set=train_set, valid_set=valid_set, test_set=test_set)

        train_idx, valid_idx = DataPrep.split_train_valid_idx(indexes=list(range(len(train_set))), valid_size=0.2)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

        return train_loader, valid_loader, test_loader
        
    def replace_classifier(self, num_classes):
        self.encoder.to('cpu')
        
        if ('Resnet' in self.encoder.get_type()):
            self.encoder.get_model().fc = nn.Linear(self.encoder.in_features, num_classes)
        elif (self.encoder.get_type() == 'SimpleNet'): 
            self.encoder.get_model().classifier = nn.Linear(self.encoder.in_features, num_classes)
        elif (self.encoder.get_type() == 'MobileNet'):
            self.encoder.get_model().classifier[1] = nn.Linear(self.encoder.in_features, num_classes)
        elif (self.encoder.get_type() == 'VIT_B_16'):
            self.encoder.get_model().heads.head = nn.Linear(self.encoder.in_features, num_classes)
        self.encoder.to(self.device)
        
    def transfer_from_model(self, model):
        if model is not None:
            self.load_state_dict(model.state_dict(), strict=True)

    def SWR(self, percentage, init_strat):
        #count = 0
        
        for name, param in self.named_parameters():
            #count += 1
            #if count < 73: #Only modify Resnet 50 blocks 3 and 4
            #    continue
            decimal_percentage = int(percentage)/100
            avg_weight = torch.mean(param.data)
            total_weights = len(param.data[param.data == param.data])
            zeroed_weights = int(decimal_percentage * total_weights)
            flat = param.data.view(-1)
            _, indexes = torch.topk(abs(flat), zeroed_weights, largest=False)

            if 'bias' in name:
                if init_strat == 'BA':
                    flat[indexes] = avg_weight
                else:
                    flat[indexes] = 0
            
            # if '0' in name or '26' in name or '4' in name or '12' in name:
            if param.requires_grad and param.data.ndimension() > 1 and init_strat != 'BA' :
                # Case 1: Mean
                if init_strat == 'M': #or init_strat == 'BA':
                    flat[indexes] = avg_weight
                
                # Case 2: Mean + noise
                if init_strat == 'MN':
                    flat[indexes] = avg_weight + torch.randn_like(avg_weight) * avg_weight/10
                
                # Case 3: 0
                if init_strat == 'Z':
                    flat[indexes] = 0

                # Case 4: Initial distribution
                if init_strat == 'Dist':
                    flat[indexes] = torch.randn_like(avg_weight)

                if init_strat == 'Norm':
                    flat[indexes] = torch.normal(mean=0, std=1, size=(zeroed_weights,))

                if init_strat == 'Norm_small':
                    flat[indexes] = torch.normal(mean=0, std=0.2, size=(zeroed_weights,))

                if init_strat == 'Dist3':
                    flat[indexes] = torch.normal(mean=0, std=0.1, size=(zeroed_weights,))

    def calculate_gradient(self, train_loader, percentage, init_strat):
        self.encoder.get_model().zero_grad()
        for images, labels, _ in train_loader: 
            images = images.to(self.device)
            labels = labels.to(self.device)
            loss_function = nn.CrossEntropyLoss()
        
            outputs = self.encoder.get_model()(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            
        print("Gradient calculated")
        for name, param in self.named_parameters():
            if param.grad != None:
                decimal_percentage = int(percentage)/100
                avg_weight = torch.mean(param.data)
                total_weights = len(param.data[param.data == param.data])
                zeroed_weights = int(decimal_percentage * total_weights)
                weights_flat = param.data.view(-1)
                grad_flat = param.grad.view(-1)
                utility_flat = weights_flat * grad_flat 

                _, indexes = torch.topk(abs(utility_flat), zeroed_weights, largest=False)

                if 'bias' in name:
                    weights_flat[indexes] = 0
                # if 'weight' in name:
                # if '0' in name or '26' in name or '4' in name or '12' in name:
                if param.requires_grad and param.data.ndimension() > 1:
                # Case 1: Mean
                    if init_strat == 'M':
                        weights_flat[indexes] = avg_weight
                    
                    # Case 2: Mean + noise
                    if init_strat == 'MN':
                        weights_flat[indexes] = avg_weight + torch.randn_like(avg_weight) * avg_weight/10
                    
                    # Case 3: 0
                    if init_strat == 'Z':
                        weights_flat[indexes] = 0

                    # Case 4: Initial distribution
                    if init_strat == 'Dist':
                        weights_flat[indexes] = torch.randn_like(avg_weight)

                    if init_strat == 'Norm':
                        weights_flat[indexes] = torch.normal(mean=0, std=1, size=(zeroed_weights,))

                    if init_strat == 'Norm_small':
                        weights_flat[indexes] = torch.normal(mean=0, std=0.2, size=(zeroed_weights,))

    def __prepare_data(self, train_set, valid_set, test_set):
        result = DataPrep.compute_mean_std(train_set)
        result['max_height'] = 224
        result['max_width'] = 224
        normalize = v2.Normalize(
            mean= result['mean'],
            std= result['std'],
        )
        train_set.transform = v2.Compose([
            v2.Resize((result['max_height'],result['max_width'])),
            v2.ColorJitter(brightness=.5, hue=.3, saturation=.3, contrast=.3),
            v2.RandomVerticalFlip(),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(180), # Look at options
            v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # v2.RandomPerspective(p=0.2),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])
        valid_set.transform = v2.Compose([
            v2.Resize((result['max_height'],result['max_width'])),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])
        test_set.transform = v2.Compose([
            v2.Resize((result['max_height'],result['max_width'])),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])
        return train_set, valid_set, test_set
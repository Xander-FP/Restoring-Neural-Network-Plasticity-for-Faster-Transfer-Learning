from Models.SimpleNetFeatureExtractor import SimpleNetFeatureExtractor
from Models.MobileNetFeatureExtractor import MobileNetFeatureExtractor
from Models.ResnetsFeatureExtractor import Resnet18FeatureExtractor, Resnet34FeatureExtractor, Resnet50FeatureExtractor
from Models.ViTFeatureExtractor import ViTFeatureExtractor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers as pl_loggers
from lightning_trainer import LitModel
import torch.nn as nn
import lightning as L
import wandb
import torch
import sys

# **********GENERAL CONFIGURATIONS*********
# Set the default tensor float32 precision to high
torch.set_float32_matmul_precision('high')

if len(sys.argv) <= 1:
    print("Please provide a mode to run the script in. Valid modes are 'vit', 'cnn', 'ocr'")
    sys.exit()

if len(sys.argv) <= 2:
    print("Please provide a resnet size to run the script in. Valid sizes are '18', '34', '50'")
    sys.exit()
MODEL_NAME = sys.argv[2]

if len(sys.argv) <= 3:
    print("Please provide whether the model should be pretrained or not. Valid options are 'pretrained', 'not_pretrained'")
    sys.exit()

if sys.argv[3] == 'pretrained':
    PRETRAINED = True
else:
    PRETRAINED = False

if len(sys.argv) <= 4:
    print("Please provide a value for the patience")
    sys.exit()
PATIENCE = sys.argv[4]
PATIENCE_NUM = int(PATIENCE[1:]) 

if len(sys.argv) > 5:
    BASE = False
    UTILITY = sys.argv[5]
else:
    BASE = True

if len(sys.argv) > 6:
    PERCENTAGE = sys.argv[6]

if len(sys.argv) > 7:
    INIT_STRAT = sys.argv[7]

if len(sys.argv) > 8:
    CHECKPOINT_PROVIDED = True
    CHECKPOINT_PATH = sys.argv[8]
    print("Loading from checkpoint at path: ", CHECKPOINT_PATH)
    if len(sys.argv) > 9:
        run_id = sys.argv[9]
        print("Resuming run with id: ", run_id)
else:
    CHECKPOINT_PROVIDED = False

OPTIONS = {
        # **********GENERAL PARAMETERS*********
        'mode': sys.argv[1], # 'ocr', 'cnn', 'vit
        'resnet_size': sys.argv[2], # '6', '18', '34', '50'
        'pretrained': PRETRAINED,
        'should_tune': False,
        'generate_activation_maps': False,
        'log_options': {
                'wandb_logs': False, 
                'console_log': True
            }, 
        # **********DATA PARAMETERS*********
        'datasets' : [
            # {'path': '../datasets/DTD', 'classes': 47, 'name': 'DTD', 'epochs': 2000, 'dataset_type': 'Folder', 'patience': PATIENCE_NUM, 'autoencoder': False},
            # {'path': '../datasets/Food101/Food101', 'classes': 101, 'name': 'Food', 'epochs': 2000, 'dataset_type': 'Folder', 'patience': PATIENCE_NUM, 'autoencoder': False},
            # {'path': '../datasets/Brain_Tumor', 'classes': 4, 'name': 'Brain', 'epochs': 2000, 'dataset_type': 'Folder', 'patience': PATIENCE_NUM, 'autoencoder': False}
            {'path': '../Datasets/Brain_Tumor2', 'classes': 4, 'name': 'Brain', 'epochs': 2000, 'dataset_type': 'Folder', 'patience': PATIENCE_NUM, 'autoencoder': False}
            ],
        'result_dir': './Data/Results',
    }

# **********MODEL PARAMETERS*********
TRAINING_PARAMETERS = {
        'batch_size': 32,
        # 'learning_rate': 0.0003, # 0.0003 1e-3
        # 'weight_decay': 0.01, # 0.01
        # 'momentum': 0.9, # 0.9
    }

def train(CHECKPOINT_PROVIDED):  
    if OPTIONS['mode'] == 'ocr':
        params = {"nHidden": 256, "nClasses": len(OPTIONS['alphabet']), "imgH": 32, "nChannels": 1}
        model = CRNN(params)
        OPTIONS['loss_function'] = nn.CTCLoss()
    elif 'cnn' in OPTIONS['mode']:
        if OPTIONS['resnet_size'] == '18':
            model = Resnet18FeatureExtractor(pretrained=OPTIONS['pretrained'])
        elif OPTIONS['resnet_size'] == '34':
            model = Resnet34FeatureExtractor(pretrained=OPTIONS['pretrained'])
        elif OPTIONS['resnet_size'] == '50':
            model = Resnet50FeatureExtractor(pretrained=OPTIONS['pretrained'])
        elif OPTIONS['resnet_size'] == 'sim':
            model = SimpleNetFeatureExtractor(pretrained=OPTIONS['pretrained'])
        elif OPTIONS['resnet_size'] == 'mob':
            model = MobileNetFeatureExtractor(pretrained=OPTIONS['pretrained'])
        OPTIONS['loss_function'] = nn.CrossEntropyLoss()
    elif 'vit' in OPTIONS['mode']:
        model = ViTFeatureExtractor(pretrained=OPTIONS['pretrained'])
        # Efficient net ViT: https://arxiv.org/abs/2305.07027
        # https://arxiv.org/abs/2304.05350
        OPTIONS['loss_function'] = nn.CrossEntropyLoss()
    prev_model = None
    prev_was_auto = False
    trainer = None

    for dataset in OPTIONS['datasets']:
        dataset_path = dataset['path']
        dataset_name = dataset['name']
        dataset_type = dataset['dataset_type']
        num_classes = dataset['classes']
        max_epochs = dataset['epochs']
        dataset_patience = dataset['patience']
        autoencoder = dataset['autoencoder']

        # **********CALLBACKS*********
        auto_text = ''
        if autoencoder or prev_was_auto:
            auto_text = 'AUTO_'

        myPrependedWord = '' + PATIENCE + '_'
        if not BASE:
            myPrependedWord = myPrependedWord + UTILITY + '_' + str(PERCENTAGE) + '_' + INIT_STRAT + '_'

        project_name = myPrependedWord + auto_text + dataset_name +'_' + OPTIONS['mode'] + '-' + str(MODEL_NAME) + ('_Pre_' if PRETRAINED else '_NotPre_') + '_'.join(f"{dataset['name']}-{dataset['epochs']}" for dataset in OPTIONS['datasets'])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='./')
        wandb.finish()
        wandb_logger = None
        if CHECKPOINT_PROVIDED:
            wandb_logger = pl_loggers.WandbLogger(save_dir='./wandb_logs/', project = project_name, resume='allow', id=run_id)
        else:
            wandb_logger = pl_loggers.WandbLogger(save_dir='./wandb_logs/', project = project_name)
        early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=dataset_patience)

        # *********MODEL AND DATA*********
        model.classify = not autoencoder
        model_new = LitModel(model, TRAINING_PARAMETERS, resnet_size=OPTIONS['resnet_size'], project_name=project_name)
        train_loader, valid_loader, test_loader = model_new.load_data(dataset_path, dataset_type, TRAINING_PARAMETERS['batch_size'])
        # **********TRANSFER LEARNING STEPS************
        model_new.replace_classifier(num_classes)
        model_new.transfer_from_model(prev_model)
        # **********PLASTICITY CHANGES************
        if not BASE and UTILITY == 'W':
            model_new.SWR(PERCENTAGE, INIT_STRAT)
        if not BASE and UTILITY == 'WG':
            model_new.calculate_gradient(train_loader, PERCENTAGE, INIT_STRAT)
        # **********TRAINING*********
        trainer = L.Trainer(limit_train_batches=None, max_epochs=max_epochs, logger=[tb_logger, wandb_logger], callbacks=[early_stop_callback])
        if CHECKPOINT_PROVIDED:
            trainer.fit(model=model_new, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=CHECKPOINT_PATH)
            CHECKPOINT_PROVIDED = False
        else:
            trainer.fit(model=model_new, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        prev_model = model_new
        prev_was_auto = autoencoder
        # **********TESTING*********    
        if model.classify:
            trainer.test(dataloaders=test_loader, model=model_new)

train(CHECKPOINT_PROVIDED)


import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from preprocess_data import get_paths_and_labels, get_transforms, get_num_classes
from FER_Dataset import FERDataset
from Model import Net
from torchinfo import summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
LEARNING_RATE = 0.002
N_EPOCHS = 100
MOMENTUM = 0.9
WEIGHT_DECAY = 9e-4
MODEL_SAVE_DIR = 'trained_model'


def plot(train_losses, train_acc, test_losses, test_acc, label):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    axs[0].plot(test_losses, label='val loss')
    axs[0].plot(train_losses, label='train loss')
    axs[0].set_title("Loss")
    axs[1].plot(test_acc, label='val accuracy')
    axs[1].plot(train_acc, label='train accuracy')
    axs[1].set_title(label)
    plt.savefig(f'{label}.png')
    plt.show()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, train_loader, val_loader, optimizer, scheduler, loss_function):
    model.train()
    pbar = tqdm(train_loader)
    running_loss = 0.0
    total_val_loss = 0.0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_function(y_pred, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={running_loss} Batch_id={batch_idx} le={get_lr(optimizer)} Accuracy={100 * correct / processed:0.2f}')
    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct / processed
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_val_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = total_val_loss / len(val_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    validation_accuracy = 100. * correct / len(val_loader.dataset)
    return train_loss, train_accuracy, val_loss, validation_accuracy


if __name__ == '__main__':
    model = Net(dropout=0.2, num_classes=get_num_classes())
    print("Creating Model")
    print(f"Using {DEVICE} for training")
    model.to(DEVICE)
    print("Creating Data loaders")
    train_transforms, test_transforms = get_transforms()
    train_data, val_data, test_data = get_paths_and_labels(val_split=0.2)

    train_dataset = FERDataset(train_data[0], train_data[1], train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    val_dataset = FERDataset(val_data[0], val_data[1], test_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    test_dataset = FERDataset(test_data[0], test_data[1], test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=2)

    sgd_optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    adam_optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizers = {'SGD': sgd_optimizer,
                  'ADAM': adam_optimizer}

    cross_entropy_loss = nn.CrossEntropyLoss()
    kl_div_loss = nn.KLDivLoss()
    loss_functions = {'cross_entropy': cross_entropy_loss,
                      'kl_div': kl_div_loss}
    input_size = (BATCH_SIZE, 3, 48, 48)
    summary(model, input_size=input_size)
    epoch_train_acc = []
    epoch_train_loss = []
    epoch_valid_acc = []
    epoch_valid_loss = []
    min_val_loss = 99999
    print("Starting Training")
    for optimizer_name, optimizer in optimizers.items():
        for loss_name, loss_function in loss_functions.items():
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, steps_per_epoch=len(train_loader),
                                                            pct_start=0.2, div_factor=10, cycle_momentum=False,
                                                            epochs=N_EPOCHS)
            print("Optimizer:", optimizer_name)
            print("Loss function:", loss_name)
            for epoch in range(N_EPOCHS):
                print("EPOCH: %s LR: %s " % (epoch, get_lr(optimizer)))
                t_loss, t_acc, v_loss, v_acc = train(model, train_loader, val_loader, optimizer, scheduler,
                                                     loss_function)
                if v_loss < min_val_loss:
                    print("Validation Loss decreased, saving model")
                    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR,
                                                                f'{optimizer_name}_{loss_name}_best_model.pth'))
                epoch_train_loss.append(t_loss)
                epoch_train_acc.append(t_acc)
                epoch_valid_acc.append(v_acc)
                epoch_valid_loss.append(v_loss)
            plot(epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc,
                 f'Loss & Accuracy with {optimizer_name} optimizer and {loss_name} loss function')

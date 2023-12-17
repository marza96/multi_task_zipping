import torch

import numpy as np  
import torch.functional as F

from torch.cuda.amp import autocast


def evaluate_acc_single_head(model, loader=None, device=None, stop=10e6):
    assert loader is not None

    unique_labels = list()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            pred = outputs.argmax(dim=1)
            correct += (labels.to(device) == pred).sum().item()
            total += len(labels)

            unique = torch.unique(labels).cpu().numpy().tolist()
            unique_labels += unique

            unique_labels = np.unique(unique_labels).tolist()

            if total > stop:
                break

    return correct / total


def evaluate_acc(model, loader=None, device=None, stop=10e6):
    assert loader is not None

    model.eval()
    correct = 0
    total = 0
    unique_labels = list()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))

            outputs0 = torch.zeros(labels.shape[0], 20).to(device)
            outputs1 = torch.zeros(labels.shape[0], 20).to(device)

            outputs0[:, 10:] = outputs[:, 10:].clone()
            outputs1[:, :10] = outputs[:, :10].clone()

            pred0 = outputs0.argmax(dim=1)
            pred1 = outputs1.argmax(dim=1)

            head0_filter = (labels >= 10).to(device)
            head1_filter = (labels < 10).to(device)

            labels = labels.to(device)
            correct += (labels[head0_filter].to(device) == pred0[head0_filter]).sum().item()
            correct += (labels[head1_filter].to(device) == pred1[head1_filter]).sum().item()

            total += len(labels)

            unique = torch.unique(labels).cpu().numpy().tolist()
            unique_labels += unique

            unique_labels = np.unique(unique_labels).tolist()

            if total > stop:
                break

    return correct / total


def evaluate_acc_loss(model, loader=None, device=None):
    assert loader is not None

    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))

            pred = outputs.argmax(dim=1)
            correct += (labels.to(device) == pred).sum().item()
            total += len(labels)

            loss = F.cross_entropy(outputs, labels.to(device))
            losses.append(loss.item())

    return correct / total, np.array(losses).mean()


def full_eval1(model, train_loader, test_loader, device=None):
    tr_acc, tr_loss = evaluate_acc_loss(model, loader=train_loader, device=device)
    te_acc, te_loss = evaluate_acc_loss(model, loader=test_loader, device=device)

    return '%.2f, %.3f, %.2f, %.3f' % (100*tr_acc, tr_loss, 100*te_acc, te_loss)


def full_eval(model, train_loader, test_loader, device=None):
    tr_acc, tr_loss = evaluate_acc_loss(model, loader=train_loader, device=device)
    te_acc, te_loss = evaluate_acc_loss(model, loader=test_loader, device=device)

    return (100*tr_acc, tr_loss, 100*te_acc, te_loss)
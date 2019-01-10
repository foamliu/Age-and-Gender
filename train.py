import time

import torch.optim
import torch.utils.data
from torch import nn

from data_gen import AGDataset
from models import AGModel
from utils import *


def main():
    global best_accuracy, epochs_since_improvement, checkpoint, start_epoch

    # Initialize / load checkpoint
    if checkpoint is None:
        model = AGModel()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    age_criterion = nn.L1Loss().cuda()
    gender_criterion = nn.CrossEntropyLoss().cuda()
    reduce_gen_loss = 0.01
    criterion_info = (age_criterion, gender_criterion, reduce_gen_loss)

    # Custom dataloaders
    train_dataset = AGDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    val_dataset = AGDataset('valid')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion_info=criterion_info,
              optimizer=optimizer,
              epoch=epoch)
        train_dataset.shuffle()

        # One epoch's validation
        recent_accuracy = validate(val_loader=val_loader,
                                   model=model,
                                   criterion_info=criterion_info)

        # Check if there was an improvement
        is_best = recent_accuracy > best_accuracy
        best_accuracy = max(recent_accuracy, best_accuracy)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, recent_accuracy, is_best)


def train(train_loader, model, criterion_info, optimizer, epoch):
    model.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    gender_accs = AverageMeter()  # top5 accuracy

    age_criterion, gender_criterion, reduce_gen_loss = criterion_info

    start = time.time()

    # Batches
    for i, (inputs, age_true, gender_true) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        inputs = inputs.to(device)
        age_true = age_true.to(device)
        gender_true = gender_true.to(device)
        print('age_true.size(): ' + str(age_true.size()))
        print('gender_true.size(): ' + str(gender_true.size()))

        # Forward prop.
        gender_out, age_out = model(inputs)
        _, gender_pred = torch.max(gender_out, 1)
        _, max_cls_pred_age = torch.max(age_out, 1)
        gender_true = gender_true.view(-1)

        # Calculate loss
        gender_loss = gender_criterion(gender_out, gender_true)
        age_loss = age_criterion(age_out, age_true)
        gender_loss *= reduce_gen_loss
        loss = gender_loss + age_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        gender_accuracy = accuracy(gender_out, gender_true)
        losses.update(loss.item())
        gender_accs.update(gender_accuracy)
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gender_loss.val:.4f} ({gender_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Accuracy {gender_accs.val:.3f} ({gender_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         batch_time=batch_time,
                                                                                         data_time=data_time,
                                                                                         loss=losses,
                                                                                         gender_loss=gender_loss,
                                                                                         age_loss=age_loss,
                                                                                         gender_accs=gender_accs))


def validate(val_loader, model, criterion_info):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()
    gender_accs = AverageMeter()

    age_criterion, gender_criterion, reduce_gen_loss = criterion_info

    start = time.time()

    # Batches
    for i, (inputs, age_true, gender_true) in enumerate(val_loader):
        data_time.update(time.time() - start)

        # Move to device, if available
        inputs = inputs.to(device)
        age_true = age_true.to(device)
        gender_true = gender_true.to(device)

        # Forward prop.
        gender_out, age_out = model(inputs)
        _, gender_pred = torch.max(gender_out, 1)
        _, max_cls_pred_age = torch.max(age_out, 1)
        gender_true = gender_true.view(-1)

        # Calculate loss
        gender_loss = gender_criterion(gender_out, gender_true)
        age_loss = age_criterion(age_out, age_true)
        gender_loss *= reduce_gen_loss
        loss = gender_loss + age_loss

        # Keep track of metrics
        gender_accuracy = accuracy(gender_out, gender_true)
        losses.update(loss.item())
        gender_accs.update(gender_accuracy)
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gender_loss.val:.4f} ({gender_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Accuracy {gender_accs.val:.3f} ({gender_accs.avg:.3f})'.format(i, len(val_loader),
                                                                                         batch_time=batch_time,
                                                                                         data_time=data_time,
                                                                                         loss=losses,
                                                                                         gender_loss=gender_loss,
                                                                                         age_loss=age_loss,
                                                                                         gender_accs=gender_accs))

    return losses.avg


if __name__ == '__main__':
    main()

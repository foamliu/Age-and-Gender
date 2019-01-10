import time

import torch.optim
import torch.utils.data
from torch import nn

from data_gen import AGDataset
from models import AGModel
from utils import *


def main():
    global best_loss, epochs_since_improvement, checkpoint, start_epoch
    best_loss = 100000

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
    age_criterion = nn.MSELoss().cuda()
    gender_criterion = nn.CrossEntropyLoss().cuda()
    reduce_gen_loss = 1
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
        recent_loss = validate(val_loader=val_loader,
                               model=model,
                               criterion_info=criterion_info)

        # Check if there was an improvement
        is_best = recent_loss < best_loss
        best_loss = max(recent_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion_info, optimizer, epoch):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    gen_losses = AverageMeter()
    age_losses = AverageMeter()
    gender_accs = AverageMeter()  # gender accuracy

    age_criterion, gender_criterion, reduce_gen_loss = criterion_info

    # Batches
    for i, (inputs, age_true, gender_true) in enumerate(train_loader):
        # Move to GPU, if available
        inputs = inputs.to(device)
        age_true = age_true.float().to(device)
        gender_true = gender_true.to(device)

        # Forward prop.
        age_out, gender_out = model(inputs)
        _, age_out = torch.max(age_out, 1)
        age_out = age_out.float()
        gender_true = gender_true.view(-1)

        # Calculate loss
        gen_loss = gender_criterion(gender_out, gender_true)
        age_loss = age_criterion(age_out, age_true)
        gen_loss *= reduce_gen_loss
        loss = gen_loss + age_loss

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
        gen_losses.update(gen_loss.item())
        age_losses.update(age_loss.item())
        gender_accs.update(gender_accuracy)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Gender Accuracy {gender_accs.val:.3f} ({gender_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         gen_loss=gen_losses,
                                                                                         age_loss=age_losses,
                                                                                         gender_accs=gender_accs))


def validate(val_loader, model, criterion_info):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()
    gen_losses = AverageMeter()
    age_losses = AverageMeter()
    gender_accs = AverageMeter()  # gender accuracy

    age_criterion, gender_criterion, reduce_gen_loss = criterion_info

    start = time.time()

    # Batches
    for i, (inputs, age_true, gender_true) in enumerate(val_loader):
        data_time.update(time.time() - start)

        # batch_size = age_true.size()[0]

        # Move to GPU, if available
        inputs = inputs.to(device)
        age_true = age_true.float().to(device)
        gender_true = gender_true.to(device)

        # Forward prop.
        gender_out, age_out = model(inputs)
        _, age_pred = torch.max(age_out, 1)
        gender_true = gender_true.view(-1)

        # Calculate loss
        gen_loss = gender_criterion(gender_out, gender_true)
        age_loss = age_criterion(age_out, age_true)
        gen_loss *= reduce_gen_loss
        loss = gen_loss + age_loss

        # Keep track of metrics
        gender_accuracy = accuracy(gender_out, gender_true)
        losses.update(loss.item())
        gen_losses.update(gen_loss.item())
        age_losses.update(age_loss.item())
        gender_accs.update(gender_accuracy)
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gender_loss.val:.4f} ({gender_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Gender Accuracy {gender_accs.val:.3f} ({gender_accs.avg:.3f})'.format(i, len(val_loader),
                                                                                         loss=losses,
                                                                                         gender_loss=gen_losses,
                                                                                         age_loss=age_losses,
                                                                                         gender_accs=gender_accs))

    return losses.avg


if __name__ == '__main__':
    main()

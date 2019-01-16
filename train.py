import torch.optim
import torch.utils.data
from torch import nn

from data_gen import AgeGenDataset
from models import AgeGenPredModelRegression
from utils import *


def main():
    global best_loss, epochs_since_improvement, checkpoint, start_epoch
    best_loss = 100000

    # Initialize / load checkpoint
    if checkpoint is None:
        model = AgeGenPredModelRegression()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))
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
    age_loss_weight = 1
    criterion_info = (age_criterion, gender_criterion, age_loss_weight)

    # Custom dataloaders
    train_dataset = AgeGenDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    val_dataset = AgeGenDataset('valid')
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
        best_loss = min(recent_loss, best_loss)
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
    gen_accs = AverageMeter()  # gender accuracy
    age_mae = AverageMeter()  # age mae

    age_criterion, gender_criterion, age_loss_weight = criterion_info

    # Batches
    for i, (inputs, age_true, gen_true) in enumerate(train_loader):
        chunk_size = inputs.size()[0]
        # Move to GPU, if available
        inputs = inputs.to(device)
        age_true = age_true.view(-1, 1).float().to(device)  # [N, 1]
        gen_true = gen_true.to(device)  # [N, 1]

        # Forward prop.
        age_out, gen_out = model(inputs)  # age_out => [N, 101], gen_out => [N, 2]

        # Calculate loss
        gen_loss = gender_criterion(gen_out, gen_true)
        age_loss = age_criterion(age_out, age_true)
        age_loss *= age_loss_weight
        loss = gen_loss + age_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        gen_accuracy = accuracy(gen_out, gen_true)
        age_mae_loss = age_criterion(age_out, age_true)
        losses.update(loss.item(), chunk_size)
        gen_losses.update(gen_loss.item(), chunk_size)
        age_losses.update(age_loss.item(), chunk_size)
        gen_accs.update(gen_accuracy, chunk_size)
        age_mae.update(age_mae_loss)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Gender Accuracy {gen_accs.val:.3f} ({gen_accs.avg:.3f})\t'
                  'Age MAE {age_mae.val:.3f} ({age_mae.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                         loss=losses,
                                                                         gen_loss=gen_losses,
                                                                         age_loss=age_losses,
                                                                         gen_accs=gen_accs,
                                                                         age_mae=age_mae))
    print('\n')


def validate(val_loader, model, criterion_info):
    model.eval()  # eval mode (no dropout or batchnorm)

    losses = AverageMeter()
    gen_losses = AverageMeter()
    age_losses = AverageMeter()
    gen_accs = AverageMeter()  # gender accuracy
    age_mae = AverageMeter()  # age mae

    age_criterion, gender_criterion, age_loss_weight = criterion_info

    # Batches
    for i, (inputs, age_true, gen_true) in enumerate(val_loader):
        chunk_size = inputs.size()[0]
        # Move to GPU, if available
        inputs = inputs.to(device)
        age_true = age_true.view(-1, 1).float().to(device)
        gen_true = gen_true.to(device)

        # Forward prop.
        age_out, gen_out = model(inputs)

        # Calculate loss
        gen_loss = gender_criterion(gen_out, gen_true)
        age_loss = age_criterion(age_out, age_true)
        age_loss *= age_loss_weight
        loss = gen_loss + age_loss

        # Keep track of metrics
        gender_accuracy = accuracy(gen_out, gen_true)
        age_mae_loss = age_criterion(age_out, age_true)
        losses.update(loss.item())
        gen_losses.update(gen_loss.item(), chunk_size)
        age_losses.update(age_loss.item(), chunk_size)
        gen_accs.update(gender_accuracy, chunk_size)
        age_mae.update(age_mae_loss, chunk_size)

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Gender Accuracy {gen_accs.val:.3f} ({gen_accs.avg:.3f})\t'
                  'Age MAE {age_mae.val:.3f} ({age_mae.avg:.3f})'.format(i, len(val_loader),
                                                                         loss=losses,
                                                                         gen_loss=gen_losses,
                                                                         age_loss=age_losses,
                                                                         gen_accs=gen_accs,
                                                                         age_mae=age_mae))
    print('\n')

    return losses.avg


if __name__ == '__main__':
    main()

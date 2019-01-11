import torch.optim
import torch.utils.data
from torch import nn

from data_gen import AGDataset
from models import AgeGenPredModel
from utils import *


def main():
    global best_loss, epochs_since_improvement, checkpoint, start_epoch
    best_loss = 100000

    # Initialize / load checkpoint
    if checkpoint is None:
        model = AgeGenPredModel()
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
    age_criterion = nn.CrossEntropyLoss().cuda()
    gender_criterion = nn.CrossEntropyLoss().cuda()
    reduce_age_loss = 1
    criterion_info = (age_criterion, gender_criterion, reduce_age_loss)

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

        print('\n')

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
    age_accs = AverageMeter()  # age accuracy

    age_criterion, gender_criterion, reduce_age_loss = criterion_info

    # Batches
    for i, (inputs, age_true, gen_true) in enumerate(train_loader):
        # Move to GPU, if available
        inputs = inputs.to(device)
        age_true = age_true.to(device)  # [N]
        gen_true = gen_true.to(device)  # [N]

        # Forward prop.
        age_out, gen_out = model(inputs)  # age_out => [N, 101], gen_out => [N, 2]
        # print('age_out: ' + str(age_out))
        # print('gen_out: ' + str(gen_out))
        # _, age_out = torch.max(age_out, 1)  # [N, 101] => [N]
        # age_out = age_out.float()
        # print('age_out.size(): ' + str(age_out.size()))
        # print('age_out: ' + str(age_out))
        # print('age_true: ' + str(age_true))
        # print('gen_true: ' + str(gen_true))

        # Calculate loss
        gen_loss = gender_criterion(gen_out, gen_true)
        age_loss = age_criterion(age_out, age_true)
        age_loss *= reduce_age_loss
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
        age_accuracy = accuracy(age_out, age_true, 5)
        losses.update(loss.item())
        gen_losses.update(gen_loss.item())
        age_losses.update(age_loss.item())
        gen_accs.update(gen_accuracy)
        age_accs.update(age_accuracy)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Gender Accuracy {gen_accs.val:.3f} ({gen_accs.avg:.3f})\t'
                  'Age Top-5 Accuracy {age_accs.val:.3f} ({age_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                      loss=losses,
                                                                                      gen_loss=gen_losses,
                                                                                      age_loss=age_losses,
                                                                                      gen_accs=gen_accs,
                                                                                      age_accs=age_accs))


def validate(val_loader, model, criterion_info):
    model.eval()  # eval mode (no dropout or batchnorm)

    losses = AverageMeter()
    gen_losses = AverageMeter()
    age_losses = AverageMeter()
    gen_accs = AverageMeter()  # gender accuracy
    age_accs = AverageMeter()  # age accuracy

    age_criterion, gender_criterion, reduce_age_loss = criterion_info

    # Batches
    for i, (inputs, age_true, gen_true) in enumerate(val_loader):
        # Move to GPU, if available
        inputs = inputs.to(device)
        age_true = age_true.to(device)
        gen_true = gen_true.to(device)

        # Forward prop.
        age_out, gen_out = model(inputs)
        # _, age_pred = torch.max(age_out, 1)
        # age_out = age_out.float()

        # Calculate loss
        gen_loss = gender_criterion(gen_out, gen_true)
        age_loss = age_criterion(age_out, age_true)
        age_loss *= reduce_age_loss
        loss = gen_loss + age_loss

        # Keep track of metrics
        gender_accuracy = accuracy(gen_out, gen_true)
        age_accuracy = accuracy(age_out, age_true, 5)
        losses.update(loss.item())
        gen_losses.update(gen_loss.item())
        age_losses.update(age_loss.item())
        gen_accs.update(gender_accuracy)
        age_accs.update(age_accuracy)

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Gender Accuracy {gen_accs.val:.3f} ({gen_accs.avg:.3f})\t'
                  'Age Top-5 Accuracy {age_accs.val:.3f} ({age_accs.avg:.3f})'.format(i, len(val_loader),
                                                                                      loss=losses,
                                                                                      gen_loss=gen_losses,
                                                                                      age_loss=age_losses,
                                                                                      gen_accs=gen_accs,
                                                                                      age_accs=age_accs))

    return losses.avg


if __name__ == '__main__':
    main()

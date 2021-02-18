import os
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import config
from data_utils import TextMelLoader, TextMelCollate
import models
import commons
import utils
device = torch.device('cuda:1'if torch.cuda.is_available() else 'cpu')

Flag_load =  True

global_step = 0
def main():
    logger = utils.get_logger(config.model_dir)
    logger.info(config)

    writer = SummaryWriter(log_dir=config.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(config.model_dir, "eval"))

    torch.manual_seed(config.seed)

    train_dataset = TextMelLoader(config.training_files, config)

    collate_fn = TextMelCollate(1)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                              collate_fn=collate_fn, drop_last=True, num_workers=0)
    val_dataset = TextMelLoader(config.validation_files, config)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collate_fn, drop_last=True, num_workers=0)
    generator = models.FlowGenerator(n_vocab = config.n_symbols,hidden_channels=config.hidden_channels,filter_channels=config.filter_channels,filter_channels_dp=config.filter_channels_dp,out_channels=config.n_mel_channels).to(device)
    optimizer_g = commons.Adam(generator.parameters(), scheduler=config.scheduler, dim_model=config.hidden_channels,
                               warmup_steps=config.warmup_steps, lr=config.learning_rate, betas=config.betas,
                               eps=config.eps)
    if Flag_load == True:
        check_dir = os.path.join(config.model_dir, "G_205.pth")
        print(check_dir)
        checkpoint = torch.load(check_dir, map_location='cuda:1')
        generator.load_state_dict(checkpoint['model'])
        optimizer_g.load_state_dict(checkpoint['optimizer'])

        epoch_str = 205
        print("\n---Model Restored at Step 205---\n")
    else:
        print("\n---Start New Training---\n")
        if not os.path.exists(config.model_dir):
            os.mkdir(config.model_dir)
        epoch_str  =1
        global_step = 0
    for epoch in range(epoch_str, config.epochs + 1):
        train(epoch, config, generator, optimizer_g, train_loader, logger, writer)
        evaluate(epoch, config, generator, optimizer_g, val_loader, logger, writer_eval)
        utils.save_checkpoint(generator, optimizer_g, config.learning_rate, epoch,
                                  os.path.join(config.model_dir, "G_{}.pth".format(epoch)))
def train( epoch, hps, generator, optimizer_g, train_loader, logger, writer):
        global global_step

        generator.train()
        for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
            x, x_lengths = x.to(device), x_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            # Train Generator
            optimizer_g.zero_grad()

            (z, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs = generator(x, x_lengths, y, y_lengths, gen=False)
            l_mle = 0.5 * math.log(2 * math.pi) + (
                        torch.sum(y_logs) + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2) - torch.sum(
                    logdet)) / (torch.sum(y_lengths // hps.n_sqz) * hps.n_sqz * hps.n_mel_channels)
            l_length = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)

            loss_gs = [l_mle, l_length]
            loss_g = sum(loss_gs)

            loss_g.backward()
            grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
            optimizer_g.step()

            if batch_idx % config.log_interval == 0:
                (y_gen, *_), *_ = generator(x[:1], x_lengths[:1], gen=True)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss_g.item()))
                logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.get_lr()])

                scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()),
                            "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()),
                            "attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
                            },
                    scalars=scalar_dict)
            global_step += 1

            logger.info('====> Epoch: {}'.format(epoch))

def evaluate(epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
    global global_step
    generator.eval()
    losses_tot = []
    with torch.no_grad():
        for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(val_loader):
            x, x_lengths = x.to(device), x_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)

            (z, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs = generator(x, x_lengths, y, y_lengths, gen=False)
            l_mle = 0.5 * math.log(2 * math.pi) + (
                        torch.sum(y_logs) + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2) - torch.sum(
                    logdet)) / (torch.sum(y_lengths // hps.n_sqz) * hps.n_sqz * hps.n_mel_channels)
            l_length = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
            loss_gs = [l_mle, l_length]
            loss_g = sum(loss_gs)

            if batch_idx == 0:
                losses_tot = loss_gs
            else:
                losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

            if batch_idx % config.log_interval == 0:
                logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(val_loader.dataset),
                           100. * batch_idx / len(val_loader),
                    loss_g.item()))
                logger.info([x.item() for x in loss_gs])

    losses_tot = [x / len(val_loader) for x in losses_tot]
    loss_tot = sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))


if __name__ == "__main__":
    main()

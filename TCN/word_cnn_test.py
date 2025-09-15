import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
sys.path.append("../../")
from TCN.word_cnn.utils import *
from TCN.word_cnn.model import *
import pickle
from random import randint
import yaml
import numpy as np
from tqdm import tqdm
import datautils

parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus (default: ./data/penn)')
parser.add_argument('--emsize', type=int, default=600,
                    help='size of word embeddings (default: 600)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=40,
                    help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=80,
                    help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
parser.add_argument("--config", type=str, default="retrieval_ele.yaml")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--type", type=str, default="encode", choices=["encode", "retrieval"]
)
parser.add_argument("--encoder", default="TCN")
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)

path = "./TCN/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["retrieval"]["encoder"] = args.encoder

corpus = data_generator(args)
n_words = len(corpus.dictionary)

L=config["retrieval"]["L"]
H=config["retrieval"]["H"]
train_data = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"], data_path="electricity_2012_hour.csv",flag='train',size=[L, 0, L])
test_data = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"], data_path="electricity_2012_hour.csv",flag='test',size=[L, 0, L])
val_data = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"], data_path="electricity_2012_hour.csv",flag='val',size=[L, 0, L])

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
valid_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
model = TCN(
        input_size=96,
        # output_size = 1024,
        output_size=96, 
        num_channels=[config["retrieval"]["length"]] * (config["retrieval"]["level"]) + [config["retrieval"]["length"]],
    ).to(config["retrieval"]["device"])
# model=torch.load(config["path"]["encoder_path"], map_location='cuda:0')

# May use adaptive softmax to speed up training
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(data_source):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    with torch.no_grad():
        for i, (seq, labels, _, _) in enumerate(tqdm(data_source)):
            input = torch.tensor(seq).transpose(1, 2).float().to(config["retrieval"]["device"])
            labels = torch.tensor(labels).transpose(1, 2).float().to(config["retrieval"]["device"])
            output = model(input)
            loss = criterion(output, labels)

            # Note that we don't add TAR loss here
            total_loss += (len(data_source)) * loss.item()
            processed_data_size += len(data_source)
        return total_loss / processed_data_size


def train():
    # Turn on training mode which enables dropout.
    global train_data
    model.train()
    total_loss = 0
    start_time = time.time()
    for i, (seq, labels, _, _) in enumerate(tqdm(train_loader)):
        input = torch.tensor(seq).transpose(1, 2).float().to(config["retrieval"]["device"])
        labels = torch.tensor(labels).transpose(1, 2).float().to(config["retrieval"]["device"])
        output = model(input)
        loss = criterion(output, labels)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_data) // args.validseqlen, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    best_vloss = 1e8

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(valid_loader)
            test_loss = evaluate(test_loader)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            test_loss, math.exp(test_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss

            # Anneal the learning rate if the validation loss plateaus
            if epoch > 5 and val_loss >= max(all_vloss[-5:]):
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

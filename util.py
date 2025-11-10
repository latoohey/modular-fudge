import os
import time
import torch

def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)

def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_mask(lengths: torch.LongTensor) -> torch.ByteTensor:
    """
    Create a mask of batch x seq where 1 is for non-padding
    and 0 is for padding.
    """
    max_seqlen = torch.max(lengths)
    # (max_seqlen, batch_size)
    expanded_lengths = lengths.unsqueeze(0).repeat((max_seqlen, 1))
    # (max_seqlen, batch_size)
    indices = torch.arange(max_seqlen).unsqueeze(1).repeat((1, lengths.size(0))).to(lengths.device)
    
    # (max_seqlen, batch_size) -> (batch_size, max_seqlen)
    return (expanded_lengths > indices).permute(1, 0)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries.append(time.ctime(time.time()))
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
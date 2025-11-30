def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path}")
    
def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_mask(lengths: torch.LongTensor) -> torch.BoolTensor:
    """
    Create a mask of batch x seq where 1 is for non-padding
    and 0 is for padding.
    """
    device = lengths.device
    max_len = lengths.max().item() # .item() converts Tensor to int

    # 1. Create range [0, 1, 2, ... max_len-1]
    # Shape: (1, max_len)
    ids = torch.arange(max_len, device=device).unsqueeze(0)

    # 2. Compare with lengths
    # Shape: (batch_size, 1)
    # Broadcasting: (1, max_len) < (batch_size, 1) -> (batch_size, max_len)
    mask = ids < lengths.unsqueeze(1)

    return mask

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
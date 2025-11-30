from .lstm import LSTMClassifier

def get_model(args, vocab_size, pad_id):
    """
    This factory function reads the --model_type argument
    and returns the correct, initialized model.
    """
    if args.model_type == 'lstm':
        return LSTMClassifier(args, vocab_size, pad_id)
    elif args.model_type == 'mamba':
        return MambaClassifier(args, vocab_size, pad_id)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
from .lstm import LSTMClassifier

def get_model(args, tokenizer_pad_id):
    """
    This factory function reads the --model_type argument
    and returns the correct, initialized model.
    """
    if args.model_type == 'lstm':
        return LSTMClassifier(args, tokenizer_pad_id)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
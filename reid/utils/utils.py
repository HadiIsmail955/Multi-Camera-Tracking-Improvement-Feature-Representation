import torch


def set_device(device: str) -> torch.device:
    """Automatically select device if 'auto' is specified.

    Args:
        device (str): The device to use. Can be 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        torch.device: The selected device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

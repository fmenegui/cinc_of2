def test():
    import torch

    gpu_available = torch.cuda.is_available()
    print("GPU is available:", gpu_available)
    if gpu_available:
        print("GPU Details:", torch.cuda.get_device_name(torch.cuda.current_device()))
    return gpu_available

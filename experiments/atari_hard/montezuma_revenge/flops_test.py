import torch
import time


print("mps available ", torch.backends.mps.is_available())

rounds = 10
n_size = [1000, 2000, 4000, 6000, 8000]

for n in n_size:
    
    y = 0

    time_start = time.time()
    for i in range(rounds):
        a = torch.randn((n, n), device="mps")
        b = torch.randn((n, n), device="mps")

        y+= a@b
    time_stop = time.time()

    flops = 2.0*rounds*(n*n*n)/(time_stop - time_start)

    print(n, round(flops/(10**9), 1), "GFlops")

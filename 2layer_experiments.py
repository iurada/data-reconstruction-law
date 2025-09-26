import torch
from torchvision.utils import save_image
from torch.func import jacrev
import os
import numpy as np
from argparse import ArgumentParser
from data.cifar10 import FlatMulticlassCIFAR10
from utils import optimal_permutation, orth_rowspan_residuals
from time import time

def main(args):

    print('\n[+] Building Data...\n')
    dset = FlatMulticlassCIFAR10(args.root)
    X, Y = dset.generate_data(n_examples_per_class=args.n//args.n_classes, n_classes=args.n_classes)
    
    print('\n[+] Building Model & Fit it on Data...\n')
    width = args.p
    activation_fn = torch.relu
    W1 = torch.randn((width, dset.input_dim)) / (dset.input_dim**0.5)
    W2 = torch.randn((dset.n_classes, width)) / (width**0.5)

    W2_init = W2.clone().detach().cuda()
    W1, W2 = W1.cuda(), W2.cuda()

    W1.requires_grad_(True)
    W2.requires_grad_(True)

    optim = torch.optim.SGD([W1, W2], lr=1e-5)
    X, Y = X.cuda(), Y.cuda()
    for it in range(int(1e6)):
        pred = (W2 @ activation_fn(W1 @ X.to(args.dtype).T)).T
        loss = torch.nn.functional.mse_loss(pred, torch.nn.functional.one_hot(Y, dset.n_classes).to(args.dtype))
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(f'[{it}] Loss: {loss.item():.6f}', end='\r')
        if loss.item() < 1e-7: break
    print()
    W1.requires_grad_(False)
    W2.requires_grad_(False)

    train_loss = loss.item()
    print(f'[+] Avg. Train Loss: {train_loss:.6f}')

    if args.recon_save_path is not None:
        save_folder = os.path.join(args.recon_save_path, f'2layer_cifar10_p={args.p}_lr={args.recon_lr}_it={args.recon_iterations}_seed={args.seed}')
        os.makedirs(save_folder, exist_ok=True)
        for j, real_x in enumerate(X):
            save_image((real_x.clone().detach() * dset.train_std + dset.train_mean).view(3, 32, 32), os.path.join(save_folder, f'real_{j}.png'))

    print('\n[+] Recovering Training data...\n')
    recon_X = torch.randn((args.n, dset.input_dim), dtype=args.dtype)
    recon_X = recon_X.cuda()
    recon_X.requires_grad_(True)

    W = (W2 - W2_init).view(-1)
    def compute_Phi(X):
        if len(X.size()) < 2: X = X.unsqueeze(0)
        def f_params(X, W1, W2):
            logits = ((W2 @ activation_fn(W1 @ X.T)).squeeze()).T
            return logits
        grads = jacrev(f_params, argnums=2)(X, W1, W2)
        return grads.view(X.size(0) * dset.n_classes, -1)

    optim = torch.optim.SGD([recon_X], lr=args.recon_lr, momentum=0.9)

    time_start = time()
    for rnd in range(args.recon_iterations):
        Phi = compute_Phi(recon_X)
        v, A = Phi @ W, Phi @ Phi.T
        alpha = torch.linalg.solve(A.to(torch.float32), v.to(torch.float32))
        recon_loss = (W.dot(W) - v.dot(alpha)) / W.norm(p=2)**2
        tot_loss = recon_loss.item()

        recon_loss.backward()
        optim.step()
        optim.zero_grad()
        
        with torch.no_grad():
            recon_X.mul_((dset.input_dim**0.5) / recon_X.norm(p=2, dim=1, keepdim=True))
            time_stop = time()
            bestfit_idx_perm = optimal_permutation(X.cpu().abs(), recon_X.cpu().to(torch.float32).abs())
            print(f'[{rnd} | {time_stop - time_start:.3f}s] dist. to optimum = {torch.norm(X.cpu().abs() - recon_X[bestfit_idx_perm].cpu().to(torch.float32).abs(), p=2, dim=1).sum().item() / ((dset.input_dim)**0.5 * args.n):.6f} | Loss = {tot_loss}', end='\r')
            if abs(tot_loss) <= 1e-7: break
            time_start = time()
    print()

    recon_X.requires_grad_(False)
    with torch.no_grad(): 
        bestfit_idx_perm = optimal_permutation(X.cpu().abs(), recon_X.cpu().to(torch.float32).abs())
        final_dto = torch.norm(X.cpu().abs() - recon_X[bestfit_idx_perm].cpu().to(torch.float32).abs(), p=2, dim=1).sum().item() / ((dset.input_dim)**0.5 * args.n)
    print(f'[FINAL] dist. to optimum = {final_dto:.6f}')

    recon_X = recon_X[bestfit_idx_perm]
    if args.recon_save_path is not None:
        for j in range(args.n):
            save_image((recon_X[j].clone().detach().cpu() * dset.train_std + dset.train_mean).view(3, 32, 32), os.path.join(save_folder, f'recon_{j}.png'))
    
    recon_Phi = compute_Phi(recon_X).cpu().numpy()
    Phi = compute_Phi(X).cpu().numpy()
    res = orth_rowspan_residuals(recon_Phi, Phi, args.p)
    print('Phi rows on rowspan(hatPhi):', np.mean(res), '±', np.std(res))


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--p', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--recon_lr', type=float, default=2000.0)
    parser.add_argument('--recon_iterations', type=int, default=1_000_000)
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64'])
    parser.add_argument('--recon_save_path', default=None)

    args = parser.parse_args()

    args.dtype = {'fp32': torch.float32, 'fp64': torch.float64}[args.dtype]

    torch.manual_seed(args.seed)
    main(args)
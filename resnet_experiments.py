import torch
from torchvision.utils import save_image
import os
import numpy as np
from argparse import ArgumentParser
from data.cifar10 import BinaryCIFAR10
from models.resnets import ResNet
from utils import optimal_permutation, orth_rowspan_residuals
from time import time

def main(args):

    print('\n[+] Building Data...\n')
    dset = BinaryCIFAR10(args.root)
    X, Y = dset.generate_data(args.n)
    
    print('\n[+] Building Model & Fit it on Data...\n')
    width = int(args.p / (7 * 7))
    model = ResNet(in_channels=3, num_classes=1, num_blocks=4, width=width, stem_kernel=28)

    head_weight_init = model.head.weight.data.clone()
    head_weight_init.requires_grad_(True)

    model.cuda()
    X, Y = X.cuda(), Y.cuda()

    gd_optim = torch.optim.SGD(model.parameters(), lr=1e-5)
    for it in range(int(1e6)):
        loss = torch.nn.functional.mse_loss(model(X).squeeze(), Y)
        loss.backward()
        gd_optim.step()
        gd_optim.zero_grad()
        print(f'[{it}] Loss: {loss.item():.9f}', end='\r')
        if loss.item() < 1e-7: break
    print()
    train_loss = loss.item()
    print(f'[+] Avg. Train Loss: {train_loss:.6f}')
    
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    X, Y = X.cpu(), Y.cpu()

    if args.recon_save_path is not None:
        save_folder = os.path.join(args.recon_save_path, f'resnet_cifar10_width={width}_lr={args.recon_lr}_it={args.recon_iterations}_seed={args.seed}')
        os.makedirs(save_folder, exist_ok=True)
        for j, real_x in enumerate(X):
            save_image((real_x.clone().detach().cpu() * torch.tensor(list(dset.train_std)).unsqueeze(-1).unsqueeze(-1) + torch.tensor(list(dset.train_mean)).unsqueeze(-1).unsqueeze(-1)), os.path.join(save_folder, f'real_{j}.png'))

    print('\n[+] Recovering Training data...\n')
    recon_X = torch.randn_like(X) / (dset.input_dim**0.5)
    recon_X = recon_X.cuda()
    recon_X.requires_grad_(True)

    W = (model.head.weight.data - head_weight_init.cuda()).view(-1)    
    get_features = lambda x : torch.cat([g.view(-1) for g in torch.autograd.grad(model(x).sum(), list(model.head.parameters()), retain_graph=True, create_graph=True)])

    optim = torch.optim.SGD([recon_X], lr=args.recon_lr, momentum=0.9)

    time_start = time()
    for rnd in range(args.recon_iterations):
        Phi = torch.cat([get_features(x.unsqueeze(0).cuda()).squeeze().unsqueeze(0) for x in recon_X])
        v, A = Phi @ W, Phi @ Phi.T
        alpha = torch.linalg.solve(A.to(torch.float32), v.to(torch.float32))
        recon_loss = (W.dot(W) - v.dot(alpha)) / W.norm(p=2)**2
        tot_loss = recon_loss.item()

        recon_loss.backward()
        optim.step()
        optim.zero_grad()
        
        with torch.no_grad():
            recon_X.mul_((dset.input_dim**0.5) / recon_X.norm(p=2, dim=(1,2,3), keepdim=True))
            time_stop = time()
            bestfit_idx_perm = optimal_permutation(X.cpu().abs().view(X.size(0), -1), recon_X.cpu().abs().view(recon_X.size(0), -1))
            print(f'[{rnd} | {time_stop - time_start:.3f}s] dist. to optimum = {torch.norm(X.cpu().abs().view(X.size(0), -1) - recon_X[bestfit_idx_perm].cpu().to(torch.float32).abs().view(recon_X.size(0), -1), p=2, dim=1).sum().item() / (dset.input_dim**0.5 * args.n):.6f} | Loss = {tot_loss}', end='\r')
            if abs(tot_loss) <= 1e-7: break
            time_start = time()
    print()

    recon_X.requires_grad_(False)
    with torch.no_grad(): 
        bestfit_idx_perm = optimal_permutation(X.cpu().abs().view(X.size(0), -1), recon_X.cpu().abs().view(recon_X.size(0), -1))
        final_dto = torch.norm(X.cpu().abs().view(X.size(0), -1) - recon_X[bestfit_idx_perm].cpu().to(torch.float32).abs().view(recon_X.size(0), -1), p=2, dim=1).sum().item() / (dset.input_dim**0.5 * args.n)
    print(f'[FINAL] dist. to optimum = {final_dto:.6f}')

    recon_X = recon_X[bestfit_idx_perm]
    if args.recon_save_path is not None:
        for j in range(args.n):
            save_image((recon_X[j].clone().detach().cpu() * torch.tensor(list(dset.train_std)).unsqueeze(-1).unsqueeze(-1) + torch.tensor(list(dset.train_mean)).unsqueeze(-1).unsqueeze(-1)), os.path.join(save_folder, f'recon_{j}.png'))
    
    recon_Phi = torch.cat([get_features(x.unsqueeze(0).cuda()).squeeze().unsqueeze(0) for x in recon_X]).cpu().numpy()
    Phi = torch.cat([get_features(x.unsqueeze(0).cuda()).squeeze().unsqueeze(0) for x in X]).cpu().numpy()
    res = orth_rowspan_residuals(recon_Phi, Phi, args.p)
    print('Phi rows on rowspan(hatPhi):', np.mean(res), '±', np.std(res))


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--p', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('--recon_lr', type=float, default=2000.0)
    parser.add_argument('--recon_iterations', type=int, default=1_000_000)
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64'])
    parser.add_argument('--recon_save_path', default=None)

    args = parser.parse_args()

    args.dtype = {'fp32': torch.float32, 'fp64': torch.float64}[args.dtype]

    torch.manual_seed(args.seed)
    main(args)
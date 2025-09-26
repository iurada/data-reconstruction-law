import torch
from torchvision.utils import save_image
import os
import numpy as np
from argparse import ArgumentParser
from data.cifar10 import FlatBinaryCIFAR10
from models.random_features import RandomFeaturesModel
from utils import optimal_permutation, orth_rowspan_residuals
from time import time

def main(args):

    print('\n[+] Building Data...\n')
    if args.dataset == 'synthetic':
        class DB: pass
        dset = DB()
        setattr(dset, 'input_dim', args.d)
        X = torch.randn((args.n, dset.input_dim))
        X = (dset.input_dim**0.5) * X / X.norm(dim=1).unsqueeze(1)

        G = torch.randn(dset.input_dim) / (dset.input_dim**0.5)
        Y = (X @ G)
        Y += torch.normal(0, 0.5, size=(args.n,))

    elif args.dataset == 'cifar10':
        dset = FlatBinaryCIFAR10(args.root)
        X, Y = dset.generate_data(args.n)

    print('\n[+] Building Model & Fit it on Data...\n')
    if args.activation_fn == 'relu':
        activation_fn = torch.relu
    elif args.activation_fn == 'relu+tanh':
        activation_fn = lambda u : torch.relu(u) + torch.tanh(u)

    model = RandomFeaturesModel(args.p, dset.input_dim, activation_fn, batch_size=args.p//args.rf_bs_V, base_seed=args.rf_seed_V, dtype=args.dtype)
    model.fit(X.cuda(), Y)

    train_loss = 0
    for i in range(X.size(0)):
        train_loss += (model.predict(X[i].cuda()).item() - Y[i].item())**2
    train_loss /= X.size(0)
    print(f'[+] Avg. Train Loss: {train_loss:.6f}')

    if args.recon_save_path is not None and args.dataset == 'cifar10':
        save_folder = os.path.join(args.recon_save_path, f'rf_cifar10_p={args.p}_act={args.activation_fn}_lr={args.recon_lr}_it={args.recon_iterations}_rfseed={args.rf_seed_V}_seed={args.seed}')
        os.makedirs(save_folder, exist_ok=True)
        for j, real_x in enumerate(X):
            save_image((real_x.clone().detach() * dset.train_std + dset.train_mean).view(3, 32, 32), os.path.join(save_folder, f'real_{j}.png'))

    print('\n[+] Recovering Training data...\n')
    recon_X = torch.randn((args.n, dset.input_dim), dtype=model.dtype)
    recon_X = recon_X.cuda()
    recon_X.requires_grad_(True)

    W = model.w.cuda()

    optim = torch.optim.SGD([recon_X], lr=args.recon_lr, momentum=0.9)

    time_start = time()
    for rnd in range(args.recon_iterations):
        with torch.no_grad():
            # Compute Recon Loss
            Phi = torch.cat([model.get_feature_vector(x.cuda()).unsqueeze(0) for x in recon_X], dim=0)
            v, A = Phi @ W, Phi @ Phi.T
            alpha = torch.linalg.solve(A.to(torch.float32), v.to(torch.float32)).to(model.dtype)
            recon_loss = (W.dot(W) - v.dot(alpha)) / W.norm(p=2)**2
            tot_loss = recon_loss.item()

            # Compute Gradient of Recon Loss
            grad_Phi = 2 * ((alpha.unsqueeze(1) @ alpha.unsqueeze(0)) @ Phi - alpha.unsqueeze(1) @ W.unsqueeze(0))
            if args.activation_fn == 'relu':
                mask = torch.where(Phi > 0, 1.0, 2e-2).to(model.dtype)
            elif args.activation_fn == 'relu+tanh':
                device = 'cuda'
                gen = model.cuda_generator
                gen.manual_seed(model.base_seed)
                num_batches = (model.p + model.batch_size - 1) // model.batch_size
                features = []
                for batch_idx in range(num_batches):
                    current_batch_size = min(model.batch_size, model.p - batch_idx * model.batch_size)
                    phi_b = torch.matmul(X.to(model.dtype).cuda(), torch.normal(0, 1 / (model.d**0.5), size=(current_batch_size, model.d), generator=gen, device=device, dtype=model.dtype).T)
                    features.append(phi_b.squeeze())
                features = torch.cat(features, dim=-1)
                mask = torch.where(features > 0, 1.0, 2e-2).to(model.dtype)
                mask = mask + (1 - torch.tanh(features).pow(2))
            
            device = 'cuda'
            gen = model.cuda_generator
            gen.manual_seed(model.base_seed)
            num_batches = (model.p + model.batch_size - 1) // model.batch_size
            recon_X.grad = torch.zeros_like(recon_X)
            prev_batch_idx = 0
            for batch_idx in range(num_batches):
                current_batch_size = min(model.batch_size, model.p - batch_idx * model.batch_size)
                recon_X.grad += grad_Phi[:,prev_batch_idx:prev_batch_idx+current_batch_size] * mask[:, prev_batch_idx:prev_batch_idx+current_batch_size] @ torch.normal(0, 1 / (model.d**0.5), size=(current_batch_size, model.d), generator=gen, device=device, dtype=model.dtype)
                prev_batch_idx += current_batch_size
            recon_X.grad /= W.norm(p=2)**2

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
    if args.recon_save_path is not None and args.dataset == 'cifar10':
        for j in range(args.n):
            save_image((recon_X[j].clone().detach().cpu() * dset.train_std + dset.train_mean).view(3, 32, 32), os.path.join(save_folder, f'recon_{j}.png'))
    
    recon_Phi = torch.cat([model.get_feature_vector(x.cuda()).unsqueeze(0) for x in recon_X], dim=0).cpu().numpy()
    Phi = torch.cat([model.get_feature_vector(x.cuda()).unsqueeze(0) for x in X], dim=0).cpu().numpy()
    res = orth_rowspan_residuals(recon_Phi, Phi, args.p)
    print('Phi rows on rowspan(hatPhi):', np.mean(res), '±', np.std(res))


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--p', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('--d', type=int)
    parser.add_argument('--activation_fn', type=str, choices=['relu', 'relu+tanh'])
    parser.add_argument('--rf_bs_V', type=int, default=1)
    parser.add_argument('--rf_seed_V', type=int, default=42)
    parser.add_argument('--recon_lr', type=float, default=2000.0)
    parser.add_argument('--recon_iterations', type=int, default=1_000_000)
    parser.add_argument('--dataset', type=str, choices=['synthetic', 'cifar10'])
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64'])
    parser.add_argument('--recon_save_path', default=None)

    args = parser.parse_args()

    args.dtype = {'fp32': torch.float32, 'fp64': torch.float64}[args.dtype]

    torch.manual_seed(args.seed)
    main(args)
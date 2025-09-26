import torch

class RandomFeaturesModel:
    def __init__(self, p, d, activation=torch.relu, batch_size=1024, base_seed=None, dtype=torch.float32):
        self.p = p
        self.d = d
        self.dtype = dtype
        self.batch_size = batch_size
        self.base_seed = torch.randint(1, int(1e6), size=(1,)).item() if base_seed is None else base_seed
        self.s = activation
        self.w = None
        self.cpu_generator = torch.Generator('cpu')
        self.cuda_generator = torch.Generator('cuda')

    def fit(self, X, Y):
        print('Fitting least squares...')
        self.w = (torch.pinverse(torch.cat([self.get_feature_vector(x.cuda()).unsqueeze(0) for x in X])).to(torch.float64) @ Y.to(torch.float64).cuda()).cuda()
        self.w = self.w.to(self.dtype)

    def predict(self, x):
        return self.get_feature_vector(x) @ self.w.cuda()
    
    def get_feature_vector(self, x):
        device = x.device
        gen = self.cpu_generator if device == torch.device('cpu') else self.cuda_generator
        gen.manual_seed(self.base_seed)
        
        features = []
        num_batches = (self.p + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            current_batch_size = min(self.batch_size, self.p - batch_idx * self.batch_size)
            phi_b = self.s(torch.matmul(x.to(self.dtype), torch.normal(0, 1 / (self.d**0.5), size=(current_batch_size, self.d), generator=gen, device=device, dtype=self.dtype).T))
            features.append(phi_b.squeeze())

        return torch.cat(features)
import tntorch as tn
import torch
import operator
torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
DEVICE = torch.device('cuda:2')


class CQFSTTSampler:
    _tol: float

    def __init__(self, tol: float = 1e-4):
        self._tol = tol

    def _optimize(self, W, s, k, eps=1e-6):
        N = W.shape[0]
        assert W.shape[1] == N

        # First term: x @ W @ x^T
        ts = []
        for n in range(N):
            c = torch.zeros([1, 2, 1], dtype=DTYPE, device=DEVICE)
            c[0, 1, 0] = 1
            t = tn.Tensor([c], device=DEVICE)[:, None]
            t.cores[1] = t.cores[1].repeat(1, N, 1)
            t.cores[1][:, :n, :] = 0
            t.cores[1][:, n + 1:, :] = 0
            t = tn.unsqueeze(t, list(range(n)) + list(range(n + 1, N)))
            t = t.repeat(*[2] * n, 1, *[2] * (N - n - 1), 1)
            ts.append(t)
        t = sum(ts)
        ts = []
        for n in range(N):
            trow = t.clone()
            trow.cores[-1] = torch.einsum('ijl,jk->ikl', trow.cores[-1], W[:, n:n + 1])
            ts.append(trow)
        term1 = tn.reduce(ts, operator.add, eps=eps)[..., 0]

        # Second term
        c = torch.eye(2, 2, dtype=DTYPE, device=DEVICE)[:, None, :].repeat(1, 2, 1)
        c[1, 0, 0] = (-k) ** 2
        c[1, 1, 0] = (1 - k) ** 2
        xs = tn.Tensor([c[-1:, :, :]] + [c] * (N - 2) + [c[:, :, 0:1]], device=DEVICE)
        term2 = s * tn.round_tt(xs, eps=eps)

        # Minimize tensor
        target = tn.round_tt(term1 + term2, eps=eps)
        return tn.argmin(target, rmax=50, verbose=True)

    def sample(self, *, FPM, s, k):
        with torch.cuda.device(DEVICE):
            FPM = torch.tensor(FPM, dtype=DTYPE, device=DEVICE)
            f = self._optimize(FPM, s, k)
        return {
            i: f[i]
            for i in range(len(f))
        }

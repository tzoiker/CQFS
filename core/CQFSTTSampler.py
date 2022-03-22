import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)


class CQFSTTSampler:
    _tol: float

    def __init__(self, tol: float = 1e-4):
        self._tol = tol

    def _optimize(self, W, s, k, eps=1e-4):
        N = W.shape[0]
        assert W.shape[1] == N

        # First term: x @ W @ x^T
        ts = []
        for n in range(N):
            c = torch.zeros([1, 2, 1])
            c[0, 1, 0] = 1
            t = tn.Tensor([c])[:, None]
            t.cores[1] = t.cores[1].repeat(1, N, 1)
            t.cores[1][:, :n, :] = 0
            t.cores[1][:, n + 1:, :] = 0
            t = tn.unsqueeze(t, list(range(n)) + list(range(n + 1, N)))
            t = t.repeat(*[2] * n, 1, *[2] * (N - n - 1), 1)
            ts.append(t)
        t = sum(ts)
        t2 = t.clone()
        t.cores[-1] = torch.einsum('ijl,jk->ikl', t.cores[-1], W)
        term1 = tn.cross(tensors=[t, t2], function=lambda x, y: x * y,
                         ranks_tt=N * 2 + 1, eps=eps)
        term1 = tn.sum(term1, dim=-1)

        # Second term
        c = torch.eye(2, 2)[:, None, :].repeat(1, 2, 1)
        c[1, 0, 0] = (-k) ** 2
        c[1, 1, 0] = (1 - k) ** 2
        xs = tn.Tensor([c[-1:, :, :]] + [c] * (N - 2) + [c[:, :, 0:1]])
        term2 = s * tn.round_tt(xs, eps=eps)

        # Minimize tensor
        target = tn.round_tt(term1 + term2, eps=eps)
        return tn.argmin(target)

    def sample(self, *, FPM, s, k):
        FPM = torch.tensor(FPM)
        f = self._optimize(FPM, s, k)
        return {
            i: f[i]
            for i in range(len(f))
        }

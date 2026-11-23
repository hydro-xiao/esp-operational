"""CMAL (countable mixture of asymmetric Laplacians) post-processor. Single
pooled model across all basins and issuance months (basin + month are input
features, not separate models), fit by maximizing log-likelihood. Ported
unchanged from the research project's cmal_lib.py so this package has no
dependency on it."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

NUM_COLS = ["log_raw_median", "log_raw_spread", "snow_pct_normal", "month"]


def ald_log_prob(y, mu, s, tau, eps=1e-6):
    s = torch.clamp(s, min=eps)
    tau = torch.clamp(tau, min=eps, max=1 - eps)
    d = y - mu
    rho = d * (tau - (d < 0).float())
    return torch.log(tau) + torch.log(1 - tau) - torch.log(s) - rho / s


class CMALNet(nn.Module):
    def __init__(self, n_features, n_components=3, hidden=32):
        super().__init__()
        self.K = n_components
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4 * n_components),
        )

    def forward(self, x):
        out = self.net(x)
        K = self.K
        alpha_logits, mu, s_raw, tau_raw = out[:, :K], out[:, K:2*K], out[:, 2*K:3*K], out[:, 3*K:4*K]
        alpha = torch.softmax(alpha_logits, dim=-1)
        s = torch.nn.functional.softplus(s_raw) + 1e-3
        tau = torch.sigmoid(tau_raw) * 0.98 + 0.01
        return alpha, mu, s, tau

    def nll(self, x, y):
        alpha, mu, s, tau = self.forward(x)
        y = y.unsqueeze(-1).expand_as(mu)
        log_comp = ald_log_prob(y, mu, s, tau) + torch.log(alpha + 1e-12)
        return -torch.logsumexp(log_comp, dim=-1).mean()


def fit_cmal(X_train, y_train, n_features, n_components=3, epochs=400, lr=5e-3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = CMALNet(n_features, n_components).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = model.nll(Xt, yt)
        loss.backward()
        opt.step()
    return model


def sample_mixture(alpha, mu, s, tau, n_samples, rng):
    K = len(alpha)
    comp = rng.choice(K, size=n_samples, p=alpha / alpha.sum())
    u = rng.uniform(1e-6, 1 - 1e-6, size=n_samples)
    m, sc, t = mu[comp], s[comp], tau[comp]
    y = np.where(
        u < t,
        m + sc / (1 - t) * np.log(u / t),
        m - sc / t * np.log((1 - u) / (1 - t)),
    )
    return y


def mixture_quantiles(alpha, mu, s, tau, qs=(0.1, 0.5, 0.9), n_grid=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    samp = sample_mixture(alpha, mu, s, tau, n_grid, rng)
    return np.quantile(samp, qs)


def build_features(df, mu_x, sd_x, onehot_cols):
    """df needs: raw_median, raw_p10, raw_p90, snow_pct_normal, month, basin.
    mu_x/sd_x/onehot_cols come from the saved fit (fit_statistical.py)."""
    df = df.copy()
    df["log_raw_median"] = np.log(df["raw_median"])
    df["log_raw_spread"] = np.log(np.maximum(df["raw_p90"] - df["raw_p10"], 1e-6))
    X_num = df[NUM_COLS].values.astype(float)
    X_num_std = (X_num - mu_x) / sd_x
    onehot = pd.get_dummies(df["basin"], prefix="b").reindex(columns=onehot_cols, fill_value=0.0).astype(float)
    return np.concatenate([X_num_std, onehot.values], axis=1)

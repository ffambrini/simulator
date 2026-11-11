#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador RMAB para alocacao de intervencoes anti-evasao no Pe-de-Meia 
Somente numpy (PPO com backprop manual). Politicas: none, random, static_topk, dynamic_topk,
whittle, whittle_eq, ppo, oracle.

ATENCAO (valores a substituir antes de submeter):
  - UF_RATE: taxas por UF lidas visualmente da Fig. 4 do artigo 1 (exatos apenas AP=4.1% e GO=76.2%).
  - UF_POP: pesos aproximados (populacao, ordem de grandeza); trocar pela contagem real da coorte.
  - Taxa de evasao anual media (--d-bar), dinamica de risco e efeito eps sao SUPOSICOES (nao identificaveis).
  - PPO: restricao suave (Lagrangiano, Bernoulli por aluno) no treino; implantacao por top-k do logit.
  - Recompensa do PPO = sobrevivencia esperada (1-h), Rao-Blackwell da sobrevivencia amostrada.
  - Perfis de eficacia da intervencao por nivel de risco (--eps-profile) sao hipoteses de modelagem, nao dados.
Uso: python simulador_pdm_rmab.py --mode run --eps-profile hump --eps 0.3 --k-frac 0.05
     python simulador_pdm_rmab.py --mode sweep --no-ppo --jobs 2
"""
import argparse
import csv
import os
import time
from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

UF_RATE = {
    "GO": 0.762, "DF": 0.700, "SP": 0.660, "MT": 0.550, "MG": 0.520, "PR": 0.500, "RS": 0.460,
    "AC": 0.450, "PI": 0.410, "CE": 0.380, "RO": 0.365, "SC": 0.365, "AM": 0.360, "MA": 0.310,
    "TO": 0.300, "SE": 0.280, "RJ": 0.255, "RN": 0.250, "PE": 0.190, "AL": 0.165, "ES": 0.155,
    "MS": 0.140, "BA": 0.135, "RR": 0.125, "PB": 0.120, "PA": 0.070, "AP": 0.041,
}
UF_POP = {
    "SP": 44.4, "MG": 20.5, "RJ": 16.0, "BA": 14.1, "PR": 11.4, "RS": 10.9, "PE": 9.0, "CE": 8.8,
    "PA": 8.1, "SC": 7.6, "GO": 7.1, "MA": 6.8, "AM": 3.9, "ES": 3.8, "PB": 4.0, "RN": 3.3,
    "MT": 3.6, "AL": 3.1, "PI": 3.3, "DF": 2.8, "MS": 2.8, "SE": 2.2, "RO": 1.6, "TO": 1.5,
    "AC": 0.8, "AP": 0.7, "RR": 0.6,
}
UFS = list(UF_RATE.keys())
DRIFT = {"slow": 0.02, "moderate": 0.05, "fast": 0.10}
POLICIES = ["none", "random", "static_topk", "dynamic_topk", "whittle", "whittle_eq", "ppo", "myopic_oracle"]
# Fatores relativos de eficacia da intervencao por nivel de risco (0=menor risco ... 4=maior risco).
# Normalizados para que a media ponderada pela distribuicao inicial seja igual a eps (comparacao justa).
#   homog: efeito igual para todos | hump: mais eficaz no risco medio | decl: mais eficaz no risco baixo
EPS_PROFILES = {"homog": [1, 1, 1, 1, 1], "hump": [0.4, 1.0, 1.6, 1.0, 0.2], "decl": [1.6, 1.3, 1.0, 0.6, 0.3]}
PAIRS = [("whittle", "dynamic_topk"), ("ppo", "dynamic_topk"), ("myopic_oracle", "dynamic_topk"),
         ("ppo", "whittle"), ("whittle_eq", "whittle"), ("dynamic_topk", "static_topk")]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def logit(p):
    return np.log(p / (1.0 - p))


@dataclass
class Cfg:
    T: int = 9
    L: int = 5
    eps: float = 0.30
    drift: float = 0.05
    down: float = 0.04
    down_boost: float = 0.10
    theta_l: float = 0.60
    theta_z: float = -0.15
    d_bar: float = 0.08
    scenario: str = "S2"
    init_levels: tuple = (0.30, 0.30, 0.20, 0.12, 0.08)
    k_frac: float = 0.05
    kappa: float = 0.5
    eps_profile: str = "homog"

    def eps_levels(self):
        base = np.array(EPS_PROFILES[self.eps_profile], float)
        f = np.interp(np.linspace(0, 1, self.L), np.linspace(0, 1, len(base)), base)
        p0 = np.array(self.init_levels, float)
        p0 /= p0.sum()
        f = f / (p0 @ f)
        return np.clip(self.eps * f, 0.0, 0.95)


class Model:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.epsl = cfg.eps_levels()
        T, L = cfg.T, cfg.L
        self.U = len(UFS)
        self.r = np.array([UF_RATE[u] for u in UFS])
        w = np.array([UF_POP[u] for u in UFS], float)
        self.w = w / w.sum()
        if cfg.scenario == "S1":
            self.ell = np.zeros(self.U)
        elif cfg.scenario == "S2":
            self.ell = np.minimum(self.r, self.r.min())
        elif cfg.scenario == "S3":
            self.ell = 0.5 * self.r
        elif cfg.scenario == "S0":   # geografia neutra: evasao latente uniforme; diferencas entre UFs = atraso administrativo
            self.ell = np.maximum(0.0, 1.0 - (1.0 - self.r) / (1.0 - cfg.d_bar))
        else:
            raise ValueError("scenario deve ser S0, S1, S2 ou S3")
        pbar = 1.0 - (1.0 - self.r) / (1.0 - self.ell)
        rel = pbar / (self.w @ pbar)
        self.d_target = np.clip(cfg.d_bar * rel, 0.002, 0.5)
        if cfg.scenario == "S0":
            self.d_target = np.full(self.U, cfg.d_bar)
        zn, zw = hermegauss(5)
        self.zn, self.zw = zn, zw / zw.sum()
        self.p0 = np.array(cfg.init_levels, float)
        self.p0 /= self.p0.sum()
        self.P = [self._P(0), self._P(1)]
        self.b = self._calibrate()
        self.d_exact = self._cum_drop(self.b)
        self.W = None
        self.indexable_frac = None

    def _P(self, a):
        c = self.cfg
        dn = c.down + c.down_boost * a
        P = np.zeros((c.L, c.L))
        for l in range(c.L):
            u = c.drift * (1 - self.epsl[l] * a) if l < c.L - 1 else 0.0
            d = dn if l > 0 else 0.0
            if l < c.L - 1:
                P[l, l + 1] = u
            if l > 0:
                P[l, l - 1] = d
            P[l, l] = 1.0 - u - d
        return P

    def hazard_grid(self, b):
        c = self.cfg
        lv = np.arange(c.L) - 2
        return sigmoid(b[:, None, None] + c.theta_l * lv[None, None, :] + c.theta_z * self.zn[None, :, None])

    def _cum_drop(self, b):
        c = self.cfg
        h = self.hazard_grid(b)
        dist = np.broadcast_to(self.p0, h.shape).copy()
        for _ in range(c.T):
            dist = (dist * (1 - h)) @ self.P[0]
        surv = dist.sum(-1)
        return 1.0 - (surv * self.zw[None, :]).sum(-1)

    def _calibrate(self):
        lo = np.full(self.U, -14.0)
        hi = np.full(self.U, 4.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            too_high = self._cum_drop(mid) > self.d_target
            hi = np.where(too_high, mid, hi)
            lo = np.where(too_high, lo, mid)
        return 0.5 * (lo + hi)

    def hazard_students(self, uf, level, z):
        c = self.cfg
        return sigmoid(self.b[uf] + c.theta_l * (level - 2) + c.theta_z * z)

    def compute_whittle(self, lam_max=3.0, iters=40):
        c = self.cfg
        T, L, U, Z = c.T, c.L, self.U, len(self.zn)
        h0 = self.hazard_grid(self.b)
        h1 = h0 * (1 - self.epsl)[None, None, :]
        P0T, P1T = self.P[0].T, self.P[1].T
        Lt = L * T
        ll = np.repeat(np.arange(L), T)
        tt = np.tile(np.arange(T), L)
        ar = np.arange(Lt)
        lo = np.zeros((U, Z, Lt))
        hi = np.full((U, Z, Lt), lam_max)
        for _ in range(iters):
            lam = 0.5 * (lo + hi)
            V = np.zeros((U, Z, Lt, L))
            D = np.zeros((T, U, Z, Lt, L))
            for t in range(T - 1, -1, -1):
                rem = T - 1 - t
                Q0 = lam[..., None] * (1 + h0[:, :, None, :] * rem) + (1 - h0[:, :, None, :]) * (1 + V @ P0T)
                Q1 = (1 - h1[:, :, None, :]) * (1 + V @ P1T) + h1[:, :, None, :] * lam[..., None] * rem
                D[t] = Q0 - Q1
                V = np.maximum(Q0, Q1)
            Dsel = D[tt, :, :, ar, ll].transpose(1, 2, 0)
            passive = Dsel >= 0
            hi = np.where(passive, lam, hi)
            lo = np.where(passive, lo, lam)
        W = hi.reshape(U, Z, L, T)
        self.W = W
        grid = np.concatenate([[0.0], np.geomspace(1e-5, lam_max, 80)])
        G = len(grid)
        lamg = np.broadcast_to(grid, (U, Z, G)).copy()
        V = np.zeros((U, Z, G, L))
        ok = np.ones((U, Z, L), bool)
        for t in range(T - 1, -1, -1):
            rem = T - 1 - t
            Q0 = lamg[..., None] * (1 + h0[:, :, None, :] * rem) + (1 - h0[:, :, None, :]) * (1 + V @ P0T)
            Q1 = (1 - h1[:, :, None, :]) * (1 + V @ P1T) + h1[:, :, None, :] * lamg[..., None] * rem
            pas = Q0 >= Q1
            ok &= np.all(np.diff(pas.astype(np.int8), axis=2) >= 0, axis=2)
            V = np.maximum(Q0, Q1)
        self.indexable_frac = float(ok.mean())
        return W


class Cohort:
    def __init__(self, model, N, rng):
        c = model.cfg
        self.N = N
        self.uf = rng.choice(model.U, size=N, p=model.w)
        self.z = rng.standard_normal(N)
        self.level0 = rng.choice(c.L, size=N, p=model.p0)
        self.zbin = np.abs(self.z[:, None] - model.zn[None, :]).argmin(1)
        self.Ud = rng.random((c.T, N))
        self.Um = rng.random((c.T, N))
        self.Ul = rng.random((c.T, N))


class Obs:
    __slots__ = ("t", "level", "uf", "z", "zbin", "flag", "alive")

    def __init__(self, t, level, uf, z, zbin, flag, alive):
        self.t, self.level, self.uf, self.z, self.zbin, self.flag, self.alive = t, level, uf, z, zbin, flag, alive


def topk_mask(score, k, eligible, rng):
    N = score.shape[0]
    m = np.zeros(N, bool)
    if k <= 0:
        return m
    s = score.astype(float) + 1e-9 * rng.random(N)
    if eligible is not None:
        s = np.where(eligible, s, -np.inf)
    n_ok = int(np.isfinite(s).sum())
    kk = min(k, n_ok)
    if kk <= 0:
        return m
    m[np.argpartition(-s, kk - 1)[:kk]] = True
    return m


def make_feats(model, level, uf, z, flag, t):
    c = model.cfg
    N = level.shape[0]
    X = np.zeros((N, c.L + 4))
    X[np.arange(N), level] = 1.0
    X[:, c.L] = model.r[uf]
    X[:, c.L + 1] = z
    X[:, c.L + 2] = flag
    X[:, c.L + 3] = t / c.T
    return X


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            s = np.sqrt(1.0 / sizes[i]) * (out_scale if i == len(sizes) - 2 else 1.0)
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1])) * s)
            self.b.append(np.zeros(sizes[i + 1]))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.step_n = 0

    def forward(self, X):
        acts = [X]
        h = X
        for i in range(len(self.W)):
            h = h @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        return h, acts

    def backward(self, acts, dout):
        gW, gb = [], []
        d = dout
        for i in reversed(range(len(self.W))):
            gW.append(acts[i].T @ d)
            gb.append(d.sum(0))
            if i > 0:
                d = (d @ self.W[i].T) * (1 - acts[i] ** 2)
        gW.reverse()
        gb.reverse()
        return gW, gb

    def adam(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.step_n += 1
        t = self.step_n
        for i in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
                p[i] -= lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + eps)


def env_step(model, level, alive, uf, z, a, Ud_t, Um_t):
    c = model.cfg
    e = model.epsl[level]
    h = model.hazard_students(uf, level, z) * (1 - e * a)
    drop = alive & (Ud_t < h)
    survive = alive & ~drop
    up = c.drift * (1 - e * a) * (level < c.L - 1)
    dn = (c.down + c.down_boost * a) * (level > 0)
    lvl_new = level + (Um_t < up).astype(int) - ((Um_t >= up) & (Um_t < up + dn)).astype(int)
    level = np.where(survive, lvl_new, level)
    return level, survive, h


def train_ppo(model, iters, n_train, seed, verbose=True, lr=2e-3, gamma=0.99, gae_lam=0.95,
              clip=0.2, epochs=4, mb=8192, ent=1e-3, eta=0.5):
    c = model.cfg
    T, k_frac = c.T, c.k_frac
    rng = np.random.default_rng(seed)
    D = c.L + 4
    pi = MLP([D, 32, 32, 1], rng, out_scale=0.1)
    pi.b[-1][:] = logit(k_frac)
    vf = MLP([D, 32, 32, 1], rng, out_scale=1.0)
    mu = 0.0
    for it in range(iters):
        coh = Cohort(model, n_train, rng)
        N = n_train
        level, alive = coh.level0.copy(), np.ones(N, bool)
        Xs = np.zeros((T, N, D))
        As = np.zeros((T, N))
        LP = np.zeros((T, N))
        Rw = np.zeros((T, N))
        Vs = np.zeros((T, N))
        for t in range(T):
            lag = coh.Ul[t] < model.ell[coh.uf]
            flag = (~alive) | lag
            X = make_feats(model, level, coh.uf, coh.z, flag.astype(float), t)
            logit_t = pi.forward(X)[0][:, 0]
            p = sigmoid(logit_t)
            a = (rng.random(N) < p).astype(float)
            Vs[t] = vf.forward(X)[0][:, 0]
            level2, alive2, h = env_step(model, level, alive, coh.uf, coh.z, a, coh.Ud[t], coh.Um[t])
            Rw[t] = alive * (1 - h)
            Xs[t], As[t] = X, a
            LP[t] = a * np.log(p + 1e-12) + (1 - a) * np.log(1 - p + 1e-12)
            level, alive = level2, alive2
        r_eff = Rw - mu * As
        adv = np.zeros((T, N))
        last = np.zeros(N)
        for t in range(T - 1, -1, -1):
            nv = Vs[t + 1] if t < T - 1 else 0.0
            delta = r_eff[t] + gamma * nv - Vs[t]
            last = delta + gamma * gae_lam * last
            adv[t] = last
        ret = (adv + Vs).reshape(-1)
        advf = adv.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        Xf, Af, LPf = Xs.reshape(-1, D), As.reshape(-1), LP.reshape(-1)
        n = Xf.shape[0]
        for _ in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                X, a, lp0, A, G = Xf[idx], Af[idx], LPf[idx], advf[idx], ret[idx]
                z, acts = pi.forward(X)
                z = z[:, 0]
                p = sigmoid(z)
                lp = a * np.log(p + 1e-12) + (1 - a) * np.log(1 - p + 1e-12)
                ratio = np.exp(lp - lp0)
                active = ((A > 0) & (ratio < 1 + clip)) | ((A < 0) & (ratio > 1 - clip))
                dz = -(A * ratio * (a - p)) * active + ent * z * p * (1 - p)
                gW, gb = pi.backward(acts, (dz / len(idx))[:, None])
                pi.adam(gW, gb, lr)
                v, vacts = vf.forward(X)
                dv = (v[:, 0] - G) / len(idx)
                gW, gb = vf.backward(vacts, dv[:, None])
                vf.adam(gW, gb, lr)
        rate = As.mean()
        mu = max(0.0, mu + eta * (rate - k_frac))
        if verbose and (it % 20 == 0 or it == iters - 1):
            print(f"  [ppo] it={it:4d} taxa_acao={rate:.4f} alvo={k_frac:.4f} mu={mu:.4f}", flush=True)
    return pi


class PolicyBase:
    name = "base"

    def reset(self, model, cohort, k, rng):
        self.model, self.cohort, self.k, self.rng = model, cohort, k, rng

    def __call__(self, o):
        raise NotImplementedError


class NonePolicy(PolicyBase):
    def __call__(self, o):
        return np.zeros(o.level.shape[0], bool)


class RandomPolicy(PolicyBase):
    def __call__(self, o):
        return topk_mask(self.rng.random(o.level.shape[0]), self.k, None, self.rng)


class StaticTopK(PolicyBase):
    def reset(self, model, cohort, k, rng):
        super().reset(model, cohort, k, rng)
        self.score = model.hazard_students(cohort.uf, cohort.level0, cohort.z)

    def __call__(self, o):
        return topk_mask(self.score, self.k, ~o.flag, self.rng)


class DynamicTopK(PolicyBase):
    def __call__(self, o):
        return topk_mask(self.model.hazard_students(o.uf, o.level, o.z), self.k, ~o.flag, self.rng)


class WhittlePolicy(PolicyBase):
    def __init__(self, equity=False, kappa=None):
        self.equity = equity
        self.kappa = kappa

    def reset(self, model, cohort, k, rng):
        super().reset(model, cohort, k, rng)
        self.wu = np.ones(model.U)
        if self.equity:
            kap = model.cfg.kappa if self.kappa is None else self.kappa
            self.wu = (model.d_target / (model.w @ model.d_target)) ** kap

    def __call__(self, o):
        s = self.model.W[o.uf, o.zbin, o.level, o.t] * self.wu[o.uf]
        return topk_mask(s, self.k, ~o.flag, self.rng)


class OraclePolicy(PolicyBase):
    def __call__(self, o):
        m = self.model
        h0 = m.hazard_students(o.uf, o.level, o.z)
        s = m.epsl[o.level] * h0 * (m.cfg.T - o.t)
        return topk_mask(s, self.k, o.alive, self.rng)


class PPOPolicy(PolicyBase):
    def __init__(self, net):
        self.net = net

    def __call__(self, o):
        X = make_feats(self.model, o.level, o.uf, o.z, o.flag.astype(float), o.t)
        z = self.net.forward(X)[0][:, 0]
        return topk_mask(z, self.k, None, self.rng)


def run_episode(model, coh, policy, k, seed):
    c = model.cfg
    rng = np.random.default_rng(seed)
    policy.reset(model, coh, k, rng)
    level, alive = coh.level0.copy(), np.ones(coh.N, bool)
    J, used = 0, 0
    for t in range(c.T):
        lag = coh.Ul[t] < model.ell[coh.uf]
        flag = (~alive) | lag
        o = Obs(t, level, coh.uf, coh.z, coh.zbin, flag, alive)
        a = policy(o)
        assert a.sum() <= k
        used += int(a.sum())
        level, alive, _ = env_step(model, level, alive, coh.uf, coh.z, a.astype(float), coh.Ud[t], coh.Um[t])
        J += int(alive.sum())
    return J, used, alive


def dual_bound(model, coh, n_grid=60, lam_hi=3.0):
    """Limite superior (valor esperado) de J em alunos-mes para qualquer politica nao antecipativa com no maximo k
    intervencoes por mes: relaxacao Lagrangiana (restricao k/mes relaxada para media temporal), minimizada em lambda."""
    c = model.cfg
    L, T = c.L, c.T
    k = int(round(c.k_frac * coh.N))
    h0 = model.hazard_students(coh.uf[:, None], np.arange(L)[None, :], coh.z[:, None])
    h1 = h0 * (1 - model.epsl)[None, :]
    P0T, P1T = model.P[0].T, model.P[1].T
    idx = np.arange(coh.N)

    def g(lam):
        V = np.zeros((coh.N, L))
        for rem in range(T):
            Q0 = lam * (1 + h0 * rem) + (1 - h0) * (1 + V @ P0T)
            Q1 = (1 - h1) * (1 + V @ P1T) + h1 * lam * rem
            V = np.maximum(Q0, Q1)
        return V[idx, coh.level0].sum() - lam * (coh.N - k) * T

    lo, hi = 0.0, lam_hi          # g(lam) e convexa em lam: busca por secao aurea
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    f1, f2 = g(x1), g(x2)
    for _ in range(24):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = g(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = g(x2)
    ub = min(f1, f2, g(0.0))
    V = np.zeros((coh.N, L))      # J esperado sem intervencao (linha de base nao ruidosa)
    for _ in range(T):
        V = (1 - h0) * (1 + V @ P0T)
    return float(ub), float(V[idx, coh.level0].sum())


def evaluate(model, policies, N, n_seeds, base_seed, min_n=30):
    k = int(round(model.cfg.k_frac * N))
    rows, uf_rows = [], []
    top5 = np.argsort(-model.r)[:5]   # 5 UFs de maior taxa de risco de continuidade (GO, DF, SP, MT, MG)
    for s in range(n_seeds):
        coh = Cohort(model, N, np.random.default_rng(base_seed + 1000 + s))
        ub, e_none = dual_bound(model, coh)
        res = {}
        for name, pol in policies.items():
            J, used, alive = run_episode(model, coh, pol, k, base_seed + 5000 + s)
            cnt = np.bincount(coh.uf, minlength=model.U)
            ret = np.bincount(coh.uf, weights=alive.astype(float), minlength=model.U) / np.maximum(cnt, 1)
            res[name] = dict(J=J, used=used, alive=int(alive.sum()), ret_uf=ret, cnt=cnt)
        for name, r in res.items():
            valid = r["cnt"] >= min_n
            extra = r["alive"] - res["none"]["alive"]
            extra_J = r["J"] - res["none"]["J"]
            ub_gain = ub - e_none
            frac_bound = extra_J / ub_gain if ub_gain > 0 else np.nan
            den = ub - res["random"]["J"]
            G = (r["J"] - res["random"]["J"]) / den if den > 0 else np.nan
            rows.append(dict(policy=name, seed=s, J_per_student=r["J"] / N, retention=r["alive"] / N,
                             extra_J=extra_J, frac_bound=frac_bound, ub_extra_J=ub_gain,
                             interventions=r["used"], extra_retained=extra,
                             retained_per_1000=(1000.0 * extra / r["used"]) if r["used"] > 0 else 0.0,
                             worst_uf=float(r["ret_uf"][valid].min()),
                             gap_uf=float(r["ret_uf"][valid].max() - r["ret_uf"][valid].min()), G=G,
                             ratio_worst_best=float(r["ret_uf"][valid].min() / r["ret_uf"][valid].max()),
                             top5_risk_ret=float((r["ret_uf"][top5] * r["cnt"][top5]).sum() / r["cnt"][top5].sum())))
            for u in range(model.U):
                uf_rows.append(dict(policy=name, seed=s, uf=UFS[u], retention=r["ret_uf"][u], n=int(r["cnt"][u])))
    return rows, uf_rows


def summarize(rows):
    out = []
    keys = ["J_per_student", "retention", "interventions", "extra_retained", "retained_per_1000",
            "worst_uf", "gap_uf", "G", "extra_J", "frac_bound", "ratio_worst_best", "top5_risk_ret"]
    for name in dict.fromkeys(r["policy"] for r in rows):
        rs = [r for r in rows if r["policy"] == name]
        if not rs:
            continue
        d = dict(policy=name, n_seeds=len(rs))
        for kx in keys:
            v = np.array([r[kx] for r in rs], float)
            m = np.nanmean(v) if not np.all(np.isnan(v)) else np.nan
            sd = np.nanstd(v, ddof=1) if len(v) > 1 and not np.all(np.isnan(v)) else 0.0
            d[kx + "_mean"] = m
            d[kx + "_ci95"] = 1.96 * sd / np.sqrt(len(v))
        out.append(d)
    return out


def print_summary(summ):
    print(f"{'politica':<14}{'J/aluno':>10}{'retencao':>10}{'extra':>9}{'por 1000':>10}{'pior UF':>9}{'gap UF':>8}"
          f"{'G':>7}{'%limite':>9}")
    for s in summ:
        print(f"{s['policy']:<14}{s['J_per_student_mean']:>10.4f}{s['retention_mean']:>10.4f}"
              f"{s['extra_retained_mean']:>9.1f}{s['retained_per_1000_mean']:>10.2f}"
              f"{s['worst_uf_mean']:>9.4f}{s['gap_uf_mean']:>8.4f}{s['G_mean']:>7.3f}"
              f"{100 * s['frac_bound_mean']:>9.1f}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_policies(model, ppo_net, kappas=()):
    pol = {"none": NonePolicy(), "random": RandomPolicy(), "static_topk": StaticTopK(),
           "dynamic_topk": DynamicTopK(), "whittle": WhittlePolicy(False),
           "whittle_eq": WhittlePolicy(True), "myopic_oracle": OraclePolicy()}
    if ppo_net is not None:
        pol["ppo"] = PPOPolicy(ppo_net)
    for kp in kappas:   # curva de Pareto retencao total x equidade: indice de Whittle reponderado por (d_u/media)^kappa
        pol[f"whittle_eq_k{kp:g}"] = WhittlePolicy(True, kappa=kp)
    return pol


def paired_diffs(rows, equiv_margin=2.0, n_boot=2000):
    """Diferencas pareadas por semente: IC95% normal e bootstrap percentil, e teste de equivalencia (TOST, alfa=5%)
    com margem +-equiv_margin alunos (aplicada so a 'alunos retidos'; a margem deve ser fixada antes dos experimentos)."""
    by = {}
    for r in rows:
        by.setdefault(r["policy"], {})[r["seed"]] = r
    brng = np.random.default_rng(12345)
    out = []
    for a, b in PAIRS:
        if a not in by or b not in by:
            continue
        seeds = sorted(set(by[a]) & set(by[b]))
        for metric, scale, label in (("extra_retained", 1.0, "alunos retidos"), ("extra_J", 1.0, "alunos-mes (J)"),
                                     ("worst_uf", 100.0, "pior UF (p.p.)")):
            x = np.array([by[a][s][metric] - by[b][s][metric] for s in seeds], float) * scale
            n = len(x)
            m = float(x.mean())
            se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            ci = 1.96 * se
            bm = x[brng.integers(0, n, size=(n_boot, n))].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            equiv = int(abs(m) + 1.645 * se < equiv_margin) if label == "alunos retidos" else -1
            out.append(dict(a=a, b=b, metric=label, mean=m, ci95=ci, boot_lo=float(lo), boot_hi=float(hi), n=n,
                            sig=int(abs(m) > ci), equiv_pm=equiv, equiv_margin=equiv_margin if equiv >= 0 else np.nan))
    return out


def print_paired(pdiff):
    print("diferencas pareadas (A - B), media +- IC95%  (* = IC exclui zero)")
    for d in pdiff:
        mark = "*" if d["sig"] else " "
        print(f"  {d['a']:<12}- {d['b']:<13}{d['metric']:<17}{d['mean']:>9.2f} +- {d['ci95']:<7.2f}{mark}")


def run_config(cfg, args, tag="", verbose=True):
    t0 = time.time()
    model = Model(cfg)
    model.compute_whittle()
    err = float(np.abs(model.d_exact - model.d_target).max())
    if verbose:
        print(f"[{tag}] cenario={cfg.scenario} perfil={cfg.eps_profile} eps={cfg.eps} drift={cfg.drift} "
              f"down_boost={cfg.down_boost} k_frac={cfg.k_frac} | erro calib.={err:.1e} | "
              f"indexabilidade={model.indexable_frac:.3f}", flush=True)
    net = None
    if not args.no_ppo:
        net = train_ppo(model, args.ppo_iters, args.ppo_n, args.seed, verbose=verbose and not args.quiet)
    kappas = [float(x) for x in args.kappa_grid.split(",") if x.strip()] if args.kappa_grid else []
    pols = build_policies(model, net, kappas)
    rows, uf_rows = evaluate(model, pols, args.N, args.seeds, args.seed)
    summ = summarize(rows)
    pdiff = paired_diffs(rows, args.equiv_margin)
    if verbose:
        print_summary(summ)
        print()
        print_paired(pdiff)
        print(f"[{tag}] tempo {time.time() - t0:.1f}s\n", flush=True)
    info = dict(indexable_frac=model.indexable_frac, calib_err=err, seconds=time.time() - t0)
    return rows, uf_rows, summ, pdiff, info


def _drift_name(v):
    return min(DRIFT, key=lambda k: abs(DRIFT[k] - v))


def map_row(cfg, summ, pdiff, info):
    sm = {s["policy"]: s for s in summ}
    pd_ = {(d["a"], d["b"], d["metric"]): d for d in pdiff}
    nan = float("nan")

    def g(a, b, key):
        d = pd_.get((a, b, "alunos retidos"))
        return d[key] if d else nan

    row = dict(scenario=cfg.scenario, profile=cfg.eps_profile, eps=cfg.eps, drift=_drift_name(cfg.drift),
               down_boost=cfg.down_boost, k_frac=cfg.k_frac,
               indexable_frac=info["indexable_frac"],
               extra_dynamic=sm["dynamic_topk"]["extra_retained_mean"],
               extra_whittle=sm["whittle"]["extra_retained_mean"],
               extra_ppo=sm["ppo"]["extra_retained_mean"] if "ppo" in sm else nan,
               extra_myopic=sm["myopic_oracle"]["extra_retained_mean"],
               bound_frac_dynamic=sm["dynamic_topk"]["frac_bound_mean"],
               bound_frac_whittle=sm["whittle"]["frac_bound_mean"],
               bound_frac_ppo=sm["ppo"]["frac_bound_mean"] if "ppo" in sm else nan)
    for a, b, nm in (("whittle", "dynamic_topk", "whittle_vs_dyn"), ("ppo", "dynamic_topk", "ppo_vs_dyn"),
                     ("myopic_oracle", "dynamic_topk", "myopic_vs_dyn"), ("dynamic_topk", "static_topk", "dyn_vs_static")):
        row[nm] = g(a, b, "mean")
        row[nm + "_ci95"] = g(a, b, "ci95")
    return row


def _sweep_worker(payload):
    cfg, args_dict = payload
    args = argparse.Namespace(**args_dict)
    _, _, summ, pdiff, info = run_config(cfg, args, verbose=False)
    return cfg, summ, pdiff, info


def print_map(maprows):
    hdr = (f"{'perfil':<7}{'drift':<9}{'down':>5}{'k':>6}{'eps':>5}{'idx':>5}{'dyn':>7}"
           f"{'W-dyn':>14}{'PPO-dyn':>14}{'%lim dyn':>10}{'%lim W':>8}")
    print(hdr)
    for r in sorted(maprows, key=lambda x: -x["whittle_vs_dyn"]):
        def f(a, b):
            if np.isnan(a):
                return f"{'--':>14}"
            return f"{a:>+8.1f}+-{b:<4.1f}" + ("*" if abs(a) > b else " ")
        print(f"{r['profile']:<7}{r['drift']:<9}{r['down_boost']:>5.2f}{r['k_frac']:>6.2f}{r['eps']:>5.2f}"
              f"{r['indexable_frac']:>5.2f}{r['extra_dynamic']:>7.1f}"
              f"{f(r['whittle_vs_dyn'], r['whittle_vs_dyn_ci95'])}{f(r['ppo_vs_dyn'], r['ppo_vs_dyn_ci95'])}"
              f"{100 * r['bound_frac_dynamic']:>10.1f}{100 * r['bound_frac_whittle']:>8.1f}")


def floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "sweep"], default="run")
    ap.add_argument("--scenario", default="S2", choices=["S0", "S1", "S2", "S3"])
    ap.add_argument("--drift", default="moderate", choices=list(DRIFT.keys()))
    ap.add_argument("--eps", type=float, default=0.30, help="eficacia media da intervencao (media ponderada pelo risco inicial)")
    ap.add_argument("--eps-profile", default="homog", choices=list(EPS_PROFILES.keys()))
    ap.add_argument("--down-boost", type=float, default=0.10, help="prob. extra mensal de descer de nivel de risco sob intervencao")
    ap.add_argument("--k-frac", type=float, default=0.05)
    ap.add_argument("--d-bar", type=float, default=0.08)
    ap.add_argument("--T", type=int, default=9)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--kappa-grid", default="", help="lista de kappas extras para a curva de Pareto, ex.: 0.25,1,2")
    ap.add_argument("--equiv-margin", type=float, default=2.0, help="margem de equivalencia (alunos retidos) do TOST")
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--ppo-iters", type=int, default=150)
    ap.add_argument("--ppo-n", type=int, default=8000)
    ap.add_argument("--no-ppo", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="processos paralelos no sweep (1 = serial; seguro no IDLE)")
    ap.add_argument("--sweep-profile", default="homog,hump,decl")
    ap.add_argument("--sweep-drift", default="slow,moderate,fast")
    ap.add_argument("--sweep-down", default="0.02,0.10,0.25")
    ap.add_argument("--sweep-k", default="0.02,0.05,0.10")
    ap.add_argument("--sweep-eps", default="0.3")
    ap.add_argument("--sweep-scen", default="S2")
    ap.add_argument("--out", default="saida_simulador")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base = Cfg(T=args.T, eps=args.eps, drift=DRIFT[args.drift], d_bar=args.d_bar, down_boost=args.down_boost,
               scenario=args.scenario, k_frac=args.k_frac, kappa=args.kappa, eps_profile=args.eps_profile)
    if args.mode == "run":
        rows, uf_rows, summ, pdiff, _ = run_config(base, args, "run")
        write_csv(os.path.join(args.out, "resultados_por_seed.csv"), rows)
        write_csv(os.path.join(args.out, "retencao_por_uf.csv"), uf_rows)
        write_csv(os.path.join(args.out, "resumo.csv"), summ)
        write_csv(os.path.join(args.out, "diferencas_pareadas.csv"), pdiff)
    else:
        cfgs = []
        for sc in args.sweep_scen.split(","):
            for prof in args.sweep_profile.split(","):
                for dr in args.sweep_drift.split(","):
                    for db in floats(args.sweep_down):
                        for kf in floats(args.sweep_k):
                            for e in floats(args.sweep_eps):
                                cfgs.append(replace(base, scenario=sc.strip(), eps_profile=prof.strip(),
                                                    drift=DRIFT[dr.strip()], down_boost=db, k_frac=kf, eps=e))
        print(f"sweep: {len(cfgs)} configuracoes, jobs={args.jobs}", flush=True)
        args_dict = {k: v for k, v in vars(args).items()}
        maprows, pairrows = [], []
        t0 = time.time()

        def consume(res, i):
            cfg, summ, pdiff, info = res
            mr = map_row(cfg, summ, pdiff, info)
            maprows.append(mr)
            for d in pdiff:
                pairrows.append(dict(scenario=cfg.scenario, profile=cfg.eps_profile, eps=cfg.eps,
                                     drift=_drift_name(cfg.drift), down_boost=cfg.down_boost, k_frac=cfg.k_frac, **d))
            write_csv(os.path.join(args.out, "sweep_mapa.csv"), maprows)
            write_csv(os.path.join(args.out, "sweep_pareado.csv"), pairrows)
            print(f"  [{i}/{len(cfgs)}] {mr['profile']} drift={mr['drift']} down={mr['down_boost']} k={mr['k_frac']} "
                  f"eps={mr['eps']} -> W-dyn={mr['whittle_vs_dyn']:+.1f}+-{mr['whittle_vs_dyn_ci95']:.1f} "
                  f"({info['seconds']:.0f}s, decorrido {time.time() - t0:.0f}s)", flush=True)

        if args.jobs > 1:
            import multiprocessing as mp
            with mp.Pool(args.jobs) as pool:
                for i, res in enumerate(pool.imap_unordered(_sweep_worker, [(c, args_dict) for c in cfgs]), 1):
                    consume(res, i)
        else:
            for i, c in enumerate(cfgs, 1):
                consume(_sweep_worker((c, args_dict)), i)
        print()
        print_map(maprows)
    print(f"Arquivos em: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

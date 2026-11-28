"""SIR resuelto con RK4 propio y ajustado por máxima verosimilitud, como en el laboratorio de EDOs."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

PASOS = 4      # pasos de RK4 por semana

def incidencia(beta, gamma, N, I0, semanas):
    """Casos nuevos por semana que predice el SIR: S(t_w) - S(t_w + 1). Tiempo en semanas."""
    h = 1.0 / PASOS
    S, I = N - I0, I0
    def f(S, I):
        inf = beta * S * I / N
        return -inf, inf - gamma * I
    salida = np.empty(semanas); S_ant = S
    for w in range(semanas):
        for _ in range(PASOS):
            a1, b1 = f(S, I)
            a2, b2 = f(S + h / 2 * a1, I + h / 2 * b1)
            a3, b3 = f(S + h / 2 * a2, I + h / 2 * b2)
            a4, b4 = f(S + h * a3, I + h * b3)
            S += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4); I += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        salida[w] = S_ant - S; S_ant = S
    return salida

def logver_binneg(y, mu, k):
    mu = np.maximum(mu, 1e-9)
    return np.sum(gammaln(y + k) - gammaln(k) - gammaln(y + 1) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))

def ajustar(y, gamma=None, inicio=None, extra=0):
    """Ajusta (R0, [gamma], N, I0, k) a la serie y. Si gamma viene dada, queda fija. Devuelve parámetros y log-verosimilitud."""
    y = np.asarray(y, float); n = len(y); total = y.sum()
    libre = gamma is None
    def desempacar(p):
        p = np.clip(p, -30, 30)
        R0 = 1 + np.exp(p[0]) if False else np.exp(p[0])
        g = np.exp(p[1]) if libre else gamma
        N = total + np.exp(p[2])                # la población efectiva es al menos los casos ya vistos
        I0 = np.exp(p[3]); k = np.exp(p[4])
        return R0 * g, g, N, min(I0, N * 0.9), k
    def menos(p):
        beta, g, N, I0, k = desempacar(p)
        mu = incidencia(beta, g, N, I0, n)
        if not np.all(np.isfinite(mu)): return 1e12
        return -logver_binneg(y, mu, k)
    arranques = [] if inicio is None else [np.array(inicio, float)]
    for r0 in (1.2, 2.0):
        for mult in (0.5, 5.0):
            arranques.append(np.array([np.log(r0), np.log(1.0), np.log(max(total * mult, 10)), np.log(max(y[0], 1.0)), np.log(10.0)]))
    mejor = None
    for x0 in arranques:
        res = minimize(menos, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 3000, "maxfev": 3000})
        if mejor is None or res.fun < mejor.fun: mejor = res
    beta, g, N, I0, k = desempacar(mejor.x)
    return {"beta": beta, "gamma": g, "R0": beta / g, "N": N, "I0": I0, "k": k, "logver": -mejor.fun, "x": mejor.x,
            "ajuste": incidencia(beta, g, N, I0, n + extra)}

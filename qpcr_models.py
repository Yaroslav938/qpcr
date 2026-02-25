"""
qpcr_models.py

Сигмоидальные модели для qPCR (L4, L5), фиттинг, расчет Ct (cpD2),
эффективности и метрик качества подгонки (RSS, R², AIC, AICc).

Идеология максимально близка к функциям pcrfit(), efficiency(),
AICc(), mselect() из пакета qpcR.[file:1][web:55]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


ModelName = Literal["L4", "L5"]


# ==========================
# МОДЕЛИ L4 / L5 (лог‑логистики)
# ==========================

def l4_model(x: np.ndarray, b: float, c: float, d: float, e: float) -> np.ndarray:
    """
    4‑параметрическая лог‑логистическая модель (аналог qpcR::l4)[file:1][web:55]:

    f(x) = d + (e - d) / (1 + exp(b * (log(x) - log(c))))

    x : циклы ( > 0 )
    b : наклон
    c : точка перегиба (cpD1 ~ cpD2)
    d : нижний асимптот (baseline)
    e : верхний асимптот (плато)
    """
    x = np.asarray(x, dtype=float)
    x = np.where(x <= 0, 0.5, x)  # защита от log(0)
    return d + (e - d) / (1.0 + np.exp(b * (np.log(x) - np.log(c))))


def l5_model(x: np.ndarray, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """
    5‑параметрическая асимметричная лог‑логистическая модель (аналог qpcR::l5)[file:1]:

    f(x) = d + (e - d) / (1 + exp(b * (log(x) - log(c))))^f

    f : параметр асимметрии.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(x <= 0, 0.5, x)
    return d + (e - d) / (1.0 + np.exp(b * (np.log(x) - np.log(c)))) ** f


# ==========================
# МЕТРИКИ КАЧЕСТВА ПОДГОНКИ
# ==========================

def gof_metrics(
    x: np.ndarray,
    y: np.ndarray,
    y_fit: np.ndarray,
    k_params: int,
) -> Tuple[float, float, float, float]:
    """
    RSS, R², AIC, AICc — формулы как в документации qpcR[file:1].

    AICc = AIC + 2k(k+1)/(n-k-1).[file:1][web:55]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_fit)
    x, y, y_fit = x[mask], y[mask], y_fit[mask]

    n = len(y)
    if n <= k_params + 1:
        return np.nan, np.nan, np.nan, np.nan

    rss = float(np.sum((y - y_fit) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    # AIC для гауссова шума[web:39]
    sigma2 = rss / n
    aic = n * np.log(sigma2) + 2 * k_params

    # коррекция Хурвича (AICc)[file:1]
    aicc = aic + (2 * k_params * (k_params + 1)) / (n - k_params - 1)

    return rss, r2, aic, aicc


# ==========================
# РЕЗУЛЬТАТ ПОДГОНКИ
# ==========================

@dataclass
class FitResult:
    model: ModelName
    params: Dict[str, float]

    # плотная сетка для графиков и производных
    x_dense: np.ndarray
    y_dense: np.ndarray

    # ключевые точки
    cpD1: Optional[float]     # максимум 1‑й производной (точка перегиба)
    cpD2: Optional[float]     # максимум 2‑й производной (Ct по qpcR::cpD2)[file:1]
    efficiency: Optional[float]  # E_n = F_n / F_{n-1} вблизи cpD2[file:1]

    # метрики качества
    rss: float
    r2: float
    aic: float
    aicc: float

    # статус
    success: bool
    message: str


# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def _clean_xy(x_raw, y_raw) -> Tuple[np.ndarray, np.ndarray]:
    """
    Очистка x, y:
    - преобразуем через pandas.to_numeric (учитывает строки с числами);
    - удаляем NaN/inf;
    - защищаемся от нулевых/отрицательных циклов (для log).
    """
    # аккуратно приводим к числам (строки '1', '2', '3' и т.п. тоже пройдут)
    x_series = pd.to_numeric(pd.Series(x_raw), errors="coerce")
    y_series = pd.to_numeric(pd.Series(y_raw), errors="coerce")

    x = x_series.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)

    # фильтруем плохие значения
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) == 0:
        return x, y

    # защита от log(0) в моделях
    x = np.where(x <= 0, 0.5, x)
    return x, y



def _has_reasonable_signal(y: np.ndarray, min_amplitude: float = 0.1) -> bool:
    """
    Проверка, что кривая не "плоская":
    размах (max - min) должен быть достаточно большим.[web:37]
    """
    if len(y) < 6:
        return False
    return (float(np.nanmax(y)) - float(np.nanmin(y))) >= min_amplitude


def _dense_grid(x: np.ndarray, n: int = 400) -> np.ndarray:
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if x_max <= x_min:
        x_max = x_min + 1.0
    return np.linspace(x_min, x_max, n)


def _compute_derivatives(x_dense: np.ndarray, y_dense: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Численные первая и вторая производные по x (np.gradient).
    """
    dy = np.gradient(y_dense, x_dense)
    d2y = np.gradient(dy, x_dense)
    return dy, d2y


def _cpD1_cpD2_from_derivatives(
    x_dense: np.ndarray,
    dy: np.ndarray,
    d2y: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    cpD1 — x при максимуме первой производной,
    cpD2 — x при максимуме второй производной (как в qpcR)[file:1].
    """
    if len(x_dense) < 5:
        return None, None

    idx1 = int(np.argmax(dy))
    idx2 = int(np.argmax(d2y))
    return float(x_dense[idx1]), float(x_dense[idx2])


def _efficiency_at_cycle(
    params: Dict[str, float],
    cpD2: float,
    model: ModelName,
) -> Optional[float]:
    """
    Оценка эффективности в окрестности cpD2, как в efficiency()[file:1]:

    E_n = F_n / F_{n-1}, где F_n — смоделированная флуоресценция
    в целочисленном цикле ближайшем к cpD2.[file:1]
    """
    if cpD2 is None or np.isnan(cpD2):
        return None

    # формируем модель по параметрам
    if model == "L4":
        b, c, d, e = (
            params["b"],
            params["c"],
            params["d"],
            params["e"],
        )
        f_model = lambda x: l4_model(x, b, c, d, e)
    else:  # L5
        b, c, d, e, f = (
            params["b"],
            params["c"],
            params["d"],
            params["e"],
            params["f"],
        )
        f_model = lambda x: l5_model(x, b, c, d, e, f)

    # целочисленные циклы
    x_min = 1
    x_max = int(np.ceil(cpD2)) + 2
    if x_max <= x_min + 1:
        return None

    cycles = np.arange(x_min, x_max + 1)
    y_vals = f_model(cycles)

    idx = int(np.argmin(np.abs(cycles - cpD2)))
    if idx == 0:
        return None

    Fn = float(y_vals[idx])
    Fprev = float(y_vals[idx - 1])
    if Fprev <= 0:
        return None

    return Fn / Fprev


# ==========================
# ФИТТИНГ ОДНОЙ КРИВОЙ
# ==========================

def fit_curve_l4(x_raw, y_raw) -> FitResult:
    """
    Подгонка L4 к одной кривой, + производные, cpD1/cpD2, эффективность, ГОФ.
    """
    x, y = _clean_xy(x_raw, y_raw)
    if len(x) < 8:
        return FitResult(
            model="L4",
            params={},
            x_dense=x,
            y_dense=y,
            cpD1=None,
            cpD2=None,
            efficiency=None,
            rss=np.nan,
            r2=np.nan,
            aic=np.nan,
            aicc=np.nan,
            success=False,
            message="Слишком мало точек для подгонки.",
        )

    if not _has_reasonable_signal(y):
        return FitResult(
            model="L4",
            params={},
            x_dense=x,
            y_dense=y,
            cpD1=None,
            cpD2=None,
            efficiency=None,
            rss=np.nan,
            r2=np.nan,
            aic=np.nan,
            aicc=np.nan,
            success=False,
            message="Сигнал слишком мал (почти плоская кривая).",
        )

    # стартовые параметры
    d0 = float(np.percentile(y, 5))
    e0 = float(np.percentile(y, 95))
    c0 = float(np.median(x))
    b0 = -1.0

    p0 = [b0, c0, d0, e0]

    # простые ограничения
    b_min, b_max = -20.0, 0.0
    c_min, c_max = float(x.min()), float(x.max())
    d_min, d_max = float(y.min() - abs(y.std())), float(y.mean())
    e_min, e_max = float(y.mean()), float(y.max() + abs(y.std()))

    bounds = (
        [b_min, c_min, d_min, e_min],
        [b_max, c_max, d_max, e_max],
    )

    try:
        popt, _ = curve_fit(
            l4_model,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
        )
    except Exception as exc:
        return FitResult(
            model="L4",
            params={},
            x_dense=x,
            y_dense=y,
            cpD1=None,
            cpD2=None,
            efficiency=None,
            rss=np.nan,
            r2=np.nan,
            aic=np.nan,
            aicc=np.nan,
            success=False,
            message=f"Не удалось подобрать L4: {exc}",
        )

    b, c, d, e = popt
    params = {"b": b, "c": c, "d": d, "e": e}

    x_dense = _dense_grid(x)
    y_dense = l4_model(x_dense, b, c, d, e)

    dy, d2y = _compute_derivatives(x_dense, y_dense)
    cpD1, cpD2 = _cpD1_cpD2_from_derivatives(x_dense, dy, d2y)
    eff = _efficiency_at_cycle(params, cpD2, model="L4")

    y_fit_on_x = l4_model(x, b, c, d, e)
    rss, r2, aic, aicc = gof_metrics(x, y, y_fit_on_x, k_params=4)

    return FitResult(
        model="L4",
        params=params,
        x_dense=x_dense,
        y_dense=y_dense,
        cpD1=cpD1,
        cpD2=cpD2,
        efficiency=eff,
        rss=rss,
        r2=r2,
        aic=aic,
        aicc=aicc,
        success=True,
        message="OK",
    )


def fit_curve_l5(x_raw, y_raw) -> FitResult:
    """
    Подгонка L5 (5‑параметрическая асимметричная лог‑логистика)[file:1].
    Логика аналогична fit_curve_l4.
    """
    x, y = _clean_xy(x_raw, y_raw)
    if len(x) < 8 or not _has_reasonable_signal(y):
        return FitResult(
            model="L5",
            params={},
            x_dense=x,
            y_dense=y,
            cpD1=None,
            cpD2=None,
            efficiency=None,
            rss=np.nan,
            r2=np.nan,
            aic=np.nan,
            aicc=np.nan,
            success=False,
            message="Недостаточно данных или сигнал слишком мал.",
        )

    d0 = float(np.percentile(y, 5))
    e0 = float(np.percentile(y, 95))
    c0 = float(np.median(x))
    b0 = -1.0
    f0 = 1.5  # умеренная асимметрия

    p0 = [b0, c0, d0, e0, f0]

    b_min, b_max = -20.0, 0.0
    c_min, c_max = float(x.min()), float(x.max())
    d_min, d_max = float(y.min() - abs(y.std())), float(y.mean())
    e_min, e_max = float(y.mean()), float(y.max() + abs(y.std()))
    f_min, f_max = 0.1, 10.0

    bounds = (
        [b_min, c_min, d_min, e_min, f_min],
        [b_max, c_max, d_max, e_max, f_max],
    )

    try:
        popt, _ = curve_fit(
            l5_model,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
        )
    except Exception as exc:
        return FitResult(
            model="L5",
            params={},
            x_dense=x,
            y_dense=y,
            cpD1=None,
            cpD2=None,
            efficiency=None,
            rss=np.nan,
            r2=np.nan,
            aic=np.nan,
            aicc=np.nan,
            success=False,
            message=f"Не удалось подобрать L5: {exc}",
        )

    b, c, d, e, f = popt
    params = {"b": b, "c": c, "d": d, "e": e, "f": f}

    x_dense = _dense_grid(x)
    y_dense = l5_model(x_dense, b, c, d, e, f)

    dy, d2y = _compute_derivatives(x_dense, y_dense)
    cpD1, cpD2 = _cpD1_cpD2_from_derivatives(x_dense, dy, d2y)
    eff = _efficiency_at_cycle(params, cpD2, model="L5")

    y_fit_on_x = l5_model(x, b, c, d, e, f)
    rss, r2, aic, aicc = gof_metrics(x, y, y_fit_on_x, k_params=5)

    return FitResult(
        model="L5",
        params=params,
        x_dense=x_dense,
        y_dense=y_dense,
        cpD1=cpD1,
        cpD2=cpD2,
        efficiency=eff,
        rss=rss,
        r2=r2,
        aic=aic,
        aicc=aicc,
        success=True,
        message="OK",
    )


def fit_curve_auto(
    x_raw,
    y_raw,
    criterion: Literal["AICc", "AIC", "R2"] = "AICc",
) -> FitResult:
    """
    Автоматический выбор лучшей модели (L4 или L5) по критерию,
    аналогично qpcR::mselect(do.all=TRUE, crit='weights'/'chisq')[file:1].

    По умолчанию выбираем модель с минимальным AICc.
    """
    res_l4 = fit_curve_l4(x_raw, y_raw)
    res_l5 = fit_curve_l5(x_raw, y_raw)

    # если обе неудачны — возвращаем ту, у которой хоть что‑то получилось
    if not res_l4.success and not res_l5.success:
        return res_l4 if np.isfinite(res_l4.aicc) else res_l5

    if criterion.upper() in ("AIC", "AICC"):
        key = "aicc" if criterion.upper() == "AICC" else "aic"
        val_l4 = getattr(res_l4, key, np.inf)
        val_l5 = getattr(res_l5, key, np.inf)
        return res_l4 if val_l4 <= val_l5 else res_l5

    if criterion.upper() == "R2":
        val_l4 = res_l4.r2 if np.isfinite(res_l4.r2) else -np.inf
        val_l5 = res_l5.r2 if np.isfinite(res_l5.r2) else -np.inf
        return res_l4 if val_l4 >= val_l5 else res_l5

    # по умолчанию — AICc
    val_l4 = res_l4.aicc if np.isfinite(res_l4.aicc) else np.inf
    val_l5 = res_l5.aicc if np.isfinite(res_l5.aicc) else np.inf
    return res_l4 if val_l4 <= val_l5 else res_l5

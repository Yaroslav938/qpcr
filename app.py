"""
app.py
Streamlit‑интерфейс к ядру Py‑qpcR
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import io
import streamlit as st
import plotly.graph_objects as go

from qpcr_data import (
    load_qpcr_csv,
    build_dataset_from_raw,
    baseline_subtract,
    QPCRDataset,
    coerce_numeric_columns,
    select_sample_columns,
)

from qpcr_models import (
    fit_curve_l4,
    fit_curve_l5,
    fit_curve_auto,
)

from qpcr_analysis import (
    batch_fit,
    calib_efficiency,
    relative_expression,
)

# ======================
# НАСТРОЙКА СТРАНИЦЫ
# ======================
st.set_page_config(
    page_title="Py-qpcR",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Py-qpcR – интерактивный аналог qpcR")

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================
def init_state():
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = None
    if "dataset_baseline" not in st.session_state:
        st.session_state["dataset_baseline"] = None
    if "batch_result" not in st.session_state:
        st.session_state["batch_result"] = None

def plot_curves(dataset: QPCRDataset, title: str, log_y: bool = False):
    df = dataset.df
    x_col = dataset.cycle_col
    y_cols = dataset.sample_cols
    fig = go.Figure()
    show_legend = len(y_cols) < 30
    for col in y_cols:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="lines+markers",
                name=col,
                marker=dict(size=4),
                line=dict(width=1),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Cycles",
        yaxis_title="Fluorescence",
        hovermode="x unified",
        height=600,
        showlegend=show_legend,
        template="plotly_white",
    )
    if log_y:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

init_state()

# ======================
# БОКОВАЯ ПАНЕЛЬ: ЗАГРУЗКА И ВЫБОР X/Y
# ======================
st.sidebar.header("1. Загрузка данных")
uploaded = st.sidebar.file_uploader(
    "CSV-файл с кривыми (разделитель , или ;)",
    type=["csv"],
)

if uploaded is not None:
    df_full = load_qpcr_csv(uploaded)
    st.session_state["dataset_baseline"] = None
    st.session_state["batch_result"] = None

    # Проверяем формат: QIAGEN если есть столбец 'sample' первым
    is_qiagen_format = ('sample' in df_full.columns and df_full.columns[0] == 'sample')
    st.session_state["is_qiagen"] = is_qiagen_format

    if is_qiagen_format:
        st.sidebar.success(f"📊 QIAGEN формат: {df_full.shape[0]} образцов × {df_full.shape[1]-1} циклов")
        st.sidebar.info("💡 Нажмите кнопку конвертации ниже для работы с кривыми амплификации")

        # Кнопка конвертации
        if st.sidebar.button("🔄 Конвертировать в нормальный формат"):
            from qpcr_data import convert_qiagen_to_normal
            try:
                df_converted = convert_qiagen_to_normal(uploaded)
                st.session_state["is_qiagen"] = False
                st.session_state["raw_df"] = df_converted

                ds_auto = build_dataset_from_raw(df_converted)
                st.session_state["dataset"] = ds_auto

                st.sidebar.success(f"✅ Конвертировано! {df_converted.shape[0]} циклов × {df_converted.shape[1]-1} образцов")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Ошибка: {e}")
                import traceback
                st.sidebar.code(traceback.format_exc())

    else:
        # Обычный формат
        st.sidebar.success(f"📈 Обычный формат: {df_full.shape[0]} строк × {df_full.shape[1]} колонок")

        # ===== ВЫБОР ДИАПАЗОНА ЯЧЕЕК ИЗ CSV =====
        st.sidebar.header("2. Диапазон из CSV (опционально)")
        use_range = st.sidebar.checkbox(
            "Использовать только часть таблицы (диапазон строк и колонок)",
            value=False,
            key="use_range_checkbox",
        )

        if use_range:
            n_rows, n_cols = df_full.shape
            all_cols = list(df_full.columns)

            row_start = st.sidebar.number_input(
                "Первая строка (1-based)",
                min_value=1,
                max_value=n_rows,
                value=1,
                step=1,
                key="range_row_start",
            )
            row_end = st.sidebar.number_input(
                "Последняя строка (1-based)",
                min_value=row_start,
                max_value=n_rows,
                value=n_rows,
                step=1,
                key="range_row_end",
            )

            col_start_name = st.sidebar.selectbox(
                "Первая колонка",
                options=all_cols,
                index=0,
                key="range_col_start",
            )
            start_idx = all_cols.index(col_start_name)
            end_options = all_cols[start_idx:]
            col_end_name = st.sidebar.selectbox(
                "Последняя колонка",
                options=end_options,
                index=len(end_options) - 1,
                key="range_col_end",
            )
            end_idx = all_cols.index(col_end_name)

            df_raw = df_full.iloc[int(row_start - 1) : int(row_end), start_idx : end_idx + 1]
            st.sidebar.info(
                f"Выбран диапазон: строки {row_start}–{row_end}, "
                f"колонки {col_start_name}…{col_end_name} "
                f"({df_raw.shape[0]}×{df_raw.shape[1]})"
            )
        else:
            df_raw = df_full

        st.session_state["raw_df"] = df_raw

        # ===== ПОСТРОЕНИЕ QPCRDataset ДЛЯ ВЫБРАННОГО ДИАПАЗОНА =====
        try:
            ds_auto = build_dataset_from_raw(df_raw)
        except Exception as e:
            st.sidebar.error(f"Ошибка при первичном анализе данных: {e}")
            ds_auto = None

        # ===== ВЫБОР X/Y, СОЗДАНИЕ dataset =====
        if ds_auto is not None:
            st.sidebar.header("3. Структура таблицы (X/Y)")
            all_cols = df_raw.columns.tolist()

            cycle_col = st.sidebar.selectbox(
                "Колонка с циклами (X):",
                options=all_cols,
                index=all_cols.index(ds_auto.cycle_col) if ds_auto.cycle_col in all_cols else 0,
                key="cycle_col_select",
            )

            df_num = coerce_numeric_columns(df_raw, exclude=[cycle_col])
            auto_samples = select_sample_columns(df_num, cycle_col)

            if not auto_samples:
                st.sidebar.error(
                    "Не найдено числовых колонок для флуоресценции в выбранном диапазоне.\n"
                    "Проверьте формат чисел или измените диапазон."
                )
            else:
                sample_cols = st.sidebar.multiselect(
                    "Пробные колонки (Y):",
                    options=[c for c in df_num.columns if c != cycle_col],
                    default=auto_samples,
                    key="sample_cols_select",
                )

                if sample_cols:
                    ds = QPCRDataset(df=df_num, cycle_col=cycle_col, sample_cols=sample_cols)
                    st.session_state["dataset"] = ds
                else:
                    st.sidebar.warning("Выберите хотя бы одну колонку как пробу.")

else:
    st.info("👈 Загрузите CSV-файл через панель слева, чтобы начать работу.")

# дальнейший код работает уже с готовым dataset
dataset: QPCRDataset = st.session_state.get("dataset")
is_qiagen = st.session_state.get("is_qiagen", False)

# ======================
# СОЗДАЁМ ВКЛАДКИ ВСЕГДА
# ======================
tab_overview, tab_baseline, tab_fit_single, tab_batch, tab_calib, tab_ratio, tab_experiment, tab_csv = st.tabs([
    "1️⃣ Обзор данных",
    "2️⃣ Baseline",
    "3️⃣ Фиттинг одной кривой",
    "4️⃣ Пакетный анализ",
    "5️⃣ Калибровка (efficiency)",
    "6️⃣ Относительная экспрессия",
    "7️⃣ Эксперимент (ΔΔCt)",
    "🔄 Excel → CSV",
])

# ========= 1. ОБЗОР ДАННЫХ =========
with tab_overview:
    if dataset is None:
        if is_qiagen:
            st.warning("⚠️ QIAGEN формат не поддерживает кривые амплификации")
            st.info("Перейдите на вкладку **7 (Эксперимент)** для анализа")
        else:
            st.info("👈 Загрузите CSV-файл через панель слева, чтобы начать работу.")
    else:
        st.subheader("Сырые данные (numeric)")
        st.write(f"Колонка циклов (X): **{dataset.cycle_col}**")
        st.write("Пробные колонки (Y):", ", ".join(dataset.sample_cols))
        plot_curves(dataset, "Сырые кривые амплификации", log_y=False)

        with st.expander("Первые строки таблицы"):
            st.dataframe(dataset.df.head(15))

# ========= 2. BASELINE =========
with tab_baseline:
    if dataset is None:
        if is_qiagen:
            st.warning("⚠️ QIAGEN формат не поддерживает эту вкладку")
            st.info("Перейдите на вкладку **7 (Эксперимент)**")
        else:
            st.info("👈 Загрузите CSV-файл через панель слева")
    else:
        st.subheader("Вычитание baseline...")

        df = dataset.df
        x = pd.to_numeric(df[dataset.cycle_col], errors="coerce")
        min_c = float(np.nanmin(x)) if np.isfinite(x).any() else 1.0
        max_c = float(np.nanmax(x)) if np.isfinite(x).any() else 40.0

        col1, col2 = st.columns(2)
        with col1:
            base_start = st.number_input(
                "Начальный цикл baseline",
                min_value=1.0,
                max_value=max_c,
                value=max(min_c, 1.0),
                step=1.0,
            )
        with col2:
            base_end = st.number_input(
                "Конечный цикл baseline",
                min_value=base_start + 1.0,
                max_value=max_c,
                value=min(base_start + 4.0, max_c),
                step=1.0,
            )

        mode = st.selectbox(
            "Режим baseline",
            options=["none", "mean", "median", "lin", "quad"],
            index=1,
        )

        base_factor = st.number_input(
            "Множитель baseline (base_factor / basefac в qpcR)",
            value=1.0,
        )

        if st.button("Применить baseline"):
            try:
                ds_corr = baseline_subtract(
                    dataset,
                    start_cycle=base_start,
                    end_cycle=base_end,
                    mode=mode,
                    base_factor=base_factor,
                )
                st.session_state["dataset_baseline"] = ds_corr
                st.success("Baseline успешно вычтен.")
                plot_curves(ds_corr, "Кривые после baseline (log‑scale)", log_y=True)

                with st.expander("Первые строки после baseline"):
                    st.dataframe(ds_corr.df.head(15))
            except Exception as e:
                st.error(f"Ошибка baseline: {e}")
        else:
            st.info("Нажмите кнопку, чтобы применить baseline-коррекцию.")

# определяем набор для фиттинга / batch (с baseline, если он есть)
dataset_fit: QPCRDataset = st.session_state["dataset_baseline"] or dataset if dataset else None

# ========= 3. ФИТТИНГ ОДНОЙ КРИВОЙ =========
with tab_fit_single:
    if dataset_fit is None:
        if is_qiagen:
            st.warning("⚠️ QIAGEN формат требует конвертации")
            st.info("Перейдите на вкладку 7 для анализа экспрессии")
        else:
            st.info("Сначала загрузите CSV-файл и выберите данные")
    else:
        st.subheader("Фиттинг одной кривой амплификации")

        target = st.selectbox("Выберите пробу:", options=dataset_fit.sample_cols)
        model_choice = st.selectbox("Модель:", options=["auto", "L4", "L5"], index=0)
        criterion = st.selectbox("Критерий для auto‑выбора модели:", options=["AICc", "AIC", "R2"], index=0)

        if st.button("Запустить фиттинг для выбранной пробы"):
            x_vals = dataset_fit.df[dataset_fit.cycle_col].values
            y_vals = dataset_fit.df[target].values

            if model_choice == "L4":
                res = fit_curve_l4(x_vals, y_vals)
            elif model_choice == "L5":
                res = fit_curve_l5(x_vals, y_vals)
            else:
                res = fit_curve_auto(x_vals, y_vals, criterion=criterion)

            if not res.success:
                st.error(f"Фиттинг не удался: {res.message}")
            else:
                st.success(f"Модель: {res.model}, сообщение: {res.message}")

                param_rows = []
                for p in ["b", "c", "d", "e", "f"]:
                    if p in res.params:
                        param_rows.append({"parameter": p, "value": res.params[p]})
                param_rows.extend([
                    {"parameter": "Ct_cpD2", "value": res.cpD2},
                    {"parameter": "Efficiency_cpD2", "value": res.efficiency},
                    {"parameter": "RSS", "value": res.rss},
                    {"parameter": "R2", "value": res.r2},
                    {"parameter": "AIC", "value": res.aic},
                    {"parameter": "AICc", "value": res.aicc},
                ])
                st.table(pd.DataFrame(param_rows))

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        name="Данные",
                        marker=dict(color="black", size=6),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=res.x_dense,
                        y=res.y_dense,
                        mode="lines",
                        name=f"Модель {res.model}",
                        line=dict(color="red", width=2),
                    )
                )
                if res.cpD2 is not None:
                    fig.add_vline(
                        x=res.cpD2,
                        line=dict(color="blue", dash="dash"),
                        annotation_text=f"Ct≈{res.cpD2:.2f}",
                        annotation_position="top left",
                    )
                fig.update_layout(
                    title=f"Фиттинг для {target}",
                    xaxis_title="Cycles",
                    yaxis_title="Fluorescence",
                    template="plotly_white",
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)

# ========= 4. ПАКЕТНЫЙ АНАЛИЗ =========
with tab_batch:
    if dataset_fit is None:
        if is_qiagen:
            st.warning("⚠️ QIAGEN формат требует конвертации")
            st.info("Перейдите на вкладку 7 для анализа экспрессии")
        else:
            st.info("Сначала загрузите CSV-файл")
    else:
        st.subheader("Пакетный анализ всех проб")

        model_choice = st.selectbox("Модель для пакетного анализа:", options=["auto", "L4", "L5"], index=0, key="batch_model_choice")
        criterion = st.selectbox("Критерий для auto‑выбора модели:", options=["AICc", "AIC", "R2"], index=0, key="batch_criterion")

        if st.button("Запустить пакетный анализ"):
            res_batch = batch_fit(
                dataset_fit,
                model=model_choice,  # type: ignore
                criterion=criterion,  # type: ignore
            )
            st.session_state["batch_result"] = res_batch
            st.success("Пакетный анализ завершён.")
            st.dataframe(res_batch.table)

            csv = res_batch.table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать результаты в CSV",
                data=csv,
                file_name="qpcr_batch_results.csv",
                mime="text/csv",
            )
        else:
            if st.session_state["batch_result"] is not None:
                st.info("Уже есть результаты последнего пакетного анализа ниже.")
                st.dataframe(st.session_state["batch_result"].table)

batch_res = st.session_state["batch_result"]

# ========= 5. КАЛИБРОВКА ЭФФЕКТИВНОСТИ =========
with tab_calib:
    if dataset_fit is None:
        if is_qiagen:
            st.warning("⚠️ QIAGEN формат требует конвертации")
            st.info("Перейдите на вкладку 7 для анализа экспрессии")
        else:
            st.info("Сначала загрузите CSV-файл")
    else:
        st.subheader("Калибровочная кривая (efficiency)")

        if batch_res is None:
            st.warning("Сначала выполните пакетный анализ (вкладка 4), чтобы получить Ct.")
        else:
            table = batch_res.table
            valid = table.dropna(subset=["Ct_cpD2"])

            if valid.empty:
                st.error("В таблице нет строк с валидным Ct_cpD2.")
            else:
                st.write("Выберите пробы, которые образуют калибровочный ряд (разведения).")
                calib_samples = st.multiselect(
                    "Пробные имена для калибровки:",
                    options=list(valid["sample"]),
                )

                if calib_samples:
                    dilutions = []
                    st.markdown("Укажите концентрации/разведения для выбранных проб (в том же порядке):")
                    for s in calib_samples:
                        val = st.number_input(
                            f"Разведение/концентрация для {s}",
                            min_value=1e-12,
                            value=1.0,
                        )
                        dilutions.append(val)

                    if st.button("Рассчитать калибровку"):
                        sub = valid[valid["sample"].isin(calib_samples)]
                        sub = sub.set_index("sample").loc[calib_samples].reset_index()
                        ct_vals = sub["Ct_cpD2"].values
                        dil_arr = np.array(dilutions, dtype=float)

                        try:
                            calib = calib_efficiency(ct_vals, dil_arr)
                        except Exception as e:
                            st.error(f"Ошибка калибровки: {e}")
                        else:
                            st.success(
                                f"Наклон: {calib.slope:.3f}, "
                                f"Эффективность: {calib.efficiency:.3f}, "
                                f"R²: {calib.r2:.3f}"
                            )

                            logd = np.log10(dil_arr)
                            ct_hat = calib.intercept + calib.slope * logd

                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=logd,
                                    y=ct_vals,
                                    mode="markers",
                                    name="Ct (данные)",
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=logd,
                                    y=ct_hat,
                                    mode="lines",
                                    name="Линейная регрессия",
                                )
                            )
                            fig.update_layout(
                                title="Калибровочная прямая Ct ~ log10(dilution)",
                                xaxis_title="log10(dilution)",
                                yaxis_title="Ct",
                                template="plotly_white",
                                height=500,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Выберите хотя бы одну пробу для калибровки.")

# ========= 6. ОТНОСИТЕЛЬНАЯ ЭКСПРЕССИЯ =========
with tab_ratio:
    if dataset_fit is None:
        if is_qiagen:
            st.warning("⚠️ QIAGEN формат требует конвертации")
            st.info("Перейдите на вкладку 7 для анализа экспрессии")
        else:
            st.info("Сначала загрузите CSV-файл")
    else:
        st.subheader("Относительная экспрессия (ΔΔCt метод)")

        if batch_res is None:
            st.info("Пакетный анализ ещё не выполнен — выполняю его автоматически (auto, AICc).")
            auto_batch = batch_fit(dataset_fit, model="auto", criterion="AICc")
            st.session_state["batch_result"] = auto_batch
            batch_res = auto_batch

        table = batch_res.table.dropna(subset=["Ct_cpD2", "Efficiency_cpD2"])

        if table.shape[0] < 2:
            st.error("Нужно минимум две пробы с валидными Ct и Efficiency.")
        else:
            target_sample = st.selectbox(
                "Target (ген/проба):",
                options=list(table["sample"]),
                key="ratio_target_select",
            )

            ref_sample = st.selectbox(
                "Reference (ген/проба):",
                options=list(table["sample"]),
                index=1 if table.shape[0] > 1 else 0,
                key="ratio_ref_select",
            )

            if target_sample == ref_sample:
                st.warning("Target и Reference должны быть разными.")
            else:
                row_t = table.set_index("sample").loc[target_sample]
                row_r = table.set_index("sample").loc[ref_sample]
                ct_t = float(row_t["Ct_cpD2"])
                ct_r = float(row_r["Ct_cpD2"])
                eff_t = float(row_t["Efficiency_cpD2"])
                eff_r = float(row_r["Efficiency_cpD2"])

                st.write(f"Ct (target) = {ct_t:.3f}, E_target = {eff_t:.3f}")
                st.write(f"Ct (ref) = {ct_r:.3f}, E_ref = {eff_r:.3f}")

                if st.button("Рассчитать относительную экспрессию (ΔCt‑подход)", key="ratio_calc_button"):
                    res_ratio = relative_expression(
                        ct_target=ct_t,
                        ct_ref=ct_r,
                        eff_target=eff_t,
                        eff_ref=eff_r,
                        mode="deltaCt",
                    )
                    st.success(
                        f"Относительная экспрессия target/ref = {res_ratio.ratio:.3f} "
                        f"(log2 = {res_ratio.log2_ratio:.3f})"
                    )
                    st.markdown(
                        "_Интерпретация: log2(ratio) > 0 — up‑regulation, "
                        "log2(ratio) < 0 — down‑regulation._"
                    )

# ========= 7. ЭКСПЕРИМЕНТ (ΔΔCt, СТАТИСТИКА) =========
with tab_experiment:
    st.subheader("Полный анализ эксперимента (автоматический режим)")

    is_qiagen = st.session_state.get("is_qiagen", False)

    if is_qiagen:
        st.warning("⚠️ QIAGEN формат обнаружен")
        st.info("Для анализа эксперимента сначала нажмите кнопку **'🔄 Конвертировать в нормальный формат'** в боковой панели")
        st.markdown("""
После конвертации:
1. Вернитесь на вкладку **4 (Пакетный анализ)** и запустите анализ
2. Затем используйте эту вкладку для ΔΔCt анализа
""")
        batch_res = None
    else:
        batch_res = st.session_state.get("batch_result")
        if batch_res is None:
            st.info("Сначала выполните пакетный анализ (вкладка 4).")

    if batch_res is not None:
        from qpcr_experiment import (
            group_replicates,
            automated_experiment_analysis,
        )

        table = batch_res.table.dropna(subset=["Ct_cpD2", "Efficiency_cpD2"])

        st.markdown("### Шаг 1: Группировка технических повторов")
        group_pattern = st.text_input(
            "Регулярное выражение для извлечения имени образца без номера повтора",
            value=r"(.+) \[\d+\]",
            key="exp_group_pattern",
            help="Пример: `(.+) \\[\\d+\\]` удалит `[1]`, `[2]` из имён типа `eef1a1a I-K-3 [1]`",
        )

        if st.button("Сгруппировать повторы", key="exp_group_button"):
            try:
                grouped = group_replicates(
                    table,
                    sample_col="sample",
                    ct_col="Ct_cpD2",
                    eff_col="Efficiency_cpD2",
                    group_pattern=group_pattern if group_pattern else None,
                )
                st.session_state["grouped_table"] = grouped
                st.success(f"✅ Группировка выполнена. Получено {len(grouped)} уникальных образцов.")

                with st.expander("Показать усреднённую таблицу"):
                    st.dataframe(grouped)
            except Exception as e:
                st.error(f"Ошибка группировки: {e}")

        grouped_table = st.session_state.get("grouped_table")

        if grouped_table is not None:
            st.markdown("---")
            st.markdown("### Шаг 2: Выбор референсных генов и контрольной группы")

            from qpcr_experiment import parse_sample_structure

            grouped_table["gene"] = grouped_table["group_name"].apply(
                lambda x: parse_sample_structure(x)[0]
            )
            grouped_table["bio_group"] = grouped_table["group_name"].apply(
                lambda x: parse_sample_structure(x)[1]
            )

            unique_genes = sorted(grouped_table["gene"].unique())
            unique_groups = sorted(grouped_table["bio_group"].unique())

            st.write(f"📊 Найдено **{len(unique_genes)} генов** и **{len(unique_groups)} биологических групп**.")

            reference_genes = st.multiselect(
                "Выберите референсные гены (housekeeping):",
                options=unique_genes,
                default=[unique_genes[0]] if unique_genes else [],
                key="exp_ref_genes",
            )

            control_group = st.selectbox(
                "Выберите контрольную группу (для расчёта ΔΔCt):",
                options=unique_groups,
                key="exp_control_group",
            )

            if reference_genes and control_group:
                if st.button("🚀 Запустить полный анализ", key="exp_auto_analysis_button"):
                    try:
                        with st.spinner("Рассчитываем ΔCt, Fold Change и статистику..."):
                            final_table = automated_experiment_analysis(
                                grouped_table=grouped_table,
                                raw_table=table,
                                reference_genes=reference_genes,
                                control_group=control_group,
                            )

                        st.session_state["final_exp_table"] = final_table
                        st.success("✅ Анализ завершён!")

                        st.markdown("---")
                        st.markdown("### 📊 Итоговая таблица результатов")
                        st.dataframe(
                            final_table.style.format({
                                "ΔCt_mean": "{:.3f}",
                                "ΔCt_sd": "{:.3f}",
                                "ΔΔCt": "{:.3f}",
                                "Fold_Change": "{:.2f}",
                                "Log2_FC": "{:.3f}",
                                "P_value": "{:.4f}",
                            }).background_gradient(subset=["Log2_FC"], cmap="RdYlGn", vmin=-3, vmax=3)
                        )

                        # Кнопки экспорта в две колонки
                        col_csv, col_excel = st.columns(2)

                        with col_csv:
                            csv = final_table.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Скачать CSV",
                                data=csv,
                                file_name="qpcr_experiment_results.csv",
                                mime="text/csv",
                                key="exp_download_csv",
                                use_container_width=True,
                            )
                         
                        with col_excel:
    # Создаем Excel файл в памяти
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                final_table.to_excel(writer, index=False, sheet_name='Results')
                            excel_data = excel_buffer.getvalue()
    
                            st.download_button(
                                label="📊 Скачать Excel",
                                data=excel_data,
                                file_name="qpcr_experiment_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="exp_download_excel",
                                use_container_width=True,
                            )


                        st.markdown("---")
                        st.markdown("### 📈 Визуализация результатов")
                        import plotly.express as px

                        # Heatmap Log2 Fold Change
                        pivot = final_table.pivot(index="Gene", columns="Group", values="Log2_FC")
                        fig = px.imshow(
                            pivot,
                            labels=dict(x="Группа", y="Ген", color="Log2 Fold Change"),
                            color_continuous_scale="RdBu_r",
                            zmin=-3,
                            zmax=3,
                            aspect="auto",
                        )
                        fig.update_layout(title="Heatmap: Log2 Fold Change по генам и группам")
                        st.plotly_chart(fig, use_container_width=True)

                        # Bar plot для одного гена
                        selected_gene = st.selectbox(
                            "Выберите ген для детального графика:",
                            options=final_table["Gene"].unique(),
                            key="exp_gene_for_plot",
                        )

                        gene_plot_data = final_table[final_table["Gene"] == selected_gene]
                        fig2 = px.bar(
                            gene_plot_data,
                            x="Group",
                            y="Fold_Change",
                            error_y="ΔCt_sd",
                            color="P_value",
                            color_continuous_scale="RdYlGn_r",
                            labels={"Fold_Change": "Fold Change", "P_value": "P-value"},
                            title=f"Fold Change для гена {selected_gene}",
                        )
                        fig2.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Baseline (FC=1)")
                        st.plotly_chart(fig2, use_container_width=True)

                        # --- Оценка стабильности референсов ---
                        st.markdown("---")
                        st.markdown("### 🧬 Стабильность референсных генов")
                        ref_mask = final_table["Gene"].isin(reference_genes)

                        if ref_mask.any():
                            ref_data = final_table[ref_mask]

                            fig_ref = px.box(
                                ref_data,
                                x="Gene",
                                y="ΔCt_mean",
                                color="Group",
                                points="all",
                                title="Распределение ΔCt референсных генов по группам",
                            )
                            st.plotly_chart(fig_ref, use_container_width=True)

                            cov_df = (
                                ref_data
                                .groupby("Gene")["ΔCt_mean"]
                                .agg(["mean", "std"])
                                .reset_index()
                            )
                            cov_df["CoV_%"] = (cov_df["std"] / cov_df["mean"].abs()) * 100
                            st.markdown("Коэффициент вариации ΔCt для референсов (чем меньше, тем лучше):")
                            st.dataframe(cov_df[["Gene", "CoV_%"]].style.format({"CoV_%": "{:.1f}"}))
                        else:
                            st.info("Референсные гены не попали в итоговую таблицу.")

                        # --- Boxplot ΔCt по генам (на сырых повторах) ---
                        st.markdown("---")
                        st.markdown("### 📦 Ящик с усами: распределение ΔCt по генам")

                        import re
                        raw_for_box = table.copy()
                        raw_for_box["gene"] = raw_for_box["sample"].apply(
                            lambda x: parse_sample_structure(x)[0]
                        )

                        def clean_bio_group(sample_name: str) -> str:
                            cleaned = re.sub(r'\s*\[\d+\]$', '', sample_name.strip())
                            parts = cleaned.split()
                            if len(parts) >= 2:
                                return " ".join(parts[1:])
                            return "Unknown"

                        raw_for_box["bio_group"] = raw_for_box["sample"].apply(clean_bio_group)

                        raw_ref = raw_for_box[raw_for_box["gene"].isin(reference_genes)]
                        ref_ct_by_group = raw_ref.groupby("bio_group")["Ct_cpD2"].mean().to_dict()

                        raw_for_box["ct_ref"] = raw_for_box["bio_group"].map(ref_ct_by_group)
                        raw_for_box["delta_ct"] = raw_for_box["Ct_cpD2"] - raw_for_box["ct_ref"]

                        box_data_raw = raw_for_box[
                            ~raw_for_box["gene"].isin(reference_genes)
                        ].dropna(subset=["delta_ct"])

                        groups_for_box = st.multiselect(
                            "Выберите группы для отображения:",
                            options=sorted(box_data_raw["bio_group"].unique()),
                            default=sorted(box_data_raw["bio_group"].unique()),
                            key="exp_groups_for_box",
                        )

                        box_plot_data = box_data_raw[box_data_raw["bio_group"].isin(groups_for_box)]
                        fig_box = px.box(
                            box_plot_data,
                            x="gene",
                            y="delta_ct",
                            color="bio_group",
                            points="all",
                            title="Распределение ΔCt для каждого технического повтора",
                            labels={"gene": "Ген", "delta_ct": "ΔCt", "bio_group": "Группа"},
                        )
                        st.plotly_chart(fig_box, use_container_width=True)

                        # --- Volcano plot ---
                        st.markdown("---")
                        st.markdown("### 🌋 Volcano plot (Log2 FC vs -log10 p-value)")
                        volcano_data = final_table[final_table["Group"] != control_group].copy()
                        volcano_data["neg_log10_p"] = -np.log10(volcano_data["P_value"].replace(0, np.nan))

                        p_thresh = 0.05
                        log2fc_thresh = 1.0
                        volcano_data["Significant"] = (
                            (volcano_data["P_value"] < p_thresh) &
                            (volcano_data["Log2_FC"].abs() >= log2fc_thresh)
                        )

                        fig_volcano = px.scatter(
                            volcano_data,
                            x="Log2_FC",
                            y="neg_log10_p",
                            color="Significant",
                            hover_data=["Gene", "Group", "Fold_Change", "P_value"],
                            color_discrete_map={True: "red", False: "gray"},
                            labels={
                                "Log2_FC": "Log2 Fold Change",
                                "neg_log10_p": "-log10(p-value)",
                            },
                            title=f"Volcano plot (контроль: {control_group})",
                        )
                        fig_volcano.add_vline(x=log2fc_thresh, line_dash="dash", line_color="blue")
                        fig_volcano.add_vline(x=-log2fc_thresh, line_dash="dash", line_color="blue")
                        fig_volcano.add_hline(y=-np.log10(p_thresh), line_dash="dash", line_color="green")
                        st.plotly_chart(fig_volcano, use_container_width=True)

                        # --- Резюме ---
                        st.markdown("---")
                        st.markdown("### 📄 Резюме по генам (значимые изменения)")
                        summary = final_table.copy()
                        summary["Significant"] = (
                            (summary["P_value"] < 0.05) &
                            (summary["Log2_FC"].abs() >= 1.0)
                        )

                        summary_short = summary[
                            ["Gene", "Group", "Fold_Change", "Log2_FC", "P_value", "Significant"]
                        ].sort_values(["Gene", "Group"])

                        st.dataframe(
                            summary_short.style.format({
                                "Fold_Change": "{:.2f}",
                                "Log2_FC": "{:.2f}",
                                "P_value": "{:.4f}",
                            })
                        )

                    except Exception as e:
                        st.error(f"Ошибка анализа: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# ===========================
# TAB 8: Конвертер Excel → CSV
# ===========================
with tab_csv:
    st.header("🔄 Конвертер Excel в CSV")
    st.markdown("""
Загрузите файл Excel (.xlsx или .xls), выберите нужный лист и скачайте его в формате CSV.
""")

    uploaded_excel = st.file_uploader(
        "Выберите Excel-файл",
        type=["xlsx", "xls"],
        key="excel_converter"
    )

    if uploaded_excel is not None:
        try:
            excel_file = pd.ExcelFile(uploaded_excel)
            sheet_names = excel_file.sheet_names
            st.success(f"✅ Файл загружен: **{uploaded_excel.name}**")
            st.info(f"Найдено листов: **{len(sheet_names)}**")

            selected_sheet = st.selectbox(
                "Выберите лист для конвертации:",
                options=sheet_names,
                key="sheet_selector"
            )

            df_excel = pd.read_excel(uploaded_excel, sheet_name=selected_sheet)
            st.markdown(f"### 📋 Превью листа: `{selected_sheet}`")
            st.markdown(f"**Размер:** {df_excel.shape[0]} строк × {df_excel.shape[1]} столбцов")

            preview_rows = st.slider(
                "Количество строк для превью:",
                min_value=5,
                max_value=min(100, len(df_excel)),
                value=min(10, len(df_excel)),
                key="preview_slider"
            )
            st.dataframe(df_excel.head(preview_rows), use_container_width=True)

            st.markdown("---")
            st.markdown("### ⚙️ Настройки экспорта")
            col1, col2 = st.columns(2)

            with col1:
                csv_separator = st.selectbox(
                    "Разделитель CSV:",
                    options=[",", ";", "\t", "|"],
                    format_func=lambda x: {",": "Запятая (,)", ";": "Точка с запятой (;)", "\t": "Табуляция (\\t)", "|": "Вертикальная черта (|)"}[x],
                    key="csv_sep"
                )

            with col2:
                include_index = st.checkbox(
                    "Включить индекс строк",
                    value=False,
                    key="csv_index"
                )

            csv_data = df_excel.to_csv(
                sep=csv_separator,
                index=include_index,
                encoding="utf-8"
            ).encode("utf-8")

            original_name = uploaded_excel.name.rsplit(".", 1)[0]
            csv_filename = f"{original_name}_{selected_sheet}.csv"

            st.download_button(
                label="⬇️ Скачать CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                key="download_csv"
            )

            st.success(f"✅ Готово! Нажмите кнопку выше для скачивания `{csv_filename}`")

        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {e}")
            st.exception(e)
    else:
        st.info("👆 Загрузите Excel-файл для начала работы")

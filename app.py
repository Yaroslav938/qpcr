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
)

from qpcr_models import (
    fit_curve_l4,
    fit_curve_l5,
    fit_curve_auto,
    l4_model,
    l5_model,
)

from qpcr_analysis import (
    batch_fit,
    calib_efficiency,
    relative_expression,
)

from qpcr_experiment import (
    group_replicates,
    automated_experiment_analysis,
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
    if "ds" not in st.session_state:
        st.session_state["ds"] = None
    if "ds_base" not in st.session_state:
        st.session_state["ds_base"] = None
    if "batch_result" not in st.session_state:
        st.session_state["batch_result"] = None
    if "grouped_table" not in st.session_state:
        st.session_state["grouped_table"] = None
    if "final_exp_table" not in st.session_state:
        st.session_state["final_exp_table"] = None


init_state()

def plot_curves(ds: QPCRDataset, title="Кривые амплификации", log_y=False):
    """
    Отрисовка всех (или выбранных) образцов из QPCRDataset.
    """
    fig = go.Figure()
    
    # Ось X — циклы
    x = pd.to_numeric(ds.df[ds.cycle_col], errors="coerce").values
    
    for col in ds.sample_cols:
        y = pd.to_numeric(ds.df[col], errors="coerce").values
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=col
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Cycle",
        yaxis_title="Fluorescence",
        hovermode="x unified",
        template="plotly_white",
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


# ======================
# БОКОВАЯ ПАНЕЛЬ (ЗАГРУЗКА)
# ======================
with st.sidebar:
    st.header("1. Загрузка данных (Сырой экспорт)")
    st.info("Поддерживаются CSV из Rotor-Gene Q и других приборов.")
    
    uploaded_file = st.file_uploader("Загрузите файл экспорта (CSV/txt/rex)", type=["csv", "txt", "rex"])
    
    if uploaded_file is not None:
        try:
            # Универсальная безопасная распаковка
            load_result = load_qpcr_csv(uploaded_file)
            
            if isinstance(load_result, tuple) and len(load_result) == 2:
                df_raw, is_qiagen = load_result
            else:
                df_raw = load_result
                is_qiagen = False
                
                first_col_name = str(df_raw.columns[0]).strip(' "')
                if "Excel Raw Data Export" in first_col_name or "QIAGEN" in first_col_name or "Unnamed" in first_col_name:
                    try:
                        if df_raw.astype(str).apply(lambda col: col.str.contains("Page 1", na=False)).any().any():
                            is_qiagen = True
                    except:
                        pass
                elif "Page 1" in df_raw.columns or "Cycle" in df_raw.columns:
                    is_qiagen = False
                        
            st.session_state["raw_df"] = df_raw
            st.session_state["is_qiagen"] = is_qiagen
            
            st.success("Файл успешно загружен!")
            
            if is_qiagen:
                st.warning("Обнаружен формат QIAGEN Rotor-Gene")
                if st.button("🔄 Конвертировать в нормальный формат"):
                    try:
                        df_raw = df_raw.dropna(how='all', axis=1).dropna(how='all', axis=0)
                        header_row_idx = None
                        for idx, row in df_raw.iterrows():
                            if any(isinstance(val, str) and 'Page 1' in val for val in row.values):
                                header_row_idx = idx
                                break
                        
                        if header_row_idx is not None:
                            df_raw.columns = df_raw.iloc[header_row_idx]
                            df_raw = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
                            df_raw = df_raw.rename(columns={'Page 1': 'Cycle'})
                            
                            st.session_state["raw_df"] = df_raw
                            st.session_state["is_qiagen"] = False
                            st.success("Конвертация выполнена успешно! Теперь можно распознать структуру.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка конвертации: {e}")
            
            st.subheader("Настройки формата")
            cycle_col = st.selectbox(
                "Укажите колонку с циклами (Cycle):", 
                options=["auto"] + list(st.session_state["raw_df"].columns)
            )
            cycle_col_val = None if cycle_col == "auto" else cycle_col
            
            if st.button("Распознать структуру"):
                ds = build_dataset_from_raw(st.session_state["raw_df"], cycle_col=cycle_col_val)
                st.session_state["ds"] = ds
                st.session_state["ds_base"] = None
                st.session_state["batch_result"] = None
                st.success(f"Распознано {len(ds.sample_cols)} образцов!")
                
        except Exception as e:
            st.error(f"Ошибка чтения: {e}")

# ======================
# ОСНОВНОЕ ОКНО (ВКЛАДКИ)
# ======================

tab_overview, tab_baseline, tab_fit_single, tab_batch, tab_calib, tab_ratio, tab_experiment, tab_multi, tab_csv = st.tabs([
    "Обзор данных", "Базовая линия", "Одиночный фиттинг",
    "Пакетный фиттинг", "Калибровка (E)", "Ratio (Простое)",
    "Анализ эксперимента", "Мульти-планшет", "CSV Конвертер"
])

# --------------------------
# Вкладка 1: Обзор данных
# --------------------------
with tab_overview:
    st.markdown("### Исходная таблица")
    if st.session_state["raw_df"] is not None:
        st.dataframe(st.session_state["raw_df"].head(15))
    else:
        st.info("Сначала загрузите файл слева.")

    if st.session_state["ds"] is not None:
        st.markdown("### Выбор колонок для анализа")
        ds = st.session_state["ds"]
        
        selected_cols = st.multiselect(
            "Выберите образцы (по умолчанию выбраны все):",
            options=ds.sample_cols,
            default=ds.sample_cols
        )
        
        if len(selected_cols) < len(ds.sample_cols):
             new_ds = QPCRDataset(
                 df=ds.df.copy(), 
                 cycle_col=ds.cycle_col, 
                 sample_cols=selected_cols
             )
             st.session_state["ds"] = new_ds
             st.warning("Набор образцов обновлён. Рекомендуется заново провести базовую линию.")
             st.rerun()

        log_scale = st.checkbox("Логарифмическая шкала Y (Raw)", value=False)
        st.plotly_chart(plot_curves(st.session_state["ds"], log_y=log_scale), use_container_width=True)

# --------------------------
# Вкладка 2: Базовая линия
# --------------------------
with tab_baseline:
    st.markdown("### Вычитание фона (Baseline Subtraction)")
    st.write("На ранних циклах флуоресценция часто имеет фоновый шум или тренд. Выберите метод вычитания.")
    
    if st.session_state["ds"] is not None:
        ds = st.session_state["ds"]
        
        col1, col2 = st.columns(2)
        with col1:
            base_mode = st.selectbox(
                "Метод базовой линии:",
                ["lin", "quad", "mean", "median", "none"],
                format_func=lambda x: {
                    "lin": "Линейный тренд (рекомендуется)",
                    "quad": "Квадратичный тренд",
                    "mean": "Среднее (по первым циклам)",
                    "median": "Медиана (по первым циклам)",
                    "none": "Без вычитания"
                }[x]
            )
        with col2:
            base_cycles = st.slider("Циклы для оценки фона (обычно 1-10 или 1-15):", 1, 30, (1, 10))
            
        if st.button("Применить вычитание фона", type="primary"):
            ds_base = baseline_subtract(
                ds, 
                start_cycle=float(base_cycles[0]),
                end_cycle=float(base_cycles[1]),
                mode=base_mode
            )
            st.session_state["ds_base"] = ds_base
            st.success("Базовая линия успешно вычтена!")
            
        if st.session_state["ds_base"] is not None:
            log_scale_b = st.checkbox("Логарифмическая шкала Y (Baseline)", value=False)
            st.plotly_chart(plot_curves(st.session_state["ds_base"], title="Кривые после вычитания фона", log_y=log_scale_b), use_container_width=True)
            
    else:
        st.info("Сначала распознайте структуру данных на первой вкладке.")

# --------------------------
# Вкладка 3: Одиночный фиттинг
# --------------------------
with tab_fit_single:
    st.markdown("### Фиттинг одной кривой")
    st.write("Изучите детальную информацию по подгонке конкретного образца.")
    
    ds_b = st.session_state.get("ds_base") or st.session_state.get("ds")
    
    if ds_b is not None:
        sample_to_fit = st.selectbox("Выберите образец:", options=ds_b.sample_cols)
        model_type = st.radio("Модель:", ["auto", "L4", "L5"], horizontal=True, 
                              help="auto - выбор лучшей модели по критерию AICc")
        
        if st.button("Подогнать кривую"):
            x_raw = pd.to_numeric(ds_b.df[ds_b.cycle_col], errors="coerce").values
            y_raw = pd.to_numeric(ds_b.df[sample_to_fit], errors="coerce").values
            
            mask = pd.notna(x_raw) & pd.notna(y_raw)
            x_raw, y_raw = x_raw[mask], y_raw[mask]
            
            if model_type == "L4":
                res = fit_curve_l4(x_raw, y_raw)
            elif model_type == "L5":
                res = fit_curve_l5(x_raw, y_raw)
            else:
                res = fit_curve_auto(x_raw, y_raw)
            
            if res.success:
                st.success(f"Фиттинг успешен! Выбрана модель: **{res.model}**")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("#### Ключевые метрики:")
                    st.write(f"- **Ct (cpD2)**: `{res.cpD2:.2f}` (максимум второй производной)")
                    st.write(f"- **Эффективность (E)**: `{res.efficiency:.3f}`")
                    st.write(f"- **R²**: `{res.r2:.4f}`")
                    st.write(f"- **AICc**: `{res.aicc:.2f}`")
                    
                    st.markdown("#### Параметры уравнения:")
                    param_rows = [{"Параметр": k, "Значение": round(v, 4)} for k, v in res.params.items()]
                    st.table(pd.DataFrame(param_rows))
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_raw, y=y_raw, mode='markers', name='Эксперимент'))
                    
                    x_smooth = np.linspace(min(x_raw), max(x_raw), 200)
                    if res.model == "L4":
                        y_smooth = l4_model(x_smooth, **res.params)
                    else:
                        y_smooth = l5_model(x_smooth, **res.params)
                        
                    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=f'Модель {res.model}'))
                    
                    if res.cpD2 is not None and not np.isnan(res.cpD2):
                        if res.model == "L4":
                            y_ct = l4_model(np.array([res.cpD2]), **res.params)[0]
                        else:
                            y_ct = l5_model(np.array([res.cpD2]), **res.params)[0]
                            
                        fig.add_trace(go.Scatter(
                            x=[res.cpD2], y=[y_ct], 
                            mode='markers', 
                            marker=dict(size=12, color='red', symbol='x'),
                            name=f'Ct = {res.cpD2:.2f}'
                        ))
                    
                    fig.update_layout(title=f"Аппроксимация для {sample_to_fit}", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Фиттинг не удался. Ошибка: {res.message}")
    else:
        st.info("Нет данных для фиттинга. Вернитесь на Шаг 1.")

# --------------------------
# Вкладка 4: Пакетный фиттинг
# --------------------------
with tab_batch:
    st.markdown("### Пакетная обработка всех кривых (Batch Fit)")
    st.write("Автоматически применяет выбранную модель ко всем образцам датасета и собирает результаты в одну таблицу.")
    
    ds_b = st.session_state.get("ds_base") or st.session_state.get("ds")
    
    if ds_b is not None:
        batch_model = st.radio("Модель для всех образцов:", ["auto", "L4", "L5"], horizontal=True, key="batch_model")
        
        if st.button("Запустить Batch Fit", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Идёт расчет...")
            
            res_batch = batch_fit(ds_b, model=batch_model)
            
            progress_bar.progress(100)
            status_text.text("Расчет завершён!")
            st.session_state["batch_result"] = res_batch
            
            total_samples = len(res_batch.table)
            success_count = res_batch.table["Ct_cpD2"].notna().sum() if "Ct_cpD2" in res_batch.table.columns else total_samples
            failed_count = total_samples - success_count
            
            st.success(f"Всего обработано: {total_samples}. Успешно: {success_count}, Ошибок: {failed_count}.")
            
            # Форматируем отображение R2 до 10 знаков
            r2_cols = [c for c in res_batch.table.columns if c.lower() == 'r2']
            styled_table = res_batch.table.style.format({col: "{:.10f}" for col in r2_cols})
            
            st.dataframe(styled_table)
            
            csv = res_batch.table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Скачать таблицу результатов (CSV)", 
                data=csv, 
                file_name="batch_fit_results.csv", 
                mime="text/csv"
            )
            
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    res_batch.table.to_excel(writer, index=False, sheet_name='Batch Fit Results')
                
                st.download_button(
                    label="⬇️ Скачать таблицу результатов (Excel)",
                    data=buffer.getvalue(),
                    file_name="batch_fit_results.xlsx",
                    mime="application/vnd.ms-excel",
                    key="batch_dl_excel"
                )
            except ImportError:
                st.warning("💡 Установите библиотеку `xlsxwriter` (добавьте в requirements.txt) для генерации Excel-файлов.")
            
    else:
        st.info("Нет данных. Вернитесь на Шаг 1.")
        
    if st.session_state.get("batch_result") is not None:
        with st.expander("Показать таблицу результатов из памяти", expanded=False):
            mem_table = st.session_state["batch_result"].table
            r2_cols_mem = [c for c in mem_table.columns if c.lower() == 'r2']
            styled_mem_table = mem_table.style.format({col: "{:.10f}" for col in r2_cols_mem})
            st.dataframe(styled_mem_table)

# --------------------------
# Вкладка 5: Калибровка (E)
# --------------------------
with tab_calib:
    st.markdown("### Оценка эффективности по калибровочной кривой")
    st.markdown("Оцените эффективность ПЦР ($E$) классическим методом стандартной кривой ($Ct$ от $\log_{10}$ разведения).")
    
    batch_res = st.session_state.get("batch_result")
    
    if batch_res is not None:
        table = batch_res.table
        valid = table.dropna(subset=["Ct_cpD2"])
        
        selected_samples = st.multiselect("Выберите образцы стандартного ряда:", options=valid["sample"].tolist())
        dilutions_str = st.text_input("Введите факторы разведения (соответственно порядку образцов, через запятую):", value="1, 0.1, 0.01, 0.001")
        
        if st.button("Построить калибровочную кривую"):
            try:
                dilutions = [float(x.strip()) for x in dilutions_str.split(",")]
                if len(selected_samples) != len(dilutions):
                    st.error(f"Количество выбранных образцов ({len(selected_samples)}) не совпадает с количеством введенных разведений ({len(dilutions)}).")
                else:
                    cts = valid.set_index("sample").loc[selected_samples, "Ct_cpD2"].values
                    eff_res = calib_efficiency(cts, dilutions)
                    
                    st.success(f"Эффективность (E): **{eff_res.efficiency:.3f}** (в идеале 2.0 = 100%)")
                    st.info(f"R² регрессии: {eff_res.r2:.4f} | Slope: {eff_res.slope:.3f}")
                    
                    fig = go.Figure()
                    log_dils = np.log10(dilutions)
                    fig.add_trace(go.Scatter(x=log_dils, y=cts, mode='markers', name='Образцы', marker=dict(size=12, color='royalblue')))
                    
                    x_line = np.array([min(log_dils), max(log_dils)])
                    y_line = eff_res.intercept + eff_res.slope * x_line
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Тренд', line=dict(color='firebrick', dash='dash')))
                    
                    fig.update_layout(
                        title="Калибровочная кривая: Ct vs log10(Разведение)",
                        xaxis_title="log10(Dilution)",
                        yaxis_title="Ct (cpD2)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка при расчете: {e}")
    else:
        st.info("Сначала выполните Пакетный фиттинг во вкладке 'Пакетный фиттинг'.")

# --------------------------
# Вкладка 6: Ratio (Простое)
# --------------------------
with tab_ratio:
    st.markdown("### Относительная экспрессия (Simple Ratio)")
    st.markdown("Быстрый расчёт $Ratio = E_{target}^{\\Delta Ct_{target}} / E_{ref}^{\\Delta Ct_{ref}}$ для конкретной пары лунок.")
    
    batch_res = st.session_state.get("batch_result")
    if batch_res is not None:
        table = batch_res.table.dropna(subset=["Ct_cpD2", "Efficiency_cpD2"])
        
        if table.shape[0] < 2:
            st.warning("Недостаточно данных для сравнения (нужно минимум 2 успешные лунки).")
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_sample = st.selectbox(
                    "Target (Целевой образец):", 
                    options=list(table["sample"]), 
                    index=0
                )
            with col2:
                ref_sample = st.selectbox(
                    "Reference (Референсный образец):", 
                    options=list(table["sample"]), 
                    index=1 if table.shape[0] > 1 else 0,
                )
            
            if st.button("Рассчитать Ratio"):
                row_t = table.set_index("sample").loc[target_sample]
                row_r = table.set_index("sample").loc[ref_sample]
                
                res_ratio = relative_expression(
                    ct_target=row_t["Ct_cpD2"],
                    ct_ref=row_r["Ct_cpD2"],
                    eff_target=row_t["Efficiency_cpD2"],
                    eff_ref=row_r["Efficiency_cpD2"],
                    mode="deltaCt"
                )
                
                st.success(f"Ratio (Target/Ref): **{res_ratio.ratio:.4f}**")
                st.info(f"Log2(Ratio): {res_ratio.log2_ratio:.4f}")
                
                st.markdown("**Детали расчёта:**")
                st.write(f"- **{target_sample}**: Ct = {row_t['Ct_cpD2']:.2f}, E = {row_t['Efficiency_cpD2']:.3f}")
                st.write(f"- **{ref_sample}**: Ct = {row_r['Ct_cpD2']:.2f}, E = {row_r['Efficiency_cpD2']:.3f}")
    else:
        st.info("Сначала выполните Пакетный фиттинг.")

# --------------------------
# Вкладка 7: Анализ эксперимента
# --------------------------
with tab_experiment:
    st.subheader("Полный анализ эксперимента (автоматический режим)")

    is_qiagen = st.session_state.get("is_qiagen", False)

    if is_qiagen:
        st.warning("⚠️ Формат сырых данных (требуется обработка)")
        st.info("Для анализа эксперимента сначала нажмите кнопку **'🔄 Конвертировать в нормальный формат'** в боковой панели")
        st.markdown("""
После конвертации:
1. Вернитесь на вкладку **4 (Пакетный анализ)** и запустите анализ
2. Затем используйте эту вкладку для ΔΔCt анализа
        """)
    else:
        batch_res = st.session_state.get("batch_result")
        
        if batch_res is not None:
            table = batch_res.table.dropna(subset=["Ct_cpD2", "Efficiency_cpD2"])
            
            st.markdown("### Шаг 1. Группировка технических реплик")
            st.markdown("Введите регулярное выражение. В новой версии работает **умная автогруппировка**, которая сама отрежет технические хвосты `[1]`, `[2]` или `.1`.")
            
            group_pattern = st.text_input(
                "Регулярное выражение (оставьте пустым для автогруппировки):",
                value="",
                key="exp_group_pattern",
                help="Например: `^(.*?)\\s*\\[\\d+\\]` отрезает номер реплики в квадратных скобках"
            )
            
            if st.button("Сгруппировать"):
                grouped = group_replicates(
                    table,
                    sample_col="sample",
                    ct_col="Ct_cpD2",
                    eff_col="Efficiency_cpD2",
                    group_pattern=group_pattern if group_pattern else None,
                )
                st.session_state["grouped_table"] = grouped
                st.success(f"Группировка выполнена! Найдено уникальных групп: {len(grouped)}")
            
            grouped_table = st.session_state.get("grouped_table")
            
            if grouped_table is not None:
                with st.expander("Показать результаты группировки", expanded=False):
                    st.dataframe(grouped_table)
                
                st.markdown("### Шаг 2. Расчет относительной экспрессии")
                st.info("Ожидается структура имен групп: **Ген Группа** (напр., `eef1a1a I-P-1`)")
                
                try:
                    grouped_table["gene"] = grouped_table["group_name"].apply(
                        lambda x: str(x).split(" ")[0]
                    )
                    grouped_table["bio_group"] = grouped_table["group_name"].apply(
                        lambda x: " ".join(str(x).split(" ")[1:]) if len(str(x).split(" ")) > 1 else "Unknown"
                    )
                    
                    unique_genes = sorted(grouped_table["gene"].unique())
                    unique_groups = sorted(grouped_table["bio_group"].unique())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        reference_genes = st.multiselect(
                            "Выберите референсные гены (один или несколько):", 
                            options=unique_genes
                        )
                    with col2:
                        control_group = st.selectbox(
                            "Выберите контрольную группу (Control):", 
                            options=unique_groups
                        )
                    
                    if len(reference_genes) > 0 and control_group:
                        st.checkbox("Использовать регрессионную нормализацию (Wang et al., 2015)", value=True, key="exp_use_reg")
                        if st.button("🚀 Рассчитать экспрессию", key="exp_calc_btn", type="primary"):
                            try:
                                use_regression = st.session_state.get("exp_use_reg", True)
                                final_table = automated_experiment_analysis(
                                    grouped_table=grouped_table,
                                    raw_table=table,
                                    reference_genes=reference_genes,
                                    control_group=control_group,
                                    use_regression_norm=use_regression
                                )
                                
                                st.session_state["final_exp_table"] = final_table
                                st.success("Расчёт завершён успешно!")
                                
                                st.markdown("#### Итоговая таблица результатов")
                                st.dataframe(
                                    final_table.style.format({
                                        "Regr_b": "{:.3f}",
                                        "ΔCt_mean": "{:.2f}",
                                        "ΔCt_sd": "{:.2f}",
                                        "ΔΔCt": "{:.2f}",
                                        "Fold_Change": "{:.3f}",
                                        "Log2_FC": "{:.2f}",
                                        "P_value": "{:.4f}"
                                    }).applymap(
                                        lambda v: 'background-color: lightgreen' if isinstance(v, float) and v < 0.05 else '', 
                                        subset=['P_value']
                                    )
                                )
                                
                                csv = final_table.to_csv(sep=";", index=False, decimal=",").encode("utf-8")
                                st.download_button(
                                    "⬇️ Скачать сводку (CSV)", 
                                    data=csv, 
                                    file_name="experiment_results.csv", 
                                    mime="text/csv"
                                )
                                
                                try:
                                    buffer = io.BytesIO()
                                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                        final_table.to_excel(writer, index=False, sheet_name='Results')
                                        grouped_table.to_excel(writer, index=False, sheet_name='Grouped Data')
                                    
                                    st.download_button(
                                        label="⬇️ Скачать полный отчет (Excel)",
                                        data=buffer.getvalue(),
                                        file_name="experiment_report.xlsx",
                                        mime="application/vnd.ms-excel"
                                    )
                                except ImportError:
                                    st.warning("💡 Установите библиотеку `xlsxwriter` (добавьте в requirements.txt) для генерации Excel-файлов.")
                                
                            except Exception as e:
                                st.error(f"Ошибка при расчёте экспрессии: {e}")
                except Exception as e:
                    st.warning(f"Не удалось автоматически разделить Гены и Группы. Убедитесь в правильности структуры названий. Ошибка: {e}")
                
                # --- ВИЗУАЛИЗАЦИЯ ---
                final_table = st.session_state.get("final_exp_table")
                if final_table is not None:
                    st.markdown("### Шаг 3. Визуализация")
                    
                    try:
                        pivot = final_table.pivot(index="Gene", columns="Group", values="Log2_FC")
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns,
                            y=pivot.index,
                            colorscale='RdBu_r',
                            zmid=0
                        ))
                        fig_heat.update_layout(title="Heatmap: Log2 Fold Change", template="plotly_white")
                        st.plotly_chart(fig_heat, use_container_width=True)
                    except Exception as e:
                        st.info(f"Не удалось построить Heatmap: {e}")
                    
                    st.markdown("#### Box-plots: Уровни экспрессии по генам")
                    
                    if not final_table.empty:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            selected_gene = st.selectbox(
                                "Выберите ген для Box-plot:",
                                options=final_table["Gene"].unique(),
                            )
                        
                        with col2:
                            gene_plot_data = final_table[final_table["Gene"] == selected_gene]
                            
                            fig_box = go.Figure()
                            
                            ref_mask = final_table["Gene"].isin(reference_genes)
                            if selected_gene in reference_genes:
                                ref_data = final_table[ref_mask]
                                raw_for_box = table.copy()
                                raw_for_box["gene"] = raw_for_box["sample"].apply(lambda x: str(x).split(" ")[0])
                                raw_for_box["bio_group"] = raw_for_box["sample"].apply(
                                    lambda x: " ".join(str(x).split(" ")[1:]).replace(r'\[\d+\]', '').strip()
                                )
                                gene_raw = raw_for_box[raw_for_box["gene"] == selected_gene]
                                
                                fig_box.add_trace(go.Box(
                                    x=gene_raw["bio_group"],
                                    y=gene_raw["Ct_cpD2"],
                                    marker_color='royalblue',
                                    name="Raw Ct"
                                ))
                                fig_box.update_layout(
                                    title=f"Сырые значения Ct для референсного гена {selected_gene}",
                                    yaxis_title="Ct",
                                    template="plotly_white"
                                )
                            else:
                                fig_box.add_trace(go.Bar(
                                    x=gene_plot_data["Group"],
                                    y=gene_plot_data["Fold_Change"],
                                    marker_color='royalblue',
                                    name="Fold Change"
                                ))
                                
                                for i, row in gene_plot_data.iterrows():
                                    pval = row.get("P_value", np.nan)
                                    if pd.notna(pval) and pval < 0.05:
                                        stars = "*" if pval < 0.05 else ""
                                        stars = "**" if pval < 0.01 else stars
                                        stars = "***" if pval < 0.001 else stars
                                        
                                        fig_box.add_annotation(
                                            x=row["Group"],
                                            y=row["Fold_Change"],
                                            text=stars,
                                            showarrow=False,
                                            yshift=10,
                                            font=dict(size=16, color="red")
                                        )
                                
                                fig_box.update_layout(
                                    title=f"Экспрессия гена {selected_gene} (относительно {control_group})",
                                    yaxis_title="Fold Change",
                                    template="plotly_white",
                                    yaxis=dict(rangemode='tozero')
                                )
                            st.plotly_chart(fig_box, use_container_width=True)
                    
                    st.markdown("#### Volcano Plot")
                    st.info("Отображает все гены во всех экспериментальных группах относительно контроля.")
                    
                    try:
                        volcano_data = final_table[final_table["Group"] != control_group].copy()
                        
                        if not volcano_data.empty:
                            volcano_data["minus_log10_p"] = -np.log10(volcano_data["P_value"].replace(0, np.nan))
                            
                            fig_volc = go.Figure()
                            
                            sig_mask = (volcano_data["P_value"] < 0.05) & (volcano_data["Log2_FC"].abs() > 1)
                            
                            fig_volc.add_trace(go.Scatter(
                                x=volcano_data[~sig_mask]["Log2_FC"],
                                y=volcano_data[~sig_mask]["minus_log10_p"],
                                mode='markers',
                                marker=dict(color='grey', size=8, opacity=0.5),
                                name='Not Sig',
                                text=volcano_data[~sig_mask]["Gene"] + " (" + volcano_data[~sig_mask]["Group"] + ")",
                                hoverinfo="text+x+y"
                            ))
                            
                            fig_volc.add_trace(go.Scatter(
                                x=volcano_data[sig_mask]["Log2_FC"],
                                y=volcano_data[sig_mask]["minus_log10_p"],
                                mode='markers',
                                marker=dict(color='red', size=10),
                                name='Significant (p<0.05, |log2FC|>1)',
                                text=volcano_data[sig_mask]["Gene"] + " (" + volcano_data[sig_mask]["Group"] + ")",
                                hoverinfo="text+x+y"
                            ))
                            
                            fig_volc.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="black")
                            fig_volc.add_vline(x=1, line_dash="dash", line_color="black")
                            fig_volc.add_vline(x=-1, line_dash="dash", line_color="black")
                            
                            fig_volc.update_layout(
                                title="Volcano Plot (все гены и группы)",
                                xaxis_title="Log2 Fold Change",
                                yaxis_title="-Log10 (P-value)",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_volc, use_container_width=True)
                            
                            st.markdown("#### Сводка значимых изменений")
                            summary = final_table.copy()
                            summary = summary[summary["P_value"] < 0.05]
                            if not summary.empty:
                                st.dataframe(summary[["Gene", "Group", "Log2_FC", "Fold_Change", "P_value"]].style.format({
                                    "Log2_FC": "{:.2f}",
                                    "Fold_Change": "{:.3f}",
                                    "P_value": "{:.4f}"
                                }))
                            else:
                                st.info("Значимых изменений экспрессии (p < 0.05) не выявлено.")
                                
                        else:
                            st.info("Недостаточно данных для Volcano Plot (только контрольная группа).")
                    except Exception as e:
                        st.warning(f"Ошибка при построении Volcano Plot: {e}")
                        
        else:
            st.info("Для анализа эксперимента необходимо сначала выполнить Пакетный фиттинг во вкладке 'Пакетный фиттинг'.")

# --------------------------
# Вкладка 8: Мульти-планшет
# --------------------------
with tab_multi:
    st.markdown("## 🌍 Глобальный анализ эксперимента (Мульти-планшет)")
    st.info("Загрузите сразу все файлы CSV, относящиеся к одному эксперименту. Приложение рассчитает кривые для каждого файла, объединит данные и проведет регрессионную нормализацию и статистику по всему массиву.")

    uploaded_files = st.file_uploader(
        "Загрузите файлы экспорта (CSV/txt/rex)", 
        type=["csv", "txt", "rex"], 
        accept_multiple_files=True,
        key="multi_upload"
    )

    if uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            ref_genes_str = st.text_input("Референсные гены (через запятую):", "eef1a1a", key="multi_ref")
            ref_genes_multi = [g.strip() for g in ref_genes_str.split(",") if g.strip()]
        with col2:
            control_group_multi = st.text_input("Контрольная группа (Bio Group):", "I-P-1", key="multi_control")
            
        use_regression_multi = st.checkbox("Использовать регрессионную нормализацию (Wang et al., 2015)", value=True, key="multi_reg")

        if st.button("🚀 Запустить глобальный расчет", type="primary", key="multi_run"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_raw_tables = []
            total_files = len(uploaded_files)
            
            try:
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Обработка файла {i+1} из {total_files}: {file.name} ...")
                    
                    load_res_m = load_qpcr_csv(file)
                    if isinstance(load_res_m, tuple) and len(load_res_m) == 2:
                        df_raw_m, is_q_m = load_res_m
                    else:
                        df_raw_m = load_res_m
                        is_q_m = False
                        first_col_m = str(df_raw_m.columns[0]).strip(' "')
                        if "Excel Raw Data Export" in first_col_m or "QIAGEN" in first_col_m:
                            is_q_m = True
                            
                    if is_q_m or ("Page 1" not in df_raw_m.columns and "Cycle" not in df_raw_m.columns):
                        try:
                            df_raw_m = df_raw_m.dropna(how='all', axis=1).dropna(how='all', axis=0)
                            header_row_idx = None
                            for idx, row in df_raw_m.iterrows():
                                if any(isinstance(val, str) and 'Page 1' in val for val in row.values):
                                    header_row_idx = idx
                                    break
                            if header_row_idx is not None:
                                df_raw_m.columns = df_raw_m.iloc[header_row_idx]
                                df_raw_m = df_raw_m.iloc[header_row_idx + 1:].reset_index(drop=True)
                                df_raw_m = df_raw_m.rename(columns={'Page 1': 'Cycle'})
                        except Exception:
                            pass
                    
                    ds_m = build_dataset_from_raw(df_raw_m)
                    
                    ds_base_m = baseline_subtract(ds_m, start_cycle=1.0, end_cycle=10.0, mode="lin")
                    
                    fit_res_m = batch_fit(ds_base_m, model="auto")
                    raw_table_m = fit_res_m.table.copy()
                    
                    raw_table_m["Source_File"] = file.name
                    all_raw_tables.append(raw_table_m)
                    
                    progress_bar.progress((i + 1) / total_files)
                
                status_text.text("Слияние данных и расчет статистики...")
                
                combined_raw_table = pd.concat(all_raw_tables, ignore_index=True)
                combined_raw_table = combined_raw_table.dropna(subset=["Ct_cpD2", "Efficiency_cpD2"])
                
                # ИСПРАВЛЕНИЕ 1: Явно указываем колонки для Ct и Эффективности, иначе KeyError
                grouped_table_m = group_replicates(
                    combined_raw_table,
                    sample_col="sample",
                    ct_col="Ct_cpD2",
                    eff_col="Efficiency_cpD2"
                )
                
                # ИСПРАВЛЕНИЕ 2: Разделяем имя лунки на Ген и Био-группу (как на вкладке 7)
                grouped_table_m["gene"] = grouped_table_m["group_name"].apply(
                    lambda x: str(x).split(" ")[0]
                )
                grouped_table_m["bio_group"] = grouped_table_m["group_name"].apply(
                    lambda x: " ".join(str(x).split(" ")[1:]) if len(str(x).split(" ")) > 1 else "Unknown"
                )
                
                final_stats_m = automated_experiment_analysis(
                    grouped_table=grouped_table_m,
                    raw_table=combined_raw_table,
                    reference_genes=ref_genes_multi,
                    control_group=control_group_multi,
                    use_regression_norm=use_regression_multi
                )
                
                progress_bar.empty()
                status_text.success(f"✅ Успешно проанализировано {total_files} файлов (всего {len(combined_raw_table)} лунок)!")
                
                st.markdown("### 📊 Итоговая сводка по всему эксперименту")
                st.dataframe(
                    final_stats_m.style.format({
                        "Regr_b": "{:.3f}",
                        "ΔCt_mean": "{:.2f}",
                        "ΔCt_sd": "{:.2f}",
                        "ΔΔCt": "{:.2f}",
                        "Fold_Change": "{:.3f}",
                        "Log2_FC": "{:.2f}",
                        "P_value": "{:.4f}"
                    }),
                    use_container_width=True
                )
                
                csv_multi = final_stats_m.to_csv(sep=";", index=False, decimal=",").encode('utf-8')
                st.download_button(
                    label="⬇️ Скачать сводную таблицу (CSV)",
                    data=csv_multi,
                    file_name="Global_Experiment_Results.csv",
                    mime="text/csv",
                    key="multi_dl_csv"
                )
                
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        final_stats_m.to_excel(writer, index=False, sheet_name='Global Results')
                        combined_raw_table.to_excel(writer, index=False, sheet_name='Raw Data Combined')
                    
                    st.download_button(
                        label="⬇️ Скачать сводную таблицу (Excel)",
                        data=buffer.getvalue(),
                        file_name="Global_Experiment_Results.xlsx",
                        mime="application/vnd.ms-excel",
                        key="multi_dl_excel"
                    )
                except ImportError:
                    st.warning("💡 Установите библиотеку `xlsxwriter` (добавьте в requirements.txt) для генерации Excel-файлов.")
                
            except Exception as e:
                st.error(f"Произошла ошибка при глобальном анализе: {e}")

# --------------------------
# Вкладка 9: CSV Конвертер
# --------------------------
with tab_csv:
    st.markdown("## 🛠️ Конвертер Excel в формат QIAGEN CSV")
    st.markdown("""
    Если ваши данные находятся в файле `.xlsx` (например, вы скопировали их из другой программы),
    вы можете загрузить его здесь. Приложение конвертирует выбранный лист в формат `.csv`, 
    который можно загрузить на вкладке «Загрузка данных».
    """)

    uploaded_excel = st.file_uploader("Загрузите Excel файл (.xlsx, .xls)", type=["xlsx", "xls"], key="excel_uploader")

    if uploaded_excel is not None:
        try:
            excel_file = pd.ExcelFile(uploaded_excel)
            sheet_names = excel_file.sheet_names
            
            st.markdown("### 📄 Выбор листа")
            selected_sheet = st.selectbox("Выберите лист с сырыми данными:", options=sheet_names)
            
            df_excel = pd.read_excel(uploaded_excel, sheet_name=selected_sheet)
            
            st.markdown("**Предпросмотр данных:**")
            st.dataframe(df_excel.head(10))
            
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
            st.error(f"Ошибка при обработке Excel файла: {e}")
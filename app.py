"""
app.py
Streamlit‑интерфейс к ядру Py‑qpcR
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import io
import re
from scipy import stats
import streamlit as st
import plotly.graph_objects as go

from qpcr_data import (
    load_qpcr_csv,
    build_dataset_from_raw,
    baseline_subtract,
    QPCRDataset,
    coerce_numeric_columns,
    convert_qiagen_to_normal,
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
    if "final_exp_table" not in st.session_state:
        st.session_state["final_exp_table"] = None
    if "bio_df" not in st.session_state:
        st.session_state["bio_df"] = None

init_state()

def robust_load_csv(file_obj):
    """Умная загрузка CSV с автоопределением разделителя"""
    file_obj.seek(0)
    content = file_obj.read().decode('utf-8', errors='ignore')
    lines = [line for line in content.split('\n') if line.strip()]
    if not lines:
        raise ValueError("Пустой файл")
        
    is_qiagen = any('QIAGEN' in line or line.startswith('"ID"') for line in lines[:30])
    if is_qiagen:
        file_obj.seek(0)
        try:
            df = convert_qiagen_to_normal(file_obj)
            return df, True
        except Exception:
            pass 
            
    header_line = lines[0]
    for line in lines[:15]:
        lower_line = line.lower()
        if 'page 1' in lower_line or 'cycle' in lower_line or 'cyc' in lower_line or 'sample' in lower_line:
            header_line = line
            break
            
    best_sep = ','
    max_cols = 0
    for s in [';', '\t', ',']:
        cols = len(header_line.split(s))
        if cols > max_cols and cols > 1:
            max_cols = cols
            best_sep = s
            
    file_obj.seek(0)
    df = pd.read_csv(file_obj, sep=best_sep, encoding='utf-8')
    return df, False

def clean_dataframe_headers(df_raw):
    """Очистка заголовков и принудительная конвертация циклов в числа"""
    df_clean = df_raw.copy()
    
    header_found = False
    for col in df_clean.columns:
        if str(col).strip().lower() in ['cycle', 'cyc', 'cycles', 'page 1']:
            df_clean = df_clean.rename(columns={col: 'Cycle'})
            header_found = True
            break
            
    if not header_found:
        try:
            df_tmp = df_clean.dropna(how='all', axis=1).dropna(how='all', axis=0)
            header_row_idx = None
            for idx, row in df_tmp.iterrows():
                row_vals = [str(val).strip().lower() for val in row.values]
                if any(c in ['page 1', 'cycle', 'cyc', 'cycles'] for c in row_vals):
                    header_row_idx = idx
                    break
            
            if header_row_idx is not None:
                df_tmp.columns = df_tmp.iloc[header_row_idx]
                df_clean = df_tmp.iloc[header_row_idx + 1:].reset_index(drop=True)
                
                for col in df_clean.columns:
                    if str(col).strip().lower() in ['page 1', 'cycle', 'cyc', 'cycles']:
                        df_clean = df_clean.rename(columns={col: 'Cycle'})
                        break
        except Exception:
            pass
            
    if "Cycle" not in df_clean.columns and len(df_clean.columns) > 0:
        df_clean = df_clean.rename(columns={df_clean.columns[0]: 'Cycle'})
        
    if "Cycle" in df_clean.columns:
        df_clean['Cycle'] = df_clean['Cycle'].astype(str).str.replace(',', '.', regex=False)
        df_clean['Cycle'] = pd.to_numeric(df_clean['Cycle'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Cycle']).reset_index(drop=True)
        
    return df_clean

def plot_curves(ds: QPCRDataset, title="Кривые амплификации", log_y=False):
    """Отрисовка кривых амплификации"""
    fig = go.Figure()
    x = pd.to_numeric(ds.df[ds.cycle_col], errors="coerce").values
    for col in ds.sample_cols:
        y = pd.to_numeric(ds.df[col], errors="coerce").values
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=col))
    
    fig.update_layout(
        title=title, xaxis_title="Cycle", yaxis_title="Fluorescence",
        hovermode="x unified", template="plotly_white",
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig

# =====================================================================
# НОВЫЙ ДВИЖОК ГЛОБАЛЬНОГО АНАЛИЗА (С ПОДДЕРЖКОЙ МНОЖЕСТВА КОНТРОЛЕЙ)
# =====================================================================
def run_advanced_analysis(raw_table, reference_genes, control_groups_str, use_regression=True, group_pattern=None):
    """
    Математически строгий глобальный анализ эксперимента.
    """
    df = raw_table.dropna(subset=["Ct_cpD2", "Efficiency_cpD2"]).copy()
    
    # 1. Очистка имен (убираем технические хвосты: [1], _1 и т.д.)
    if group_pattern and group_pattern.strip():
        def clean_name(x):
            m = re.search(group_pattern, str(x))
            return m.group(1).strip() if m else str(x).strip()
        df["group_name"] = df["sample"].apply(clean_name)
    else:
        df["group_name"] = df["sample"].apply(lambda x: re.sub(r'(\s*\[\d+\]|\s*_\d+|\s*\.\d+|\s*-\d+)\s*$', '', str(x)).strip())
        
    # 2. Усреднение технических реплик
    bio_df = df.groupby("group_name").agg({
        "Ct_cpD2": "mean",
        "Efficiency_cpD2": "mean"
    }).reset_index()
    bio_df.rename(columns={"Ct_cpD2": "ct_mean", "Efficiency_cpD2": "eff_mean"}, inplace=True)
    
    # 3. Разделение на Ген, Имя образца и Экспериментальную Группу
    bio_df["Gene"] = bio_df["group_name"].apply(lambda x: str(x).split(" ")[0])
    bio_df["Bio_Sample"] = bio_df["group_name"].apply(lambda x: " ".join(str(x).split(" ")[1:]))
    bio_df["Treatment_Group"] = bio_df["Bio_Sample"].apply(lambda x: re.sub(r'[\s\-_]*\d+$', '', x).strip())
    bio_df["Treatment_Group"] = np.where(bio_df["Treatment_Group"] == "", bio_df["Bio_Sample"], bio_df["Treatment_Group"])
    
    # 4. Расчет индекса референсных генов
    ref_data = bio_df[bio_df["Gene"].isin(reference_genes)]
    if ref_data.empty:
        raise ValueError("Референсные гены не найдены в данных! Проверьте названия.")
    
    ref_index = ref_data.groupby("Bio_Sample")["ct_mean"].mean().reset_index()
    ref_index.rename(columns={"ct_mean": "Ref_Ct"}, inplace=True)
    
    bio_df = bio_df.merge(ref_index, on="Bio_Sample", how="left")
    bio_df = bio_df.dropna(subset=["Ref_Ct"])
    
    target_genes = [g for g in bio_df["Gene"].unique() if g not in reference_genes]
    
    # Парсим список контрольных групп
    ctrl_list = [c.strip() for c in control_groups_str.split(",") if c.strip()]
    if not ctrl_list:
        ctrl_list = list(bio_df["Treatment_Group"].unique())
        
    results = []
    
    for gene in target_genes:
        g_data = bio_df[bio_df["Gene"] == gene].copy()
        if g_data.empty: continue
        
        # 5. Глобальная регрессия
        if use_regression and len(g_data) >= 3:
            slope, _, _, _, _ = stats.linregress(g_data["Ref_Ct"], g_data["ct_mean"])
            b = slope
        else:
            b = 1.0
            
        g_data["Delta_Ct"] = g_data["ct_mean"] - b * g_data["Ref_Ct"]
        
        group_stats = g_data.groupby("Treatment_Group")["Delta_Ct"].agg(['mean', 'std', 'count']).reset_index()
        dct_lists = g_data.groupby("Treatment_Group")["Delta_Ct"].apply(list).to_dict()
        
        for _, row in group_stats.iterrows():
            t_group = row["Treatment_Group"]
            mean_dct = row["mean"]
            sd_dct = row["std"] if pd.notna(row["std"]) else 0.0
            
            # 6. Умный поиск нужного контроля
            matched_ctrl = ctrl_list[0]
            if t_group in ctrl_list:
                matched_ctrl = t_group
            else:
                max_match = -1
                for c in ctrl_list:
                    match = 0
                    for i in range(min(len(t_group), len(c))):
                        if t_group[i] == c[i]: match += 1
                        else: break
                    if match > max_match:
                        max_match = match
                        matched_ctrl = c
                        
            if matched_ctrl in group_stats["Treatment_Group"].values:
                ctrl_mean = group_stats.loc[group_stats["Treatment_Group"] == matched_ctrl, "mean"].values[0]
                ctrl_items = dct_lists.get(matched_ctrl, [])
            else:
                ctrl_mean = mean_dct
                ctrl_items = dct_lists.get(t_group, [])
                
            ddct = mean_dct - ctrl_mean
            
            eff_mean = g_data["eff_mean"].mean()
            if pd.isna(eff_mean) or eff_mean <= 0: eff_mean = 2.0
            
            fc = eff_mean ** (-ddct)
            log2_fc = -ddct * np.log2(eff_mean)
            
            # 7. Статистика (t-test)
            trt_items = dct_lists.get(t_group, [])
            if len(trt_items) >= 2 and len(ctrl_items) >= 2 and t_group != matched_ctrl:
                try:
                    _, p_val = stats.ttest_ind(trt_items, ctrl_items, equal_var=False)
                except:
                    p_val = np.nan
            else:
                p_val = np.nan
                
            results.append({
                "Gene": gene,
                "Group": t_group,
                "Control_Used": matched_ctrl,
                "Regr_b": b,
                "ΔCt_mean": mean_dct,
                "ΔCt_sd": sd_dct,
                "ΔΔCt": ddct,
                "Fold_Change": fc,
                "Log2_FC": log2_fc,
                "P_value": p_val,
                "N_BioReps": int(row["count"])
            })
            
    # Референсные гены (для полноты таблицы)
    for ref in reference_genes:
        ref_data_g = bio_df[bio_df["Gene"] == ref]
        if ref_data_g.empty: continue
        group_stats = ref_data_g.groupby("Treatment_Group")["ct_mean"].agg(['count']).reset_index()
        for _, r_row in group_stats.iterrows():
            results.append({
                "Gene": ref,
                "Group": r_row["Treatment_Group"],
                "Control_Used": "-",
                "Regr_b": 1.0,
                "ΔCt_mean": 0.0, "ΔCt_sd": 0.0, "ΔΔCt": 0.0,
                "Fold_Change": 1.0, "Log2_FC": 0.0, "P_value": np.nan,
                "N_BioReps": int(r_row["count"])
            })
            
    return pd.DataFrame(results), bio_df

# ======================
# БОКОВАЯ ПАНЕЛЬ
# ======================
with st.sidebar:
    st.header("1. Загрузка данных (Сырой экспорт)")
    st.info("Поддерживаются CSV из Rotor-Gene Q и других приборов.")
    
    uploaded_file = st.file_uploader("Загрузите файл экспорта (CSV/txt/rex)", type=["csv", "txt", "rex"])
    
    if uploaded_file is not None:
        try:
            df_raw, is_qiagen = robust_load_csv(uploaded_file)
            st.session_state["raw_df"] = clean_dataframe_headers(df_raw)
            st.session_state["is_qiagen"] = False
            
            if is_qiagen:
                st.success("Файл успешно загружен и автоматически конвертирован из формата QIAGEN!")
            else:
                st.success("Файл успешно загружен!")
            
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
    "Анализ (1 планшет)", "Глобальный Анализ", "CSV Конвертер"
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
# Вкладка 7: Анализ (1 планшет)
# --------------------------
with tab_experiment:
    st.subheader("Продвинутый анализ эксперимента (Один файл)")
    st.info("Внимание: алгоритм автоматически сопоставляет ваши опытные группы с их собственными контролями. Если в файле несколько контролей, перечислите их все через запятую.")

    batch_res = st.session_state.get("batch_result")
    
    if batch_res is not None:
        table = batch_res.table.dropna(subset=["Ct_cpD2", "Efficiency_cpD2"])
        
        col1, col2 = st.columns(2)
        with col1:
            ref_genes_str = st.text_input("Референсные гены (через запятую):", "actb1", key="s_ref")
            ref_genes = [g.strip() for g in ref_genes_str.split(",") if g.strip()]
        with col2:
            control_group_str = st.text_input("Контрольная группа (или несколько через запятую):", "CTRL", key="s_ctrl")
            
        use_regression = st.checkbox("Использовать регрессионную нормализацию (Wang et al., 2015)", value=True, key="s_reg")
        
        if st.button("🚀 Рассчитать экспрессию", key="s_calc", type="primary"):
            try:
                final_table, bio_df = run_advanced_analysis(
                    raw_table=table,
                    reference_genes=ref_genes,
                    control_groups_str=control_group_str,
                    use_regression=use_regression
                )
                
                st.session_state["final_exp_table"] = final_table
                st.session_state["bio_df"] = bio_df
                st.success("Расчёт завершён успешно!")
                
                st.markdown("#### Итоговая таблица результатов")
                st.dataframe(
                    final_table.style.format({
                        "Regr_b": "{:.3f}", "ΔCt_mean": "{:.2f}", "ΔCt_sd": "{:.2f}",
                        "ΔΔCt": "{:.2f}", "Fold_Change": "{:.3f}", "Log2_FC": "{:.2f}", "P_value": "{:.4f}"
                    }).applymap(
                        lambda v: 'background-color: lightgreen' if isinstance(v, float) and v < 0.05 else '', 
                        subset=['P_value']
                    ),
                    use_container_width=True
                )
                
                csv = final_table.to_csv(sep=";", index=False, decimal=",").encode("utf-8")
                st.download_button("⬇️ Скачать сводку (CSV)", data=csv, file_name="experiment_results.csv", mime="text/csv")
                
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        final_table.to_excel(writer, index=False, sheet_name='Results')
                        bio_df.to_excel(writer, index=False, sheet_name='Biological Data')
                    
                    st.download_button(
                        label="⬇️ Скачать полный отчет (Excel)",
                        data=buffer.getvalue(),
                        file_name="experiment_report.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                except ImportError:
                    pass
                    
            except Exception as e:
                st.error(f"Ошибка при расчёте: {e}")
                
        # --- БЛОК ПОЛНОЙ ВИЗУАЛИЗАЦИИ ИЗ СТАРОЙ ВЕРСИИ ---
        final_table = st.session_state.get("final_exp_table")
        bio_df = st.session_state.get("bio_df")
        
        if final_table is not None and not final_table.empty:
            st.markdown("### Шаг 3. Визуализация")
            
            # 1. Heatmap
            try:
                pivot = final_table.pivot(index="Gene", columns="Group", values="Log2_FC")
                fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='RdBu_r', zmid=0))
                fig_heat.update_layout(title="Heatmap: Log2 Fold Change", template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
            except Exception as e:
                st.info(f"Не удалось построить Heatmap: {e}")
            
            st.markdown("#### Box-plots: Уровни экспрессии по генам")
            
            # 2. Box-plots
            col_b1, col_b2 = st.columns([1, 3])
            with col_b1:
                selected_gene = st.selectbox("Выберите ген для графика:", options=final_table["Gene"].unique())
            with col_b2:
                fig_box = go.Figure()
                if selected_gene in ref_genes:
                    gene_raw = bio_df[bio_df["Gene"] == selected_gene]
                    fig_box.add_trace(go.Box(x=gene_raw["Treatment_Group"], y=gene_raw["ct_mean"], marker_color='royalblue', name="Mean Ct"))
                    fig_box.update_layout(title=f"Сырые значения Ct для референсного гена {selected_gene}", yaxis_title="Ct", template="plotly_white")
                else:
                    gene_plot_data = final_table[final_table["Gene"] == selected_gene]
                    fig_box.add_trace(go.Bar(x=gene_plot_data["Group"], y=gene_plot_data["Fold_Change"], marker_color='royalblue', name="Fold Change"))
                    for i, row in gene_plot_data.iterrows():
                        pval = row.get("P_value", np.nan)
                        if pd.notna(pval) and pval < 0.05:
                            stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
                            fig_box.add_annotation(x=row["Group"], y=row["Fold_Change"], text=stars, showarrow=False, yshift=10, font=dict(size=16, color="red"))
                    fig_box.update_layout(title=f"Экспрессия гена {selected_gene}", yaxis_title="Fold Change", template="plotly_white", yaxis=dict(rangemode='tozero'))
                st.plotly_chart(fig_box, use_container_width=True)

            # 3. Восстановленный Volcano Plot
            st.markdown("#### Volcano Plot")
            st.info("Отображает все гены во всех экспериментальных группах (значимые изменения P<0.05 выделены цветом).")
            
            try:
                # Фильтруем данные, оставляя только те, где есть p-value (исключая референсы и контроли, для которых p=NaN)
                volcano_data = final_table.dropna(subset=["P_value"]).copy()
                
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
                        title="Volcano Plot (все опытные гены и группы)",
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
                    st.info("Недостаточно данных для Volcano Plot (только контрольные группы).")
            except Exception as e:
                st.warning(f"Ошибка при построении Volcano Plot: {e}")

# --------------------------
# Вкладка 8: Мульти-планшет
# --------------------------
with tab_multi:
    st.markdown("## 🌍 Глобальный анализ (Мульти-планшет + Мульти-контроль)")
    st.info("Загрузите все файлы экспериментов (3 точки). Программа соберет их вместе, построит **единую регрессионную кривую**, и **автоматически сопоставит опытные группы с их собственными контролями** по префиксу названия.")

    uploaded_files = st.file_uploader(
        "Загрузите файлы экспорта (CSV/txt/rex)", 
        type=["csv", "txt", "rex"], 
        accept_multiple_files=True,
        key="m_upload"
    )

    if uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            ref_genes_str = st.text_input("Референсные гены (через запятую):", "actb1", key="m_ref")
            ref_genes_multi = [g.strip() for g in ref_genes_str.split(",") if g.strip()]
        with col2:
            # СЮДА МОЖНО ВВЕСТИ НЕСКОЛЬКО КОНТРОЛЕЙ!
            control_group_multi = st.text_input(
                "Контрольные группы всех опытов (через запятую):", 
                "0311CTRL, 0325CTRL", 
                key="m_ctrl",
                help="Программа сама поймет, что группу 0311PREB надо сравнивать с 0311CTRL, а 0325PREB - с 0325CTRL."
            )
            
        use_regression_multi = st.checkbox("Глобальная регрессионная нормализация по всему массиву данных (Wang et al.)", value=True, key="m_reg")

        if st.button("🚀 Запустить глобальный расчет", type="primary", key="m_run"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_raw_tables = []
            total_files = len(uploaded_files)
            
            try:
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Обработка файла {i+1} из {total_files}: {file.name} ...")
                    df_raw_m, _ = robust_load_csv(file)
                    df_raw_m = clean_dataframe_headers(df_raw_m)
                    
                    try:
                        ds_m = build_dataset_from_raw(df_raw_m)
                        ds_base_m = baseline_subtract(ds_m, start_cycle=1.0, end_cycle=10.0, mode="lin")
                    except Exception as parse_e:
                        raise ValueError(f"Ошибка в файле '{file.name}'. Подробности: {parse_e}")
                        
                    fit_res_m = batch_fit(ds_base_m, model="auto")
                    all_raw_tables.append(fit_res_m.table.copy())
                    progress_bar.progress((i + 1) / total_files)
                
                status_text.text("Слияние данных и глобальный расчет статистики...")
                combined_raw_table = pd.concat(all_raw_tables, ignore_index=True)
                
                # Запуск нового продвинутого анализа
                final_stats_m, bio_df_m = run_advanced_analysis(
                    raw_table=combined_raw_table,
                    reference_genes=ref_genes_multi,
                    control_groups_str=control_group_multi,
                    use_regression=use_regression_multi
                )
                
                # Сохраняем в память для графиков
                st.session_state["multi_final_table"] = final_stats_m
                st.session_state["multi_bio_df"] = bio_df_m
                
                progress_bar.empty()
                status_text.success(f"✅ Успешно проанализировано {total_files} файлов. Найдено уникальных биологических групп: {len(final_stats_m['Group'].unique())}")
                
                st.markdown("### 📊 Итоговая сводка по всем экспериментам")
                # Таблица теперь покажет: Группу, Какой контроль был применен, N повторностей, и всю статистику
                st.dataframe(
                    final_stats_m.style.format({
                        "Regr_b": "{:.3f}", "ΔCt_mean": "{:.2f}", "ΔCt_sd": "{:.2f}",
                        "ΔΔCt": "{:.2f}", "Fold_Change": "{:.3f}", "Log2_FC": "{:.2f}", "P_value": "{:.4f}"
                    }).applymap(
                        lambda v: 'background-color: lightgreen' if isinstance(v, float) and v < 0.05 else '', 
                        subset=['P_value']
                    ),
                    use_container_width=True
                )
                
                csv_multi = final_stats_m.to_csv(sep=";", index=False, decimal=",").encode('utf-8')
                st.download_button(
                    label="⬇️ Скачать сводную таблицу (CSV)", data=csv_multi,
                    file_name="Global_Experiment_Results.csv", mime="text/csv", key="m_dl_csv"
                )
                
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        final_stats_m.to_excel(writer, index=False, sheet_name='Global Results')
                        bio_df_m.to_excel(writer, index=False, sheet_name='Biological Samples')
                        combined_raw_table.to_excel(writer, index=False, sheet_name='Raw Data Combined')
                    st.download_button(
                        label="⬇️ Скачать подробный отчет (Excel)", data=buffer.getvalue(),
                        file_name="Global_Experiment_Results.xlsx", mime="application/vnd.ms-excel", key="m_dl_excel"
                    )
                except ImportError:
                    pass
                
            except Exception as e:
                st.error(f"Произошла ошибка при глобальном анализе: {e}")

        # --- Визуализация Глобального Анализа ---
        m_final = st.session_state.get("multi_final_table")
        m_bio = st.session_state.get("multi_bio_df")
        
        if m_final is not None and not m_final.empty:
            show_advanced_visualizations(m_final, m_bio, ref_genes_multi, use_regression_multi, widget_key="tab8")

# --------------------------
# Вкладка 9: CSV Конвертер
# --------------------------
with tab_csv:
    st.markdown("## 🛠️ Конвертер Excel в формат QIAGEN CSV")
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
                    "Разделитель CSV:", options=[",", ";", "\t", "|"],
                    format_func=lambda x: {",": "Запятая (,)", ";": "Точка с запятой (;)", "\t": "Табуляция (\\t)", "|": "Вертикальная черта (|)"}[x],
                    key="csv_sep"
                )
            with col2:
                include_index = st.checkbox("Включить индекс строк", value=False, key="csv_index")
            csv_data = df_excel.to_csv(sep=csv_separator, index=include_index, encoding="utf-8").encode("utf-8")
            original_name = uploaded_excel.name.rsplit(".", 1)[0]
            csv_filename = f"{original_name}_{selected_sheet}.csv"
            st.download_button("⬇️ Скачать CSV", data=csv_data, file_name=csv_filename, mime="text/csv", key="download_csv")
            st.success(f"✅ Готово! Нажмите кнопку выше для скачивания `{csv_filename}`")
        except Exception as e:
            st.error(f"Ошибка при обработке Excel файла: {e}")
import json
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import nolds
import importlib.util  # 使用官方的模块导入库，更稳妥

# ==============================================================================
# 函数定义部分 (Functions Definition)
# ==============================================================================

def load_target_kpis_from_py(py_file_path):
    """
    [已重写] 使用 importlib 从指定的 .py 文件中安全地加载 TARGET_KPIS 列表。
    这是最健壮的方法。
    """
    print(f"正在从 {py_file_path} 加载KPI配置...")
    try:
        # 创建一个模块规范
        spec = importlib.util.spec_from_file_location("config_module", py_file_path)
        # 根据规范创建一个模块
        config_module = importlib.util.module_from_spec(spec)
        # 执行模块代码
        spec.loader.exec_module(config_module)
        
        # 从加载的模块中获取 TARGET_KPIS 变量
        kpi_list = getattr(config_module, 'TARGET_KPIS', None)
        
        if kpi_list is not None:
            print(f"KPI配置加载成功. 找到 {len(kpi_list)} 个KPI。")
            return kpi_list
        else:
            print(f"[错误] 在文件 {py_file_path} 中未找到 'TARGET_KPIS' 变量定义。")
            return []
            
    except Exception as e:
        print(f"[错误] 使用importlib解析KPI配置文件时出错: {e}")
        return []

def load_target_cells(json_path):
    """从指定的JSON文件中加载目标小区ID。"""
    print(f"正在从 {json_path} 加载待测小区列表...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cell_ids = [item['cell_id'] for item in data]
        print("待测小区列表加载成功。")
        return cell_ids
    except Exception as e:
        print(f"[错误] 解析小区列表文件时出错: {e}")
        return []

def load_kpi_data_for_cells(base_dir, cell_ids, kpi_names):
    """为指定的小区加载KPI数据（宽格式）。"""
    all_data = {}
    print(f"\n开始从 '{base_dir}' 文件夹加载数据 (宽格式)...")
    
    for cell_id in cell_ids:
        kpi_file_path = os.path.join(base_dir, cell_id, 'KPI.csv')
        if os.path.exists(kpi_file_path):
            try:
                df = pd.read_csv(kpi_file_path)
                if 'date_time' not in df.columns:
                    print(f"  [失败] 小区 {cell_id} 的 CSV 文件缺少 'date_time' 列。")
                    continue
                
                existing_kpis = [col for col in kpi_names if col in df.columns]
                if not existing_kpis:
                    print(f"  [警告] 小区 {cell_id} 中未找到任何待测KPI列。")
                    continue
                    
                df_filtered = df[['date_time'] + existing_kpis]
                df_filtered['date_time'] = pd.to_datetime(df_filtered['date_time'])
                df_filtered.set_index('date_time', inplace=True)
                
                all_data[cell_id] = df_filtered
                print(f"  [成功] 已加载并处理小区: {cell_id}")
                
            except Exception as e:
                print(f"  [失败] 处理小区 {cell_id} 时出现意外错误: {e}")
        else:
            print(f"  [警告] 找不到文件: {kpi_file_path}")
    return all_data

def analyze_kpi_predictability(all_cell_data):
    """对每个KPI时间序列进行可预测性分析。"""
    analysis_report = []
    if not all_cell_data:
        print("没有数据可供分析。")
        return pd.DataFrame()

    print(f"\n--- 开始对 {len(all_cell_data)} 个小区的KPI进行可预测性分析 ---")
    for cell_id, df_kpis in all_cell_data.items():
        print(f"正在分析小区: {cell_id}")
        for kpi in df_kpis.columns:
            ts = df_kpis[kpi].fillna(method='ffill').dropna()
            if len(ts) < 50: continue

            p_value, samp_ent = np.nan, np.nan
            try: p_value = adfuller(ts)[1]
            except Exception: pass
            try: samp_ent = nolds.sampen(ts, emb_dim=2, tolerance=0.2 * ts.std())
            except Exception: pass
            
            analysis_report.append({
                "Cell_ID": cell_id, "KPI": kpi, "Data_Points": len(ts),
                "ADF_p_value": p_value, "Sample_Entropy": samp_ent,
            })
    
    if not analysis_report:
        print("分析完成，但没有生成任何结果。")
        return pd.DataFrame()

    df_results = pd.DataFrame(analysis_report)
    conditions = [
        (df_results['ADF_p_value'] <= 0.05) & (df_results['Sample_Entropy'] <= 0.3),
        (df_results['ADF_p_value'] > 0.05) & (df_results['Sample_Entropy'] > 0.8)
    ]
    df_results['预测难度'] = np.select(conditions, ['低', '高'], default='中')
    return df_results

# ==============================================================================
# 主程序入口 (Main Execution Block)
# ==============================================================================
if __name__ == '__main__':
    base_dir = '/root'
    train_data_dir = os.path.join(base_dir, '2025CCF国际AIOps挑战赛-赛道二/train')
    test_cells_json_path = os.path.join(base_dir, '2025CCF国际AIOps挑战赛-赛道二/changes-test_1.json')
    kpi_py_file_path = os.path.join(base_dir, 'aiops-challenge-2025/aiops_challenge_2025/experiment/__init__.py')

    target_kpis = load_target_kpis_from_py(kpi_py_file_path)
    target_cell_ids = load_target_cells(test_cells_json_path)
    
    print(f"\n分析目标已确定：\n- {len(target_cell_ids)} 个待测小区\n- {len(target_kpis)} 个待测KPI")

    if not target_kpis or not target_cell_ids:
        print("\n错误：未能加载KPI列表或小区列表。分析中止。")
    else:
        all_cell_data = load_kpi_data_for_cells(train_data_dir, target_cell_ids, target_kpis)
        if not all_cell_data:
            print("\n未能成功加载任何小区的KPI数据，分析中止。")
        else:
            predictability_report = analyze_kpi_predictability(all_cell_data)
            if not predictability_report.empty:
                print("\n--- 可预测性分析报告 ---")
                pd.set_option('display.max_rows', 100)
                pd.set_option('display.width', 150)
                agg_report = predictability_report.groupby('KPI').agg(
                    Avg_ADF_p_value=('ADF_p_value', 'mean'),
                    Avg_Sample_Entropy=('Sample_Entropy', 'mean'),
                    Num_Cells_Analyzed=('Cell_ID', 'count')
                ).reset_index()
                conditions = [
                    (agg_report['Avg_ADF_p_value'] <= 0.05) & (agg_report['Avg_Sample_Entropy'] <= 0.3),
                    (agg_report['Avg_ADF_p_value'] > 0.05) & (agg_report['Avg_Sample_Entropy'] > 0.8)
                ]
                agg_report['总体预测难度'] = np.select(conditions, ['低', '高'], default='中')
                print("\n--- 各KPI总体预测难度评估 (基于所有已分析小区的平均值) ---")
                print(agg_report.sort_values(by=['总体预测难度', 'Avg_Sample_Entropy']))

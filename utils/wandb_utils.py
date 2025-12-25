import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from scipy.stats import bootstrap

def get_filtered_data(entity, project, config, x_key='_step', y_key='eval/success'):
    api = wandb.Api()
    
    # WandB API Filter 생성 (MongoDB Query Style)
    # config['filters']에 있는 내용을 API 쿼리로 변환
    query_filters = {}
    for k, v in config['filters'].items():
        # config.으로 시작하지 않는 일반 키(group, state 등) 처리
        if "." in k and not k.startswith("config.") and k != "summary_metrics":
            if isinstance(v, list):
                # [핵심] 사용자가 ["A", "B"]로 넣으면 -> {"$in": ["A", "B"]}로 변환하여 API에 전달
                query_filters[k] = {"$in": v}
            else:
                query_filters[k] = v
        query_filters[k] = v

    print(f"Fetching runs with filters: {query_filters}...")
    runs = api.runs(f"{entity}/{project}", filters=query_filters)
    
    all_dfs = []

    group_key = config['group_by']
    target_values = config.get('select_values')

    print("Found", len(runs), "runs.")

    for run in tqdm(runs):
        group_val = run.config
        if group_key is not None:
            for key_part in group_key.replace("config.", "").split("."):
                group_val = group_val.get(key_part, "Unknown")
            
            if target_values is not None and group_val not in target_values:
                continue

        hist = run.history(keys=[x_key, y_key])
        hist["group_id"] = f"{config['prefix']}_{group_val}"
        all_dfs.append(hist)

    if not all_dfs:
        print("조건에 맞는 Run을 찾지 못했습니다!")
        return pd.DataFrame()

    return pd.concat(all_dfs).sort_values(by=x_key)

def calculate_scipy_ci(data, ci=95):
    """
    Scipy를 사용하여 95% 신뢰구간 계산
    """
    # 1. 데이터 정리 (NaN 제거 및 배열 변환)
    data = np.array(data)
    data = data[~np.isnan(data)]

    if len(data) < 2 or np.all(data == data[0]):
        mean_val = np.mean(data)
        return mean_val, mean_val

    res = bootstrap(
        (data,), 
        statistic=np.mean, 
        confidence_level=ci/100, 
        n_resamples=1000, 
        method='percentile' # RL에서는 0점/100점이 많아 'percentile'이 안전합니다.
    )
    
    return res.confidence_interval.low, res.confidence_interval.high

def plot_custom_config(df, filename='plot'):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # 사용자가 지정한 순서대로 그리기 위해 정렬
    # if config['select_values']:
    #     groups = config['select_values']
    # else:
    x_key = df.keys()[0]
    y_key = df.keys()[1]
    groups = sorted(df['group_id'].unique())

    colors = sns.color_palette("husl", len(groups))

    for i, group_name in enumerate(groups):
        # 해당 그룹 데이터 추출
        group_df = df[df["group_id"] == group_name]
        if group_df.empty: continue

        # 통계 계산 (Mean & Std)
        agg_df = group_df.groupby(x_key)[y_key].apply(list).reset_index(name='values')
        
        x_list = []
        means = []
        lowers = []
        uppers = []

        for _, row in agg_df.iterrows():
            vals = row['values']
            mean_val = np.mean(vals)
            
            # [변경점] 여기서 Scipy 함수 호출!
            lower, upper = calculate_scipy_ci(vals, ci=95)
            
            x_list.append(row[x_key])
            means.append(mean_val)
            lowers.append(lower)
            uppers.append(upper)
        
        # Numpy 배열 변환
        x, mean, lower, upper = map(np.array, [x_list, means, lowers, uppers])
        # Label 매핑 적용
        label = group_name

        # Plot
        plt.plot(x, mean, label=label, color=colors[i], linewidth=2)
        plt.fill_between(x, lower, upper, color=colors[i], alpha=0.15)
    
    title = filename

    plt.title(f"{title}", fontsize=16)
    plt.xlabel(x_key, fontsize=12)
    plt.ylabel(y_key, fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()

## MAIN FUNCTION
def plot_and_save_from_wandb(PLOT_CONFIG):
    x_key = PLOT_CONFIG['x_key']
    y_key = PLOT_CONFIG['y_key']

    dfs = []

    for config in PLOT_CONFIG['configs']:
        df = get_filtered_data(PLOT_CONFIG['WANDB_ENTITY'], PLOT_CONFIG['WANDB_PROJECT'], config, x_key, y_key)
        dfs.append(df)

    if len(dfs) > 0:
        dfs = pd.concat(dfs)
    # return dfs
    plot_custom_config(dfs, PLOT_CONFIG['title'])
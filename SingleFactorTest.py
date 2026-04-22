"""
单因子批量回测主程序。

文件职责：
1. 读取 `data_factor.csv`
2. 每次只按一个选定的调仓频率运行，频率在 `REBALANCE_FREQUENCY` 处手动切换
3. 对每个可用因子执行样本筛选、缺失值处理、MAD 去极值和分组标准化
4. 输出单因子分组回测结果与因子表现汇总
5. 汇总结果采用“总表 + 分年度明细表”结构，分年度表会输出 `total / 2018 / 2019 / 2020 / 2021 / 2022 / 2023`
6. 汇总文件名会附带频率后缀，例如 `因子表现汇总_双周度.xlsx`，避免不同频率结果互相覆盖
"""

import os
import time
import warnings

import pandas as pd
import pymssql
from tqdm import tqdm

from Function import (
    Back_Testing,
    Build_RebalanceMap,
    Cal_Daily_Return,
    Cal_FactorPerformance,
    Cal_Position,
    Cal_Sift,
    Cal_Stratify,
    Fillna_WithIndustryMean,
    Load_Calender,
    Load_Factor,
    Load_SQLData,
    LoadRiskFreeReturn,
    Mad_Standard,
)

warnings.filterwarnings("ignore")


INPUT_PATH = "Input/"
OUTPUT_PATH = "Output/"
GROUP_NUM = 3
SINGLE_FACTOR_COST_RATE = 0.0
# 在这里切换调仓频率：可选 "周度" / "双周度" / "月度"。
# 脚本每次只跑当前选中的一个频率，不会把三种频率一起跑。
REBALANCE_FREQUENCY = "双周度"
YEARLY_ANALYSIS_YEARS = ["2018", "2019", "2020", "2021", "2022", "2023"]
SQL_HOST = "(local)"
SQL_USER = "sa" 
SQL_PASSWORD = "MyStrongPass123"
STOCK_DB = "winddb1101"
BOND_DB = "winddbbond1101"

STOCK_MOMENTUM_WINDOWS = [5, 10, 20, 60, 120, 240]
STOCK_AMIHUD_WINDOWS = [5, 10, 20, 60, 120]
STOCK_TURNOVER_WINDOWS = [5, 10, 20, 60, 120]
STOCK_MFI_WINDOWS = [5, 10, 20, 60, 120, 240]
STOCK_PERCENT_B_WINDOWS = [5, 10, 20, 60, 120, 240]
STOCK_PRICE_TO_HIGH_WINDOWS = [10, 20, 60, 120, 240]
STOCK_RSI_WINDOWS = [5, 10, 20, 60, 120, 240]
BOND_PCTCHANGE_WINDOWS = [5, 10, 20, 60, 120, 240]
BOND_TURNOVER_WINDOWS = [5, 10, 20, 60, 120, 240]
STOCK_BOND_SPREAD_WINDOWS = [5, 10, 20, 60, 120, 240]
STOCK_BOND_CORR_WINDOWS = [20, 60, 120, 240]
MINUTE_FACTOR_WINDOWS = [5, 10, 20, 60, 120]


def _ts_name(label, window):
    return f"{label}的{'6M' if window == 120 else '1Y'}时序Zscore因子值"


FACTOR_CANDIDATES = [
    "双低",
    "纯债溢价率",
    "收盘价",
    "转股溢价率",
    "转债隐含波动率",
    "隐波差",
]
FACTOR_CANDIDATES.extend([f"转债近{window}日换手率" for window in BOND_TURNOVER_WINDOWS])
FACTOR_CANDIDATES.extend([f"转债近{window}日涨跌幅（%）" for window in BOND_PCTCHANGE_WINDOWS])
FACTOR_CANDIDATES.extend([f"正股{window}日Amihud指标" for window in STOCK_AMIHUD_WINDOWS])
FACTOR_CANDIDATES.extend([f"正股日均换手率{window}" for window in STOCK_TURNOVER_WINDOWS])
FACTOR_CANDIDATES.extend([f"正股{window}日MFI指标" for window in STOCK_MFI_WINDOWS])
FACTOR_CANDIDATES.extend([f"正股近{window}日涨跌幅（%）" for window in STOCK_MOMENTUM_WINDOWS])
FACTOR_CANDIDATES.extend([f"正股{window}日PercentB指标" for window in STOCK_PERCENT_B_WINDOWS])
FACTOR_CANDIDATES.extend([f"正股{window}日PriceToHigh指标" for window in STOCK_PRICE_TO_HIGH_WINDOWS])
FACTOR_CANDIDATES.extend([f"正股{window}日RSI指标" for window in STOCK_RSI_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日转债正股涨跌幅之差" for window in STOCK_BOND_SPREAD_WINDOWS])
FACTOR_CANDIDATES.extend([f"转债和正股近{window}日涨跌幅相关系数" for window in STOCK_BOND_CORR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日日内5分钟线RSI" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日温和收益均值" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日日内5分钟线收益率方差" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日日内5分钟线收益率偏度均值" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日日内5分钟线量价相关系数均值" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日日内5分钟线量价相关系数波动率" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日日内5分钟线成交量变化方差均值" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日日内5分钟线成交量变化偏度均值" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日开盘半小时成交量占比均值" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend([f"近{window}日尾盘半小时成交量占比均值" for window in MINUTE_FACTOR_WINDOWS])
FACTOR_CANDIDATES.extend(
    [
        _ts_name(label, ts_window)
        for label in ["双低", "纯债溢价率", "收盘价", "转股溢价率", "隐波差", "隐波"]
        for ts_window in (120, 240)
    ]
)
FACTOR_CANDIDATES.extend(
    [
        _ts_name(f"正股近{window}日涨跌幅（%）", ts_window)
        for window in STOCK_MOMENTUM_WINDOWS
        for ts_window in (120, 240)
    ]
)
FACTOR_CANDIDATES.extend(
    [
        _ts_name(f"近{window}日转债正股涨跌幅之差", ts_window)
        for window in STOCK_BOND_SPREAD_WINDOWS
        for ts_window in (120, 240)
    ]
)
FACTOR_CANDIDATES.extend(
    [
        _ts_name(f"转债和正股近{window}日涨跌幅相关系数", ts_window)
        for window in STOCK_BOND_CORR_WINDOWS
        for ts_window in (120, 240)
    ]
)

PERFORM_COLUMNS = [
    "年化收益(%)",
    "基准年化收益(%)",
    "超额年化收益(%)",
    "年化波动(%)",
    "基准年化波动(%)",
    "超额年化波动(%)",
    "最大回撤(%)",
    "基准最大回撤(%)",
    "超额最大回撤(%)",
    "夏普比率",
    "基准夏普比率",
    "信息比率",
    "收益回撤比",
    "基准收益回撤比",
    "超额收益回撤比",
    "胜率(%)",
    "换手率(年均)",
]

PERFORM_PERCENT_COLUMNS = {
    "年化收益(%)",
    "基准年化收益(%)",
    "超额年化收益(%)",
    "年化波动(%)",
    "基准年化波动(%)",
    "超额年化波动(%)",
    "最大回撤(%)",
    "基准最大回撤(%)",
    "超额最大回撤(%)",
    "胜率(%)",
    "换手率(年均)",
}

FACTOR_PERCENT_COLUMNS = {
    "RankIC胜率",
    "多头超额收益率",
    "多头平均换手率（单边）",
    "多空年化收益率",
    "TOP30组合超额收益率",
    "TOP31-60组合超额收益率",
    "Bottom60-31组合超额收益率",
    "Bottom30组合超额收益率",
}


# ================================== 因子列表 ==================================
def get_available_factors(factor_file):
    columns = pd.read_csv(factor_file, nrows=1).columns.tolist()
    available = [factor for factor in FACTOR_CANDIDATES if factor in columns]
    missing = [factor for factor in FACTOR_CANDIDATES if factor not in columns]
    return available, missing


def get_selected_frequency_calendar(weekly_calendar, monthly_calendar):
    if REBALANCE_FREQUENCY == "周度":
        return REBALANCE_FREQUENCY, weekly_calendar.reset_index(drop=True)
    if REBALANCE_FREQUENCY == "双周度":
        return REBALANCE_FREQUENCY, weekly_calendar[0::2].reset_index(drop=True)
    if REBALANCE_FREQUENCY == "月度":
        return REBALANCE_FREQUENCY, monthly_calendar.reset_index(drop=True)
    raise ValueError(f'未知调仓频率: {REBALANCE_FREQUENCY}，请在 REBALANCE_FREQUENCY 中填写 "周度"、"双周度" 或 "月度"。')


def filter_by_year(data, year, date_col="TRADE_DT", use_index=False):
    if data is None or len(data) == 0:
        return data
    year = str(year)
    if use_index:
        mask = pd.Index(data.index).astype(str).str[:4] == year
        return data.loc[mask].copy()
    mask = data[date_col].astype(str).str[:4] == year
    return data.loc[mask].copy()


def build_empty_factor_performance(metric_columns):
    return pd.DataFrame([{column: pd.NA for column in metric_columns}])


def build_yearly_factor_performance(
    total_factor_performance,
    factor_mad_stand,
    sample_pool,
    price,
    risk_free_return,
    cost_rate,
    yearly_analysis_years,
):
    metric_columns = total_factor_performance.columns.tolist()
    yearly_frames = []

    total_row = total_factor_performance.copy()
    total_row.insert(0, "年份", "total")
    yearly_frames.append(total_row)

    for year in yearly_analysis_years:
        factor_year = filter_by_year(factor_mad_stand, year)
        sample_year = filter_by_year(sample_pool, year)
        price_year = filter_by_year(price, year)
        risk_free_return_year = filter_by_year(risk_free_return, year, use_index=True)

        if (
            factor_year is None
            or sample_year is None
            or price_year is None
            or len(factor_year) == 0
            or len(sample_year) == 0
            or len(price_year) == 0
            or sample_year["TRADE_DT"].nunique() < 2
        ):
            yearly_row = build_empty_factor_performance(metric_columns)
        else:
            try:
                bench_position_year = Cal_Position(
                    sample_year[["TRADE_DT", "S_INFO_WINDCODE"]].drop_duplicates().reset_index(drop=True)
                )
                bench_return_year = Cal_Daily_Return(bench_position_year, price_year, cost_rate=0)
                yearly_row, _ = Cal_FactorPerformance(
                    factor_year,
                    price_year,
                    bench_return_year,
                    risk_free_return_year,
                    cost_rate=cost_rate,
                )
                yearly_row = yearly_row.reindex(columns=metric_columns)
            except Exception:
                yearly_row = build_empty_factor_performance(metric_columns)

        yearly_row.insert(0, "年份", str(year))
        yearly_frames.append(yearly_row)

    return pd.concat(yearly_frames, ignore_index=True)


def format_percentage_columns(worksheet, percentage_headers, number_format="0.00%"):
    header_row = next(worksheet.iter_rows(min_row=1, max_row=1))
    percent_columns = [
        cell.column for cell in header_row
        if cell.value in percentage_headers
    ]
    for column_index in percent_columns:
        for column_cells in worksheet.iter_cols(
            min_col=column_index,
            max_col=column_index,
            min_row=2,
            max_row=worksheet.max_row,
        ):
            for cell in column_cells:
                if isinstance(cell.value, (int, float)):
                    cell.number_format = number_format


# ================================== 单因子回测 ==================================
def run_single_factor_backtest(
    factor_name,
    frequency_label,
    period_map,
    factor_file,
    price,
    industry,
    list_date,
    suspend,
    balance,
    rating,
    cbond_type,
    turnover,
    exit_active,
    risk_free_return,
    output_path,
    group_num,
    cost_rate,
    yearly_analysis_years,
    trading_dates=None,
):
    factor = Load_Factor(period_map, factor_file)[["TRADE_DT", "S_INFO_WINDCODE", factor_name]].copy()
    factor.rename(columns={factor_name: "Factor"}, inplace=True)

    # 先做样本池筛选，再做因子预处理。
    factor = Cal_Sift(
        factor,
        list_date,
        suspend,
        balance,
        rating,
        turnover,
        exit_active,
        trading_dates=trading_dates,
    )
    sample_pool = factor[["TRADE_DT", "S_INFO_WINDCODE"]].drop_duplicates().reset_index(drop=True)

    # 缺失值处理：按中信一级行业中位数填充。
    factor = (
        factor.dropna(subset=["TRADE_DT"])
        .groupby("TRADE_DT", group_keys=False)
        .apply(Fillna_WithIndustryMean, industry)
        .reset_index(drop=True)
    )

    # 预处理：MAD 去极值 + 按转债类型分组标准化。
    factor_mad_stand = (
        factor.groupby("TRADE_DT", group_keys=False)
        .apply(Mad_Standard, cbond_type)
        .reset_index(drop=True)
    )
    factor_group = Cal_Stratify(factor_mad_stand, group_num)

    # 基准按样本池等权指数构造，与研报口径更接近。
    bench_position = Cal_Position(sample_pool)
    bench_return = Cal_Daily_Return(bench_position, price, cost_rate=0)

    factor_performance, rank_ic_series = Cal_FactorPerformance(
        factor_mad_stand,
        price,
        bench_return,
        risk_free_return,
        cost_rate=cost_rate,
    )
    factor_performance_by_year = build_yearly_factor_performance(
        factor_performance,
        factor_mad_stand,
        sample_pool,
        price,
        risk_free_return,
        cost_rate,
        yearly_analysis_years,
    )

    perform = pd.DataFrame(
        index=[f"第{i}组" for i in range(1, group_num + 1)],
        columns=PERFORM_COLUMNS,
    )
    net_value = pd.DataFrame(index=bench_return.index)

    output_file = os.path.join(output_path, f"{factor_name}分组回测结果_{frequency_label}.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        for i in range(1, group_num + 1):
            hold = factor_group[factor_group["GROUP"] == str(i)][["TRADE_DT", "S_INFO_WINDCODE"]].reset_index(drop=True)
            position = Cal_Position(hold)
            position.to_excel(writer, sheet_name=f"第{i}组仓位", index=False)

            net_value_tmp, perform_tmp = Back_Testing(
                position,
                price,
                bench_return,
                risk_free_return,
                cost_rate=cost_rate,
            )
            perform.loc[f"第{i}组"] = perform_tmp.iloc[0].values
            net_value["样本池等权指数净值"] = net_value_tmp.iloc[:, 0]
            net_value[f"第{i}组净值"] = net_value_tmp.iloc[:, 1]
            net_value[f"第{i}组超额净值"] = net_value_tmp.iloc[:, 2]

        perform.to_excel(writer, sheet_name=f"绩效统计_{frequency_label}")
        net_value.to_excel(writer, sheet_name="净值统计")
        factor_performance.to_excel(writer, sheet_name="因子表现")
        factor_performance_by_year.to_excel(writer, sheet_name="因子表现分年度", index=False)
        rank_ic_series.to_frame(name="RankIC").to_excel(writer, sheet_name="RankIC序列")
        format_percentage_columns(writer.sheets[f"绩效统计_{frequency_label}"], PERFORM_PERCENT_COLUMNS)
        format_percentage_columns(writer.sheets["因子表现"], FACTOR_PERCENT_COLUMNS)
        format_percentage_columns(writer.sheets["因子表现分年度"], FACTOR_PERCENT_COLUMNS)

    factor_performance.index = [factor_name]
    factor_performance_by_year.insert(0, "因子", factor_name)
    factor_performance_by_year["年份"] = pd.Categorical(
        factor_performance_by_year["年份"].astype(str),
        categories=["total"] + [str(year) for year in yearly_analysis_years],
        ordered=True,
    )
    factor_performance_by_year = factor_performance_by_year.sort_values("年份").reset_index(drop=True)
    factor_performance_by_year["年份"] = factor_performance_by_year["年份"].astype(str)
    return output_file, factor_performance, factor_performance_by_year


def run_frequency_backtests(
    frequency_label,
    rebalance_calendar,
    daily_calendar,
    factor_file,
    available_factors,
    use_local_cache,
):
    conn1101 = None
    conn1101bond = None
    need_bond_conn = (not use_local_cache) or (
        use_local_cache and not os.path.exists(os.path.join(OUTPUT_PATH, "BondExitEvents.csv"))
    )
    if not use_local_cache:
        conn1101 = pymssql.connect(SQL_HOST, SQL_USER, SQL_PASSWORD, STOCK_DB)
    if need_bond_conn:
        conn1101bond = pymssql.connect(SQL_HOST, SQL_USER, SQL_PASSWORD, BOND_DB)

    try:
        period_map = Build_RebalanceMap(rebalance_calendar, daily_calendar)
        print(f"读取{frequency_label}回测基础数据中...")
        price, industry, list_date, suspend, balance, rating, cbond_type, turnover, exit_active = Load_SQLData(
            conn1101,
            conn1101bond,
            period_map,
            local=use_local_cache,
        )
        risk_free_return = LoadRiskFreeReturn(os.path.join(INPUT_PATH, "无风险利率.xlsx"), period_map)
        print(f"{frequency_label}回测基础数据读取完成")
    finally:
        if conn1101 is not None:
            conn1101.close()
        if conn1101bond is not None:
            conn1101bond.close()

    trading_dates = daily_calendar["TRADE_DT"].astype(str).tolist()
    success_factors = []
    failed_factors = []
    factor_performance_summary = []
    factor_performance_yearly_summary = []

    for factor_name in tqdm(available_factors, desc=f"{frequency_label}逐个因子回测"):
        factor_start_time = time.time()
        try:
            output_file, factor_performance, factor_performance_by_year = run_single_factor_backtest(
                factor_name,
                frequency_label,
                period_map,
                factor_file,
                price,
                industry,
                list_date,
                suspend,
                balance,
                rating,
                cbond_type,
                turnover,
                exit_active,
                risk_free_return,
                OUTPUT_PATH,
                GROUP_NUM,
                SINGLE_FACTOR_COST_RATE,
                YEARLY_ANALYSIS_YEARS,
                trading_dates=trading_dates,
            )
            success_factors.append(factor_name)
            factor_performance_summary.append(factor_performance)
            factor_performance_yearly_summary.append(factor_performance_by_year)
            print(
                f"{frequency_label} 因子 {factor_name} 回测完成，结果已输出至 {output_file}，"
                f"耗时 {time.time() - factor_start_time:.2f} 秒"
            )
        except Exception as exc:
            failed_factors.append((factor_name, str(exc)))
            print(f"{frequency_label} 因子 {factor_name} 回测失败：{exc}")

    summary_df = pd.concat(factor_performance_summary) if factor_performance_summary else pd.DataFrame()
    yearly_summary_df = (
        pd.concat(factor_performance_yearly_summary, ignore_index=True)
        if factor_performance_yearly_summary else pd.DataFrame()
    )
    if not yearly_summary_df.empty:
        yearly_summary_df["年份"] = pd.Categorical(
            yearly_summary_df["年份"].astype(str),
            categories=["total"] + YEARLY_ANALYSIS_YEARS,
            ordered=True,
        )
        yearly_summary_df = yearly_summary_df.sort_values(["因子", "年份"]).reset_index(drop=True)
        yearly_summary_df["年份"] = yearly_summary_df["年份"].astype(str)
    return summary_df, yearly_summary_df, success_factors, failed_factors


# ================================== 批量主程序 ==================================
def main():
    factor_file = os.path.join(OUTPUT_PATH, "data_factor.csv")
    available_factors, missing_factors = get_available_factors(factor_file)

    print("本次准备回测的因子数量：", len(available_factors))
    if missing_factors:
        print("以下候选因子在 data_factor.csv 中不存在：")
        for factor_name in missing_factors:
            print(" -", factor_name)

    use_local_cache = True
    start_time = time.time()
    weekly_calendar = Load_Calender(os.path.join(INPUT_PATH, "调仓日期（周频）.xlsx"))
    daily_calendar = Load_Calender(os.path.join(INPUT_PATH, "调仓日期（日频）.xlsx"))
    monthly_calendar = Load_Calender(os.path.join(INPUT_PATH, "调仓日期（月频）.xlsx"))
    frequency_label, rebalance_calendar = get_selected_frequency_calendar(weekly_calendar, monthly_calendar)
    print(f"当前调仓频率：{frequency_label}；如需切换，请修改 REBALANCE_FREQUENCY。")

    summary_df, yearly_summary_df, success_factors, failed_factors = run_frequency_backtests(
        frequency_label,
        rebalance_calendar,
        daily_calendar,
        factor_file,
        available_factors,
        use_local_cache,
    )

    # 汇总文件按频率分别保存，避免周度/双周度/月度互相覆盖。
    summary_file = os.path.join(OUTPUT_PATH, f"因子表现汇总_{frequency_label}.xlsx")
    with pd.ExcelWriter(summary_file) as writer:
        summary_df.to_excel(writer, sheet_name=f"绩效统计_{frequency_label}")
        yearly_summary_df.to_excel(writer, sheet_name="因子表现分年度", index=False)
        format_percentage_columns(writer.sheets[f"绩效统计_{frequency_label}"], FACTOR_PERCENT_COLUMNS)
        format_percentage_columns(writer.sheets["因子表现分年度"], FACTOR_PERCENT_COLUMNS)
    print(f"{frequency_label}因子表现汇总已输出至 {summary_file}")

    print(f"全部回测完成，总耗时 {time.time() - start_time:.2f} 秒")
    print(f"{frequency_label}成功运行的因子数量：", len(success_factors))
    if failed_factors:
        print(f"{frequency_label}以下因子运行失败：")
        for factor_name, error_message in failed_factors:
            print(f" - {factor_name}: {error_message}")
if __name__ == "__main__":
    main()

    

"""
可转债样本池提取与因子构建。

文件职责：
1. 从SQL/本地缓存读取转债和正股数据
2. 整理样本池基础字段
3. 构建研报复现所需的原始因子与时序因子
4. 输出 `Output/data_filled.csv` 和 `Output/data_factor.csv`
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql

warnings.filterwarnings("ignore")


INPUT_PATH = "Input/"
OUTPUT_PATH = "Output/"
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
MINUTE_INPUT_PATH = os.path.join(INPUT_PATH, "minute")
MINUTE_CACHE_FILE = os.path.join(OUTPUT_PATH, "MinuteDailyFeatures.csv")


def _ts_name(source_column, label, window):
    return f"{label}的{'6M' if window == 120 else '1Y'}时序Zscore因子值"


TIME_SERIES_ZSCORE_CONFIGS = [
    ("双低", _ts_name("双低", "双低", 120), 120),
    ("双低", _ts_name("双低", "双低", 240), 240),
    ("纯债溢价率", _ts_name("纯债溢价率", "纯债溢价率", 120), 120),
    ("纯债溢价率", _ts_name("纯债溢价率", "纯债溢价率", 240), 240),
    ("收盘价", _ts_name("收盘价", "收盘价", 120), 120),
    ("收盘价", _ts_name("收盘价", "收盘价", 240), 240),
    ("转股溢价率", _ts_name("转股溢价率", "转股溢价率", 120), 120),
    ("转股溢价率", _ts_name("转股溢价率", "转股溢价率", 240), 240),
    ("隐波差", _ts_name("隐波差", "隐波差", 120), 120),
    ("隐波差", _ts_name("隐波差", "隐波差", 240), 240),
    ("转债隐含波动率", _ts_name("转债隐含波动率", "隐波", 120), 120),
    ("转债隐含波动率", _ts_name("转债隐含波动率", "隐波", 240), 240),
]
TIME_SERIES_ZSCORE_CONFIGS.extend(
    [
        (f"正股近{window}日涨跌幅（%）", _ts_name(f"正股近{window}日涨跌幅（%）", f"正股近{window}日涨跌幅（%）", ts_window), ts_window)
        for window in STOCK_MOMENTUM_WINDOWS
        for ts_window in (120, 240)
    ]
)
TIME_SERIES_ZSCORE_CONFIGS.extend(
    [
        (
            f"近{window}日转债正股涨跌幅之差",
            _ts_name(f"近{window}日转债正股涨跌幅之差", f"近{window}日转债正股涨跌幅之差", ts_window),
            ts_window,
        )
        for window in STOCK_BOND_SPREAD_WINDOWS
        for ts_window in (120, 240)
    ]
)
TIME_SERIES_ZSCORE_CONFIGS.extend(
    [
        (
            f"转债和正股近{window}日涨跌幅相关系数",
            _ts_name(f"转债和正股近{window}日涨跌幅相关系数", f"转债和正股近{window}日涨跌幅相关系数", ts_window),
            ts_window,
        )
        for window in STOCK_BOND_CORR_WINDOWS
        for ts_window in (120, 240)
    ]
)


# ================================== 公共工具 ==================================
def _resolve_localfile(local, path):
    return path if local and os.path.exists(path) else False


def print_hi(text):
    print(f"hi,{text}")


def Load_Calender(filename):
    """读取调仓日历并统一为 YYYYMMDD 字符串。"""
    data = pd.read_excel(filename)
    data["调仓日期"] = pd.to_datetime([str(i) for i in data["调仓日期"]]).strftime("%Y%m%d")
    data.rename(columns={"调仓日期": "TRADE_DT"}, inplace=True)
    return data


# ================================== 数据读取 ==================================
def Load_Valuation(conn, StartDate, EndDate, localfile=False):
    """读取转债估值与衍生指标。"""
    if localfile:
        data = pd.read_csv(localfile)
    else:
        sql = (
            "select TRADE_DT,S_INFO_WINDCODE,"
            "CB_ANAL_STRBPREMIUMRATIO '纯债溢价率',"
            "CB_ANAL_CONVPREMIUMRATIO '转股溢价率',"
            "CB_ANAL_PARITYBASEPRICE '平价/底价',"
            "CB_ANAL_IMPLIEDVOL '转债隐含波动率' "
            "from CCBondValuation "
            f"where TRADE_DT>='{StartDate}' AND TRADE_DT<='{EndDate}'"
        )
        data = pd.read_sql(sql, conn)
        data.to_csv(os.path.join(OUTPUT_PATH, "Valuation.csv"), index=False)

    data["TRADE_DT"] = data["TRADE_DT"].astype(str)
    data = data[(data["TRADE_DT"] >= StartDate) & (data["TRADE_DT"] <= EndDate)]
    data.sort_values(by=["TRADE_DT", "S_INFO_WINDCODE"], inplace=True)

    # Wind 隐波字段中会出现 0 或负值，通常代表无效值，需先清洗为缺失。
    data["转债隐含波动率"] = pd.to_numeric(data["转债隐含波动率"], errors="coerce")
    data.loc[data["转债隐含波动率"] <= 0, "转债隐含波动率"] = np.nan
    print("可转债衍生指标已读取")
    return data


def Load_ShareCode(conn, localfile=False):
    """读取公司代码与正股代码映射。"""
    if localfile:
        data = pd.read_csv(localfile)
    else:
        sql = (
            "SELECT [S_INFO_WINDCODE] 'AShare_WINDCODE',"
            "[S_INFO_NAME] 'AShare_NAME',"
            "[S_INFO_COMPCODE] "
            "FROM [winddb1101].[dbo].[ASHAREDESCRIPTION]"
        )
        data = pd.read_sql(sql, conn)
        data.to_csv(os.path.join(OUTPUT_PATH, "ShareCode.csv"), index=False)

    print("正股代码已读取")
    return data


def Load_CbondRating(conn, EndDate, localfile=False):
    """读取主体评级，供样本池剔除评级低于 A 的转债。"""
    if localfile:
        data = pd.read_csv(localfile)
    else:
        sql = (
            "select S_INFO_COMPCODE, ANN_DT 'TRADE_DT', B_INFO_CREDITRATING '当前评级' "
            "from CBondIssuerRating "
            f"where ANN_DT <= '{EndDate}' and B_RATE_STYLE = 1 and B_INFO_CREDITRATING is not null"
        )
        data = pd.read_sql(sql, conn)
        data.to_csv(os.path.join(OUTPUT_PATH, "CbondRating.csv"), index=False)

    data["TRADE_DT"] = data["TRADE_DT"].apply(lambda x: str(int(x)) if pd.notnull(x) else x)
    data = data[data["TRADE_DT"] <= EndDate]
    data = data[["S_INFO_COMPCODE", "TRADE_DT", "当前评级"]]
    data.sort_values(by=["S_INFO_COMPCODE", "TRADE_DT"], inplace=True)
    data.drop_duplicates(subset=["S_INFO_COMPCODE", "TRADE_DT"], keep="last", inplace=True)
    print("可转债评级数据已读取")
    return data


def Load_EODPRICES_Balance(conn1101bond, StartDate, EndDate, localfile=False):
    """读取转债日行情、上市信息与余额。"""
    if localfile:
        data = pd.read_csv(localfile)
    else:
        sql = (
            "select CBONDEODPRICES.[S_INFO_WINDCODE],CBONDEODPRICES.[TRADE_DT],"
            "CBONDEODPRICES.[S_DQ_PRECLOSE] '昨日收盘价',"
            "CBONDEODPRICES.[S_DQ_CLOSE] '收盘价',"
            "CBONDEODPRICES.[S_DQ_VOLUME] '成交量(手)',"
            "CBONDEODPRICES.[S_DQ_AMOUNT] '成交金额(千元)',"
            "CBONDEODPRICES.[S_DQ_TRADESTATUS] '交易状态',"
            "[S_INFO_NAME] '债券简称',"
            "CCBOND.CB_INFO_LISTEDDATE '转债上市日期',"
            "CCBOND.B_ISSUE_AMOUNTACT '实际发行总量(亿元)',"
            "CCBOND.S_INFO_COMPCODE,"
            "CBONDAMOUNT.[B_INFO_OUTSTANDINGBALANCE] 当前余额 "
            "from "
            "(select CBONDEODPRICES.[S_INFO_WINDCODE],CBONDEODPRICES.[TRADE_DT],"
            "CBONDEODPRICES.[S_DQ_PRECLOSE],CBONDEODPRICES.[S_DQ_CLOSE],"
            "CBONDEODPRICES.[S_DQ_VOLUME],CBONDEODPRICES.[S_DQ_AMOUNT],"
            "CBONDEODPRICES.[S_DQ_TRADESTATUS] from CBONDEODPRICES) as CBONDEODPRICES "
            "join "
            "(SELECT CBONDDESCRIPTION.[S_INFO_WINDCODE],[B_INFO_FULLNAME],[S_INFO_NAME],"
            "CCBONDISSUANCE.CB_INFO_LISTEDDATE,CBONDDESCRIPTION.B_ISSUE_AMOUNTACT,"
            "CCBONDISSUANCE.S_INFO_COMPCODE "
            "from CBONDDESCRIPTION "
            "join CCBONDISSUANCE on CBONDDESCRIPTION.[S_INFO_WINDCODE]=CCBONDISSUANCE.[S_INFO_WINDCODE] "
            "where B_INFO_FULLNAME like '%可转换%' and IS_FAILURE=0 and CCBONDISSUANCE.[CB_INFO_LISTDATE]=3 "
            "and (CCBONDISSUANCE.[S_INFO_WINDCODE] like '%.SZ' or CCBONDISSUANCE.[S_INFO_WINDCODE] like '%.SH')) as CCBOND "
            "on CCBOND.S_INFO_WINDCODE=CBONDEODPRICES.S_INFO_WINDCODE "
            "left join "
            "(select CBONDAMOUNT.S_INFO_WINDCODE,CBONDAMOUNT.[S_INFO_ENDDATE],"
            "CBONDAMOUNT.[B_INFO_OUTSTANDINGBALANCE] from CBONDAMOUNT) as CBONDAMOUNT "
            "on CCBond.S_INFO_WINDCODE=CBONDAMOUNT.S_INFO_WINDCODE "
            "and CBONDEODPRICES.TRADE_DT=CBONDAMOUNT.[S_INFO_ENDDATE] "
            f"where TRADE_DT>='{StartDate}' AND TRADE_DT<='{EndDate}'"
        )
        data = pd.read_sql(sql, conn1101bond)
        data["TRADE_DT"] = data["TRADE_DT"].astype(str)
        data = data[(data["TRADE_DT"] >= StartDate) & (data["TRADE_DT"] <= EndDate)]
        data.sort_values(by=["S_INFO_WINDCODE", "TRADE_DT"], inplace=True)
        data.to_csv(os.path.join(OUTPUT_PATH, "EODPRICES_Balance.csv"), index=False)

    print("可转债交易和余额数据已读取")
    return data


def _stock_factor_columns():
    columns = [
        "TRADE_DT",
        "AShare_WINDCODE",
        "正股日涨跌幅",
        "UNDERLYING_VOL_20",
        "UNDERLYING_MONTHLY_VOL_MEDIAN_1Y",
    ]
    columns.extend([f"正股{window}日Amihud指标" for window in STOCK_AMIHUD_WINDOWS])
    columns.extend([f"正股日均换手率{window}" for window in STOCK_TURNOVER_WINDOWS])
    columns.extend([f"正股{window}日MFI指标" for window in STOCK_MFI_WINDOWS])
    columns.extend([f"正股近{window}日涨跌幅（%）" for window in STOCK_MOMENTUM_WINDOWS])
    columns.extend([f"正股{window}日PercentB指标" for window in STOCK_PERCENT_B_WINDOWS])
    columns.extend([f"正股{window}日PriceToHigh指标" for window in STOCK_PRICE_TO_HIGH_WINDOWS])
    columns.extend([f"正股{window}日RSI指标" for window in STOCK_RSI_WINDOWS])
    return columns


def _minute_daily_metric_columns():
    return [
        "TRADE_DT",
        "S_INFO_WINDCODE",
        "日内RSI",
        "温和收益",
        "日内5分钟线收益率方差",
        "日内5分钟线收益率偏度",
        "日内5分钟线成交量变化方差",
        "日内5分钟线成交量变化偏度",
        "日内5分钟线量价相关系数",
        "开盘半小时成交量占比",
        "尾盘半小时成交量占比",
    ]


def _minute_factor_columns():
    columns = []
    columns.extend([f"近{window}日日内5分钟线RSI" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日温和收益均值" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日日内5分钟线收益率方差" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日日内5分钟线收益率偏度均值" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日日内5分钟线量价相关系数均值" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日日内5分钟线量价相关系数波动率" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日日内5分钟线成交量变化方差均值" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日日内5分钟线成交量变化偏度均值" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日开盘半小时成交量占比均值" for window in MINUTE_FACTOR_WINDOWS])
    columns.extend([f"近{window}日尾盘半小时成交量占比均值" for window in MINUTE_FACTOR_WINDOWS])
    return columns


def _calc_daily_minute_metrics(group):
    open_price = pd.to_numeric(group["OPEN"], errors="coerce").replace(0, np.nan)
    close_price = pd.to_numeric(group["CLOSE"], errors="coerce")
    volume = pd.to_numeric(group["VOLUME"], errors="coerce").fillna(0.0)

    # Within-bar return: used for variance, skewness (paper says "48个" matching within-bar count)
    bar_return = close_price / open_price - 1
    bar_return = bar_return.replace([np.inf, -np.inf], np.nan)

    # Bar-to-bar return: used for RSI and 温和收益 (close-to-close captures actual price trajectory)
    # Σlog(close_i/close_{i-1}) ≈ log(close_last/close_first) = actual intraday cumulative return
    bar_return_btob = close_price / close_price.shift(1) - 1
    bar_return_btob = bar_return_btob.replace([np.inf, -np.inf], np.nan)
    log_return_btob = np.log(close_price / close_price.shift(1)).replace([np.inf, -np.inf], np.nan)
    # Mask session boundaries: bar 0 = first morning bar, bar 24 = first afternoon bar
    if len(bar_return_btob) > 0:
        bar_return_btob.iloc[0] = np.nan
        log_return_btob.iloc[0] = np.nan
    if len(bar_return_btob) > 24:
        bar_return_btob.iloc[24] = np.nan
        log_return_btob.iloc[24] = np.nan

    positive_sum = bar_return_btob.clip(lower=0).sum(min_count=1)
    absolute_sum = bar_return_btob.abs().sum(min_count=1)
    intraday_rsi = positive_sum / absolute_sum if pd.notna(absolute_sum) and absolute_sum != 0 else np.nan

    valid_log_return_btob = log_return_btob.dropna()
    if len(valid_log_return_btob) == 0:
        gentle_return = np.nan
    else:
        median = valid_log_return_btob.median()
        mad = (valid_log_return_btob - median).abs().median()
        threshold = 1.96 * mad if pd.notna(mad) else np.nan
        if pd.isna(threshold):
            gentle_return = np.nan
        else:
            mask = (valid_log_return_btob - median).abs() <= threshold
            gentle_return = valid_log_return_btob.loc[mask].sum()

    valid_bar_return = bar_return.dropna()
    return_var = valid_bar_return.var(ddof=1) if len(valid_bar_return) > 1 else np.nan
    return_skew = valid_bar_return.skew() if len(valid_bar_return) > 2 else np.nan

    prev_volume = volume.shift(1).replace(0, np.nan)
    volume_change = (volume / prev_volume - 1).replace([np.inf, -np.inf], np.nan).dropna()
    volume_change_var = volume_change.var(ddof=1) if len(volume_change) > 1 else np.nan
    volume_change_skew = volume_change.skew() if len(volume_change) > 2 else np.nan

    if close_price.nunique(dropna=True) > 1 and volume.nunique(dropna=True) > 1:
        price_volume_corr = close_price.corr(volume)
    else:
        price_volume_corr = np.nan

    total_volume = volume.sum()
    open_ratio = volume.iloc[:6].sum() / total_volume if total_volume > 0 else np.nan
    close_ratio = volume.iloc[-6:].sum() / total_volume if total_volume > 0 else np.nan

    return pd.Series(
        {
            "日内RSI": intraday_rsi,
            "温和收益": gentle_return,
            "日内5分钟线收益率方差": return_var,
            "日内5分钟线收益率偏度": return_skew,
            "日内5分钟线成交量变化方差": volume_change_var,
            "日内5分钟线成交量变化偏度": volume_change_skew,
            "日内5分钟线量价相关系数": price_volume_corr,
            "开盘半小时成交量占比": open_ratio,
            "尾盘半小时成交量占比": close_ratio,
        }
    )


def Load_MinuteDailyMetrics(StartDate, EndDate, localfile=False, minute_dir=MINUTE_INPUT_PATH):
    """从原始 1 分钟线中聚合 5 分钟 K 线，并生成分钟因子的日度底层特征。"""
    empty_columns = _minute_daily_metric_columns()
    history_start = pd.to_datetime(StartDate) - pd.Timedelta(days=400)
    end_dt = pd.to_datetime(EndDate)

    if localfile:
        data = pd.read_csv(localfile)
    else:
        minute_root = Path(minute_dir)
        if not minute_root.exists():
            print("未找到分钟线目录，分钟线因子将留空。")
            return pd.DataFrame(columns=empty_columns)

        csv_files = []
        for path in minute_root.rglob("*.csv"):
            try:
                file_dt = pd.to_datetime(path.stem)
            except Exception:
                continue
            if history_start <= file_dt <= end_dt:
                csv_files.append((file_dt, path))
        csv_files.sort(key=lambda item: item[0])

        if len(csv_files) == 0:
            print("分钟线目录中没有落在样本区间内的文件，分钟线因子将留空。")
            return pd.DataFrame(columns=empty_columns)

        daily_frames = []
        for file_index, (_, path) in enumerate(csv_files, start=1):
            minute_data = pd.read_csv(
                path,
                usecols=["CODE_BOND", "TRADE_DT", "OPEN", "CLOSE", "VOLUME"],
            )
            if len(minute_data) == 0:
                continue

            minute_data["TRADE_DT"] = pd.to_datetime(minute_data["TRADE_DT"])
            minute_data.sort_values(by=["CODE_BOND", "TRADE_DT"], inplace=True)
            minute_data["TRADE_DATE"] = minute_data["TRADE_DT"].dt.normalize()
            minute_data["BAR_ID"] = minute_data.groupby(["CODE_BOND", "TRADE_DATE"]).cumcount() // 5

            bars = (
                minute_data.groupby(["CODE_BOND", "TRADE_DATE", "BAR_ID"], as_index=False)
                .agg({"OPEN": "first", "CLOSE": "last", "VOLUME": "sum"})
            )

            daily_metrics = (
                bars.groupby(["CODE_BOND", "TRADE_DATE"], group_keys=False)
                .apply(_calc_daily_minute_metrics)
                .reset_index()
                .rename(columns={"CODE_BOND": "S_INFO_WINDCODE", "TRADE_DATE": "TRADE_DT"})
            )
            daily_frames.append(daily_metrics)

            if file_index % 100 == 0 or file_index == len(csv_files):
                print(f"分钟线日度特征已处理 {file_index}/{len(csv_files)} 个文件")

        data = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(columns=empty_columns)
        data.to_csv(MINUTE_CACHE_FILE, encoding="utf_8_sig", index=False)

    if len(data) == 0:
        return pd.DataFrame(columns=empty_columns)

    data["TRADE_DT"] = pd.to_datetime(data["TRADE_DT"])
    data = data[(data["TRADE_DT"] >= history_start) & (data["TRADE_DT"] <= end_dt)].copy()
    data.sort_values(by=["S_INFO_WINDCODE", "TRADE_DT"], inplace=True)
    print("转债分钟线日度特征已读取")
    return data[empty_columns]


def Load_StockFactors(conn, StartDate, EndDate, StockCodes, localfile=False):
    """读取正股日线数据并构建正股量价因子。"""
    empty_columns = _stock_factor_columns()
    if localfile:
        data = pd.read_csv(localfile)
    else:
        stock_codes = sorted(
            [
                code
                for code in pd.Series(StockCodes).dropna().astype(str).unique()
                if code.endswith(".SH") or code.endswith(".SZ")
            ]
        )
        if len(stock_codes) == 0:
            return pd.DataFrame(columns=empty_columns)

        query_start_date = (pd.to_datetime(StartDate) - pd.Timedelta(days=1000)).strftime("%Y%m%d")
        code_list = "','".join(stock_codes)
        sql = (
            "select p.TRADE_DT,p.S_INFO_WINDCODE,p.S_DQ_HIGH,p.S_DQ_LOW,p.S_DQ_CLOSE,"
            "p.S_DQ_VOLUME,p.S_DQ_AMOUNT,p.S_DQ_PCTCHANGE,d.S_DQ_FREETURNOVER "
            "from ASHAREEODPRICES p "
            "left join ASHAREEODDERIVATIVEINDICATOR d "
            "on p.S_INFO_WINDCODE=d.S_INFO_WINDCODE and p.TRADE_DT=d.TRADE_DT "
            f"where p.TRADE_DT>='{query_start_date}' AND p.TRADE_DT<='{EndDate}' "
            f"and p.S_INFO_WINDCODE in ('{code_list}')"
        )
        data = pd.read_sql(sql, conn)
        data.to_csv(os.path.join(OUTPUT_PATH, "UnderlyingFactors.csv"), index=False)

    if len(data) == 0:
        return pd.DataFrame(columns=empty_columns)

    data["TRADE_DT"] = pd.to_datetime([str(i) for i in data["TRADE_DT"]])
    data.sort_values(by=["S_INFO_WINDCODE", "TRADE_DT"], inplace=True)
    if "S_DQ_FREETURNOVER" not in data.columns:
        data["S_DQ_FREETURNOVER"] = np.nan
    data["S_DQ_FREETURNOVER"] = pd.to_numeric(data["S_DQ_FREETURNOVER"], errors="coerce")
    data["S_DQ_PCTCHANGE"] = data["S_DQ_PCTCHANGE"] / 100
    grouped = data.groupby("S_INFO_WINDCODE")
    data["正股日涨跌幅"] = data["S_DQ_PCTCHANGE"]

    # 正股 20 日历史波动率保留用于其他分析。
    data["UNDERLYING_VOL_20"] = grouped["S_DQ_PCTCHANGE"].transform(
        lambda x: x.rolling(window=20, min_periods=20).std(ddof=1) * np.sqrt(250) * 100
    )

    # 隐波差新口径：正股过去一年（月频）历史波动率的中位数。
    data["YEAR_MONTH"] = data["TRADE_DT"].dt.to_period("M")
    monthly_vol = (
        data.groupby(["S_INFO_WINDCODE", "YEAR_MONTH"])["S_DQ_PCTCHANGE"]
        .std(ddof=1)
        .mul(np.sqrt(250) * 100)
        .rename("MONTHLY_VOL")
        .reset_index()
        .sort_values(by=["S_INFO_WINDCODE", "YEAR_MONTH"])
    )
    monthly_vol["UNDERLYING_MONTHLY_VOL_MEDIAN_1Y"] = (
        monthly_vol.groupby("S_INFO_WINDCODE")["MONTHLY_VOL"]
        .transform(lambda x: x.shift(1).rolling(window=12, min_periods=12).median())
    )
    data = pd.merge(
        data,
        monthly_vol[["S_INFO_WINDCODE", "YEAR_MONTH", "UNDERLYING_MONTHLY_VOL_MEDIAN_1Y"]],
        on=["S_INFO_WINDCODE", "YEAR_MONTH"],
        how="left",
    )
    # merge 会生成新的 DataFrame / RangeIndex；后续滚动计算必须基于合并后的
    # 最新顺序重新分组，否则旧 groupby 对象的结果会按索引错位。
    data.sort_values(by=["S_INFO_WINDCODE", "TRADE_DT"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Amihud: |收益率| / 成交额，再做滚动均值。
    data["正股日Amihud"] = data["S_DQ_PCTCHANGE"].abs() / data["S_DQ_AMOUNT"].replace(0, np.nan)
    for window in STOCK_AMIHUD_WINDOWS:
        data[f"正股{window}日Amihud指标"] = data.groupby("S_INFO_WINDCODE")["正股日Amihud"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )

    # 正股日均换手率使用 Wind 的自由流通换手率字段。
    for window in STOCK_TURNOVER_WINDOWS:
        data[f"正股日均换手率{window}"] = data.groupby("S_INFO_WINDCODE")["S_DQ_FREETURNOVER"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )

    # MFI: 基于典型价格和资金流的 N 日指标。
    data["典型价格"] = (data["S_DQ_HIGH"] + data["S_DQ_LOW"] + data["S_DQ_CLOSE"]) / 3
    data["原始资金流"] = data["典型价格"] * data["S_DQ_VOLUME"]
    prev_typical = data.groupby("S_INFO_WINDCODE")["典型价格"].shift(1)
    data["正资金流"] = np.where(data["典型价格"] > prev_typical, data["原始资金流"], 0)
    data["负资金流"] = np.where(data["典型价格"] < prev_typical, data["原始资金流"], 0)
    for window in STOCK_MFI_WINDOWS:
        positive_flow = data.groupby("S_INFO_WINDCODE")["正资金流"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).sum()
        )
        negative_flow = data.groupby("S_INFO_WINDCODE")["负资金流"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).sum()
        )
        money_ratio = positive_flow / negative_flow.replace(0, np.nan)
        column = f"正股{window}日MFI指标"
        data[column] = 100 - 100 / (1 + money_ratio)
        data.loc[(negative_flow == 0) & (positive_flow > 0), column] = 100

    close_return = data.groupby("S_INFO_WINDCODE")["S_DQ_CLOSE"].pct_change()
    positive_return = close_return.clip(lower=0)
    absolute_return = close_return.abs()
    for window in STOCK_MOMENTUM_WINDOWS:
        data[f"正股近{window}日涨跌幅（%）"] = (
            data.groupby("S_INFO_WINDCODE")["S_DQ_CLOSE"].pct_change(periods=window) * 100
        )

    for window in STOCK_PERCENT_B_WINDOWS:
        rolling_mean = data.groupby("S_INFO_WINDCODE")["S_DQ_CLOSE"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )
        rolling_std = data.groupby("S_INFO_WINDCODE")["S_DQ_CLOSE"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).std(ddof=1)
        )
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        denom = (upper_band - lower_band).replace(0, np.nan)
        data[f"正股{window}日PercentB指标"] = (data["S_DQ_CLOSE"] - lower_band) / denom

    for window in STOCK_PRICE_TO_HIGH_WINDOWS:
        rolling_high = data.groupby("S_INFO_WINDCODE")["S_DQ_HIGH"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).max()
        )
        data[f"正股{window}日PriceToHigh指标"] = data["S_DQ_CLOSE"] / rolling_high.replace(0, np.nan)

    for window in STOCK_RSI_WINDOWS:
        positive_sum = positive_return.groupby(data["S_INFO_WINDCODE"]).transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).sum()
        )
        absolute_sum = absolute_return.groupby(data["S_INFO_WINDCODE"]).transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).sum()
        )
        data[f"正股{window}日RSI指标"] = positive_sum / absolute_sum.replace(0, np.nan)

    data.drop(columns=["YEAR_MONTH"], inplace=True, errors="ignore")
    data.rename(columns={"S_INFO_WINDCODE": "AShare_WINDCODE"}, inplace=True)
    print("正股量价因子已读取")
    return data[empty_columns]


def Load_Sample(conn1101bond, conn1101, Calender, local=False):
    """拼接转债、主体评级、正股和估值数据，形成因子底表。"""
    StartDate, EndDate = Calender["TRADE_DT"].values[0], Calender["TRADE_DT"].values[-1]

    share_code = Load_ShareCode(conn1101, _resolve_localfile(local, os.path.join(OUTPUT_PATH, "ShareCode.csv")))
    valuation = Load_Valuation(conn1101bond, StartDate, EndDate, _resolve_localfile(local, os.path.join(OUTPUT_PATH, "Valuation.csv")))
    cbond_rating = Load_CbondRating(conn1101bond, EndDate, _resolve_localfile(local, os.path.join(OUTPUT_PATH, "CbondRating.csv")))
    eodprices_balance = Load_EODPRICES_Balance(
        conn1101bond,
        StartDate,
        EndDate,
        _resolve_localfile(local, os.path.join(OUTPUT_PATH, "EODPRICES_Balance.csv")),
    )

    result = pd.merge(eodprices_balance, share_code, on="S_INFO_COMPCODE", how="left")
    result["TRADE_DT"] = result["TRADE_DT"].astype(str)
    result = pd.merge(result, valuation, on=["S_INFO_WINDCODE", "TRADE_DT"], how="left")
    result = result.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y", "Unnamed: 0"], errors="ignore")
    result = pd.merge(result, cbond_rating, on=["S_INFO_COMPCODE", "TRADE_DT"], how="left")

    stock_factors = Load_StockFactors(
        conn1101,
        StartDate,
        EndDate,
        result["AShare_WINDCODE"].dropna().unique(),
        _resolve_localfile(local, os.path.join(OUTPUT_PATH, "UnderlyingFactors.csv")),
    )
    result["TRADE_DT"] = pd.to_datetime(result["TRADE_DT"])
    result = pd.merge(result, stock_factors, on=["TRADE_DT", "AShare_WINDCODE"], how="left")
    return result


# ================================== 样本整理 ==================================
def forward_fill_sample_fields(group):
    """按转债代码在时间序列上补齐余额和主体评级。"""
    group = group.sort_values("TRADE_DT").copy()
    group["当前余额"] = group["当前余额"].ffill()
    group["当前评级"] = group["当前评级"].ffill()
    group["当前余额"] = group["当前余额"].fillna(group["实际发行总量(亿元)"])
    return group


def calculate_turnover_factors(data):
    """计算当日换手率及近 N 日累计换手率。"""
    data = data.copy()
    data["TRADE_DT"] = pd.to_datetime(data["TRADE_DT"])
    data["当日换手率（金额/余额）"] = data["成交金额(千元)"] / data["当前余额"] / 100000
    print("当日换手率计算完成！")

    grouped = data.groupby("S_INFO_WINDCODE")
    for window in BOND_TURNOVER_WINDOWS:
        near_sum = grouped["当日换手率（金额/余额）"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).sum()
        )
        data[f"转债近{window}日换手率"] = near_sum
        if window == 20:
            data["前20个交易日换手率加和"] = grouped["当日换手率（金额/余额）"].transform(
                lambda x, w=window: x.rolling(window=w, min_periods=w).sum()
            )
    return data


def calculate_bond_pctchange_factors(data):
    """计算截至信号日的转债近 N 日涨跌幅。"""
    data = data.copy()
    data["TRADE_DT"] = pd.to_datetime(data["TRADE_DT"])
    grouped = data.groupby("S_INFO_WINDCODE")

    for window in BOND_PCTCHANGE_WINDOWS:
        data[f"转债近{window}日涨跌幅（%）"] = grouped["收盘价"].pct_change(periods=window) * 100
    return data


def calculate_stock_bond_spread_factors(data):
    """计算转债与正股在多窗口上的涨跌幅差。"""
    data = data.copy()
    for window in STOCK_BOND_SPREAD_WINDOWS:
        data[f"近{window}日转债正股涨跌幅之差"] = (
            data[f"转债近{window}日涨跌幅（%）"] - data[f"正股近{window}日涨跌幅（%）"]
        )
    return data


def calculate_stock_bond_correlation_factors(data):
    """计算转债与正股日收益率序列的滚动相关系数。"""
    data = data.copy()
    data["转债日涨跌幅"] = data.groupby("S_INFO_WINDCODE")["收盘价"].pct_change()
    data["正股日涨跌幅"] = pd.to_numeric(data["正股日涨跌幅"], errors="coerce")

    for window in STOCK_BOND_CORR_WINDOWS:
        column = f"转债和正股近{window}日涨跌幅相关系数"
        corr_values = pd.Series(index=data.index, dtype=float)
        for _, group in data.groupby("S_INFO_WINDCODE", sort=False):
            corr_values.loc[group.index] = (
                group["转债日涨跌幅"]
                .rolling(window=window, min_periods=window)
                .corr(group["正股日涨跌幅"])
            )
        data[column] = corr_values
    return data


def calculate_minute_factors(data, minute_metrics):
    """把分钟线日度特征合并到样本池，并生成研报中的分钟线量价因子。"""
    data = data.copy()
    data["TRADE_DT"] = pd.to_datetime(data["TRADE_DT"])

    if minute_metrics is not None and len(minute_metrics) > 0:
        minute_metrics = minute_metrics.copy()
        minute_metrics["TRADE_DT"] = pd.to_datetime(minute_metrics["TRADE_DT"])
        data = pd.merge(data, minute_metrics, on=["TRADE_DT", "S_INFO_WINDCODE"], how="left")
        data.sort_values(by=["S_INFO_WINDCODE", "TRADE_DT"], inplace=True)
        data.reset_index(drop=True, inplace=True)
    else:
        for column in _minute_daily_metric_columns():
            if column not in ("TRADE_DT", "S_INFO_WINDCODE") and column not in data.columns:
                data[column] = np.nan
        data.sort_values(by=["S_INFO_WINDCODE", "TRADE_DT"], inplace=True)
        data.reset_index(drop=True, inplace=True)

    code_index = data["S_INFO_WINDCODE"]
    # 研报公式：N日日内RSI = (1/N) × Σ(日内RSIt / 换手率t)，即 RSI/换手率 的简单滚动均值。
    rsi_over_turnover = (
        pd.to_numeric(data["日内RSI"], errors="coerce")
        / pd.to_numeric(data["当日换手率（金额/余额）"], errors="coerce").replace(0, np.nan)
    )

    for window in MINUTE_FACTOR_WINDOWS:
        data[f"近{window}日日内5分钟线RSI"] = rsi_over_turnover.groupby(code_index).transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )

        data[f"近{window}日温和收益均值"] = data.groupby("S_INFO_WINDCODE")["温和收益"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )
        data[f"近{window}日日内5分钟线收益率方差"] = data.groupby("S_INFO_WINDCODE")["日内5分钟线收益率方差"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )
        data[f"近{window}日日内5分钟线收益率偏度均值"] = data.groupby("S_INFO_WINDCODE")["日内5分钟线收益率偏度"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )
        data[f"近{window}日日内5分钟线量价相关系数均值"] = data.groupby("S_INFO_WINDCODE")["日内5分钟线量价相关系数"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )
        data[f"近{window}日日内5分钟线量价相关系数波动率"] = data.groupby("S_INFO_WINDCODE")["日内5分钟线量价相关系数"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).std(ddof=1)
        )
        data[f"近{window}日日内5分钟线成交量变化方差均值"] = data.groupby("S_INFO_WINDCODE")["日内5分钟线成交量变化方差"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )
        data[f"近{window}日日内5分钟线成交量变化偏度均值"] = data.groupby("S_INFO_WINDCODE")["日内5分钟线成交量变化偏度"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )
        data[f"近{window}日开盘半小时成交量占比均值"] = data.groupby("S_INFO_WINDCODE")["开盘半小时成交量占比"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )
        data[f"近{window}日尾盘半小时成交量占比均值"] = data.groupby("S_INFO_WINDCODE")["尾盘半小时成交量占比"].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).mean()
        )

    return data


# ================================== 因子构建 ==================================
def calculate_core_factors(data):
    """构建双低、隐波差和转债类型。"""
    data = data.copy()
    data["双低"] = data["转股溢价率"] + data["收盘价"]
    data["隐波差"] = data["转债隐含波动率"] - data["UNDERLYING_MONTHLY_VOL_MEDIAN_1Y"]
    # 平底溢价率：平价≥底价时取转股溢价率，否则取纯债溢价率（均为百分比，如30代表30%）
    data["平底溢价率"] = np.where(
        data["平价/底价"] >= 100,
        data["转股溢价率"],
        data["纯债溢价率"],
    )
    data["转债类型"] = np.select(
        [data["平底溢价率"] > 20, data["平底溢价率"] < -20],
        ["偏股型", "偏债型"],
        default="平衡型",
    )
    return data


def calculate_time_series_zscore(data, source_column, output_column, window):
    """按转债代码做时序 ZScore，用于复现研报中的 6M/1Y 时序因子。

    min_periods 设为 window//2：对于隐含波动率等早期覆盖率低（2018年仅31%）的字段，
    若保持 min_periods=window 则 rolling 统计量几乎全为 NaN，导致 Zscore 失效。
    放宽为 window//2 后，只要半个窗口内有足够非空值即可计算，以少量精度换取有效样本量。
    """
    min_obs = max(window // 2, 30)
    grouped = data.groupby("S_INFO_WINDCODE")[source_column]
    rolling_mean = grouped.transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_obs).mean())
    rolling_std = grouped.transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_obs).std(ddof=1))
    data[output_column] = (data[source_column] - rolling_mean) / rolling_std.replace(0, np.nan)
    return data


def build_time_series_factors(data):
    """批量生成研报里用到的估值类时序 ZScore。"""
    data = data.sort_values(by=["S_INFO_WINDCODE", "TRADE_DT"]).copy()
    print("开始计算时序Zscore...")
    for source_column, output_column, window in TIME_SERIES_ZSCORE_CONFIGS:
        data = calculate_time_series_zscore(data, source_column, output_column, window)
        print(f"{output_column} 计算完成")
    return data


# ================================== 主程序 ==================================
def build_factor_dataset(local=False):
    calendar_file = os.path.join(INPUT_PATH, "调仓日期（日频）.xlsx")
    calendar = Load_Calender(calendar_file)

    conn1101 = None
    conn1101bond = None
    if not local:
        conn1101 = pymssql.connect(SQL_HOST, SQL_USER, SQL_PASSWORD, STOCK_DB)
        conn1101bond = pymssql.connect(SQL_HOST, SQL_USER, SQL_PASSWORD, BOND_DB)

    try:
        print("读取数据中...")
        data = Load_Sample(conn1101bond, conn1101, calendar, local=local)
        print("读取完成！")
    finally:
        if conn1101 is not None:
            conn1101.close()
        if conn1101bond is not None:
            conn1101bond.close()

    # 先在单券时间序列内补齐余额和主体评级，再计算流动性与收益类因子。
    data_filled = (
        data.groupby("S_INFO_WINDCODE", group_keys=False)
        .apply(forward_fill_sample_fields)
        .reset_index(drop=True)
    )
    print("填充完成！")

    data_filled = calculate_turnover_factors(data_filled)
    print("转债近N日换手率及前20个交易日换手率加和完成！")

    data_filled = calculate_bond_pctchange_factors(data_filled)
    print("前5/20/60个交易日涨跌幅计算完成！")

    data_filled = calculate_stock_bond_spread_factors(data_filled)
    print("正股量价因子和转债-正股涨跌幅差计算完成！")

    data_filled = calculate_stock_bond_correlation_factors(data_filled)
    print("转债-正股日频相关系数因子计算完成！")

    minute_localfile = _resolve_localfile(local, MINUTE_CACHE_FILE)
    if os.path.isdir(MINUTE_INPUT_PATH):
        minute_localfile = False

    minute_metrics = Load_MinuteDailyMetrics(
        calendar["TRADE_DT"].values[0],
        calendar["TRADE_DT"].values[-1],
        minute_localfile,
    )
    data_filled = calculate_minute_factors(data_filled, minute_metrics)
    print("转债分钟线量价因子计算完成！")

    data_filled = calculate_core_factors(data_filled)
    data_filled = data_filled.drop(["S_INFO_COMPCODE", "AShare_NAME"], axis=1, errors="ignore")
    data_filled.to_csv(os.path.join(OUTPUT_PATH, "data_filled.csv"), encoding="utf_8_sig", index=False)
    print("写入计算指标完成！")

    data_factor = build_time_series_factors(data_filled)
    data_factor.sort_values(by=["S_INFO_WINDCODE", "TRADE_DT"], inplace=True)
    data_factor.to_csv(os.path.join(OUTPUT_PATH, "data_factor.csv"), encoding="utf_8_sig", index=False)
    print_hi("因子计算完成！")
    return data_factor


def main():
    print_hi("PyCharm")
    build_factor_dataset(local=False)


if __name__ == "__main__":
    main()

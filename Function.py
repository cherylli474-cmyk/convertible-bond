
"""
回测与因子分析通用函数库。

主要包含四部分：
1. 调仓日映射与基础数据读取
2. 样本池筛选与因子预处理
3. 持仓、收益与回测绩效计算
4. 因子表现与 RankIC 统计
"""

import math
import os
from datetime import datetime, date

import numpy as np
import pandas as pd
# from WindPy import w
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression as LR
import time


# ================================== 调仓日映射与基础工具 ==================================
def _resolve_localfile(local, path):
    return path if local and os.path.exists(path) else False


def Build_RebalanceMap(RebalanceCalender, DailyCalender):
    """把调仓日映射到前一交易日的信号日，避免未来函数。"""
    RebalanceCalender = RebalanceCalender.copy().sort_values(by='TRADE_DT').reset_index(drop=True)
    DailyTradeDates = np.array(sorted(DailyCalender['TRADE_DT'].astype(str).unique()))
    RebalanceTradeDates = RebalanceCalender['TRADE_DT'].astype(str).values
    SignalIndex = np.searchsorted(DailyTradeDates, RebalanceTradeDates, side='left') - 1
    if np.any(SignalIndex < 0):
        raise ValueError('Rebalance calendar starts before daily calendar.')
    RebalanceCalender['SIGNAL_DT'] = DailyTradeDates[SignalIndex]
    return RebalanceCalender


def _align_to_rebalance_calendar(Calender, data):
    """把日频字段对齐到调仓日；若有 SIGNAL_DT，则取信号日的数据。"""
    if 'SIGNAL_DT' not in Calender.columns:
        return pd.merge(Calender[['TRADE_DT']], data, on='TRADE_DT', how='left')

    result = pd.merge(
        Calender[['TRADE_DT', 'SIGNAL_DT']],
        data,
        left_on='SIGNAL_DT',
        right_on='TRADE_DT',
        how='left'
    )
    result.drop(columns=['SIGNAL_DT', 'TRADE_DT_y'], inplace=True, errors='ignore')
    result.rename(columns={'TRADE_DT_x': 'TRADE_DT'}, inplace=True)
    return result


# ================================== 因子与回测基础数据读取 ==================================

def Load_Factor(Calender, filename):
    '''
    函数名称：Load_Factor
    函数功能：读取因子值数据
    输入参数：Calender：调仓日期；filename：因子值文件
    输出参数：data：调仓日期上的因子值
    '''
    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]
    data = pd.read_csv(filename)
    data['TRADE_DT'] = pd.to_datetime([str(i) for i in data['TRADE_DT']]).strftime('%Y%m%d')

    if 'SIGNAL_DT' in Calender.columns:
        # 研报口径是“前一交易日算分，调仓日交易”，这里先按信号日取值，
        # 再把信号日因子映射回对应的调仓日。
        SignalStartDate, SignalEndDate = Calender['SIGNAL_DT'].values[0], Calender['SIGNAL_DT'].values[-1]
        data = data[(data['TRADE_DT'] >= SignalStartDate) & (data['TRADE_DT'] <= SignalEndDate)]
        data = data.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT']).reset_index(drop=True)
        data = pd.merge(
            Calender[['TRADE_DT', 'SIGNAL_DT']],
            data,
            left_on='SIGNAL_DT',
            right_on='TRADE_DT',
            how='inner'
        )
        data.drop(columns=['TRADE_DT_y', 'SIGNAL_DT'], inplace=True, errors='ignore')
        data.rename(columns={'TRADE_DT_x': 'TRADE_DT'}, inplace=True)
    else:
        data = data[(data['TRADE_DT'] >= StartDate) & (data['TRADE_DT'] <= EndDate)]
        data = data.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT']).reset_index(drop=True)
        data = pd.merge(Calender, data, on='TRADE_DT', how='inner')

    data = data.sort_values(by=['TRADE_DT', 'S_INFO_WINDCODE']).reset_index(drop=True)
    return data



def Load_Calender(filename):
    '''
    函数名称：Load_Calender
    函数功能：读取调仓日期数据
    输入参数：filename：文件名
    输出参数：data：调仓日期
    '''
    data = pd.read_excel(filename)
    data['调仓日期'] = pd.to_datetime([str(i) for i in data['调仓日期']]).strftime('%Y%m%d')
    data.rename(columns={'调仓日期': 'TRADE_DT'}, inplace=True)
    return data

def Load_SQLData(conn1101, conn1101bond, Calender, local=False):
    '''
    函数名称：Load_SQLData
    函数功能：从sql server读取交易日、转债价格、市值、所属行业、ST日期、交易状态（是否停牌）、上市日期，并保存到本地csv文件
    输入参数：conn：sql连接工具；StartDate：开始日期；EndDate：截止日期；local：是否从本地读取（初次运行需设为False，再次运行可设为True，节省时间）
    输出参数：Price：股票价格；Industry：所属行业；
            ST：ST日期；Suspend：交易状态（是否停牌）；ListDate：上市日期
    '''

    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]

    localfile = _resolve_localfile(local, 'Output/Industry.csv')
    Industry = Load_Industry(conn1101, localfile)

    localfile = _resolve_localfile(local, 'Output/Price.csv')
    Price = Load_Price(conn1101bond, StartDate, EndDate, localfile)
    ExitActive = (
        Price[Price['TRADE_DT'].isin(Calender['TRADE_DT'].astype(str).unique())][['TRADE_DT', 'S_INFO_WINDCODE', 'EXIT_ACTIVE']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    localfile = _resolve_localfile(local, 'Output/ListDate.csv')
    ListDate = Load_ListDate(conn1101bond, localfile)

    localfile = _resolve_localfile(local, 'Output/Suspend.csv')
    Suspend = Load_Suspend(conn1101bond, localfile)

    localfile = _resolve_localfile(local, 'Output/Balance.csv')
    Balance = Load_Balance(localfile)

    localfile = _resolve_localfile(local, 'Output/Rating.csv')
    Rating = Load_Rating(localfile)

    localfile = _resolve_localfile(local, 'Output/CbondType.csv')
    CbondType = Load_CbondType(localfile)

    localfile = _resolve_localfile(local, 'Output/Turnover.csv')
    Turnover = Load_Turnover(localfile)

    Suspend = _align_to_rebalance_calendar(Calender, Suspend)
    Balance = _align_to_rebalance_calendar(Calender, Balance)
    Rating = _align_to_rebalance_calendar(Calender, Rating)
    CbondType = _align_to_rebalance_calendar(Calender, CbondType)
    Turnover = _align_to_rebalance_calendar(Calender, Turnover)

    return Price, Industry, ListDate, Suspend, Balance, Rating, CbondType, Turnover, ExitActive


def Load_BondExitEvents(conn, localfile=False):
    columns = ['S_INFO_WINDCODE', 'EVENT_DT', 'EVENT_TYPE']
    if localfile:
        data = pd.read_csv(localfile)
    elif conn is None:
        return pd.DataFrame(columns=columns)
    else:
        forced_sql = (
            "select S_INFO_WINDCODE,ANN_DT 'EVENT_DT' "
            "from CCBONDCONVERSION "
            "where ISFORCED=1 or FORCED_CONV_DATE is not null"
        )
        forced = pd.read_sql(forced_sql, conn)
        forced['EVENT_TYPE'] = 'FORCED_REDEEM'

        desc_sql = (
            "select S_INFO_WINDCODE,B_INFO_DELISTDATE,B_INFO_MATURITYDATE,B_INFO_ENDDATE "
            "from CBONDDESCRIPTION where B_INFO_FULLNAME like '%可转换%'"
        )
        desc = pd.read_sql(desc_sql, conn)
        event_frames = [forced]
        for source_col, event_type in [
            ('B_INFO_DELISTDATE', 'DELIST'),
            ('B_INFO_MATURITYDATE', 'MATURITY'),
            ('B_INFO_ENDDATE', 'ENDDATE'),
        ]:
            event_frame = desc[['S_INFO_WINDCODE', source_col]].rename(columns={source_col: 'EVENT_DT'}).copy()
            event_frame['EVENT_TYPE'] = event_type
            event_frames.append(event_frame)

        data = pd.concat(event_frames, ignore_index=True)
        data.to_csv('Output/BondExitEvents.csv', index=False)

    data = data.drop(columns=[col for col in data.columns if str(col).startswith('Unnamed:')], errors='ignore')
    if len(data) == 0:
        return pd.DataFrame(columns=columns)

    data = data[columns].copy()
    data['EVENT_DT'] = pd.to_datetime(data['EVENT_DT'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
    data = data.dropna(subset=['S_INFO_WINDCODE', 'EVENT_DT'])
    data['S_INFO_WINDCODE'] = data['S_INFO_WINDCODE'].astype(str)
    data['EVENT_TYPE'] = data['EVENT_TYPE'].astype(str)
    data = data.sort_values(by=['S_INFO_WINDCODE', 'EVENT_DT', 'EVENT_TYPE']).drop_duplicates().reset_index(drop=True)
    return data


def _resolve_exit_sell_date(available_dates, event_dt, event_type):
    if len(available_dates) == 0 or pd.isna(event_dt):
        return None

    if event_type == 'FORCED_REDEEM':
        start_idx = np.searchsorted(available_dates, event_dt, side='right')
        sell_idx = start_idx + 4
        return available_dates[sell_idx] if sell_idx < len(available_dates) else None

    sell_idx = np.searchsorted(available_dates, event_dt, side='left')
    if sell_idx < len(available_dates):
        return available_dates[sell_idx]

    sell_idx = np.searchsorted(available_dates, event_dt, side='right') - 1
    return available_dates[sell_idx] if sell_idx >= 0 else None


def Apply_BondExitEvents(Price, ExitEvents):
    Price = Price.copy()
    Price['FORCED_SELL'] = False
    Price['EXIT_ACTIVE'] = False
    Price['EXIT_EVENT_TYPE'] = ''

    if len(Price) == 0 or len(ExitEvents) == 0:
        return Price

    mapped_rows = []
    for code, events in ExitEvents.groupby('S_INFO_WINDCODE'):
        code_dates = (
            Price.loc[Price['S_INFO_WINDCODE'] == code, ['TRADE_DT', 'CAN_SELL']]
            .drop_duplicates()
            .sort_values(by='TRADE_DT')
        )
        available_dates = code_dates.loc[code_dates['CAN_SELL'].fillna(False), 'TRADE_DT'].astype(str).to_numpy()
        if len(available_dates) == 0:
            continue

        for _, event in events.sort_values(by=['EVENT_DT', 'EVENT_TYPE']).iterrows():
            sell_dt = _resolve_exit_sell_date(available_dates, event['EVENT_DT'], event['EVENT_TYPE'])
            if sell_dt is not None:
                mapped_rows.append((code, sell_dt, event['EVENT_DT'], event['EVENT_TYPE']))

    if len(mapped_rows) == 0:
        return Price

    mapped = pd.DataFrame(mapped_rows, columns=['S_INFO_WINDCODE', 'SELL_DT', 'EVENT_DT', 'EVENT_TYPE'])
    earliest_sell = mapped.groupby('S_INFO_WINDCODE')['SELL_DT'].transform('min')
    mapped = mapped[mapped['SELL_DT'] == earliest_sell].copy()

    sell_flags = (
        mapped.groupby(['S_INFO_WINDCODE', 'SELL_DT'])['EVENT_TYPE']
        .agg(lambda x: '|'.join(sorted(set(x))))
        .reset_index()
        .rename(columns={'SELL_DT': 'TRADE_DT', 'EVENT_TYPE': 'EXIT_EVENT_TYPE'})
    )
    first_exit = (
        mapped.groupby('S_INFO_WINDCODE')['SELL_DT']
        .min()
        .reset_index(name='EXIT_DT')
    )

    Price = pd.merge(Price, sell_flags, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left', suffixes=('', '_EVENT'))
    Price['EXIT_EVENT_TYPE'] = Price['EXIT_EVENT_TYPE_EVENT'].fillna(Price['EXIT_EVENT_TYPE'])
    Price['FORCED_SELL'] = Price['EXIT_EVENT_TYPE'].ne('')
    Price.drop(columns=['EXIT_EVENT_TYPE_EVENT'], inplace=True, errors='ignore')

    Price = pd.merge(Price, first_exit, on='S_INFO_WINDCODE', how='left')
    Price['EXIT_ACTIVE'] = Price['EXIT_DT'].notna() & (Price['TRADE_DT'] >= Price['EXIT_DT'])
    Price['CAN_BUY'] = Price['CAN_BUY'] & (~Price['EXIT_ACTIVE'])
    Price['CAN_SELL'] = Price['CAN_SELL'] | Price['FORCED_SELL']
    Price.drop(columns=['EXIT_DT'], inplace=True, errors='ignore')
    return Price


def Load_Price(conn, StartDate, EndDate, localfile=False):
    '''
    函数名称：Load_Price
    函数功能：从sql server读取股票价格数据
    输入参数：conn：sql连接工具；StartDate：开始日期；EndDate：截止日期；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：股票价格数据
    '''
    required_columns = [
        'TRADE_DT',
        'S_INFO_WINDCODE',
        'S_DQ_PRECLOSE',
        'S_DQ_CLOSE',
        'S_DQ_VOLUME',
        'S_DQ_AMOUNT',
        'S_DQ_TRADESTATUS',
    ]

    if localfile:
        data = pd.read_csv(localfile)
    else:
        sql = (
            "select TRADE_DT,CbondEODPrices.S_INFO_WINDCODE,S_DQ_PRECLOSE,S_DQ_CLOSE,"
            "S_DQ_VOLUME,S_DQ_AMOUNT,S_DQ_TRADESTATUS "
            "from CbondEODPrices "
            "join CbondDescription on CbondEODPrices.S_INFO_Windcode=CbondDescription.S_INFO_Windcode "
            "where CbondDescription.B_INFO_FULLNAME like '%可转换%' "
            f"and TRADE_DT>='{StartDate}' AND TRADE_DT<= '{EndDate}'"
        )
        data = pd.read_sql(sql, conn)
        data.to_csv('Output/Price.csv', index=False)

    data = data.drop(columns=[col for col in data.columns if str(col).startswith('Unnamed:')], errors='ignore')

    # 兼容旧版缓存文件：若 Price.csv 缺少成交量/金额/交易状态，则尝试从样本底表缓存补齐。
    missing_columns = [col for col in required_columns if col not in data.columns]
    eod_balance_file = 'Output/EODPRICES_Balance.csv'
    if missing_columns and os.path.exists(eod_balance_file):
        supplement = pd.read_csv(eod_balance_file)
        supplement = supplement.rename(
            columns={
                '昨日收盘价': 'S_DQ_PRECLOSE',
                '收盘价': 'S_DQ_CLOSE',
                '成交量(手)': 'S_DQ_VOLUME',
                '成交金额(千元)': 'S_DQ_AMOUNT',
                '交易状态': 'S_DQ_TRADESTATUS',
            }
        )
        supplement = supplement[
            [col for col in required_columns if col in supplement.columns]
        ].drop_duplicates(subset=['TRADE_DT', 'S_INFO_WINDCODE'])
        data = pd.merge(
            data,
            supplement,
            on=['TRADE_DT', 'S_INFO_WINDCODE'],
            how='left',
            suffixes=('', '_SUPP')
        )
        for col in required_columns:
            supp_col = f'{col}_SUPP'
            if supp_col in data.columns:
                if col not in data.columns:
                    data[col] = data[supp_col]
                else:
                    data[col] = data[col].fillna(data[supp_col])
                data.drop(columns=[supp_col], inplace=True)

    for col in required_columns:
        if col not in data.columns:
            data[col] = np.nan

    trade_dt = data['TRADE_DT'].astype(str).str.replace('.0', '', regex=False)
    data['TRADE_DT'] = pd.to_datetime(trade_dt, format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
    data = data.dropna(subset=['TRADE_DT', 'S_INFO_WINDCODE'])
    data = data[(data.TRADE_DT >= StartDate) & (data.TRADE_DT <= EndDate)]
    data.sort_values(by=['TRADE_DT', 'S_INFO_WINDCODE'], inplace=True)

    numeric_columns = ['S_DQ_PRECLOSE', 'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    amount = data['S_DQ_AMOUNT']
    volume = data['S_DQ_VOLUME']
    data['VWAP'] = np.where(
        volume.gt(0) & amount.gt(0),
        amount * 100.0 / volume,
        np.nan
    )
    data['VWAP'] = data['VWAP'].fillna(data['S_DQ_CLOSE'])

    trade_status = data['S_DQ_TRADESTATUS'].fillna('交易').astype(str)
    data['IS_SUSPENDED'] = trade_status.str.contains('停牌')
    data['CAN_BUY'] = (~data['IS_SUSPENDED']) & data['VWAP'].notna() & data['S_DQ_VOLUME'].fillna(0).gt(0)
    data['CAN_SELL'] = data['CAN_BUY']

    if 'BUY_BLOCKED' in data.columns:
        data['CAN_BUY'] = data['CAN_BUY'] & (~data['BUY_BLOCKED'].fillna(False).astype(bool))
    if 'SELL_BLOCKED' in data.columns:
        data['CAN_SELL'] = data['CAN_SELL'] & (~data['SELL_BLOCKED'].fillna(False).astype(bool))

    exit_event_localfile = 'Output/BondExitEvents.csv' if localfile and os.path.exists('Output/BondExitEvents.csv') else False
    exit_events = Load_BondExitEvents(conn, exit_event_localfile)
    data = Apply_BondExitEvents(data, exit_events)

    data['PRE_TO_VWAP_RETURN'] = np.where(
        data['S_DQ_PRECLOSE'].gt(0) & data['VWAP'].notna(),
        data['VWAP'] / data['S_DQ_PRECLOSE'] - 1,
        0
    )
    data['VWAP_TO_CLOSE_RETURN'] = np.where(
        data['VWAP'].gt(0) & data['S_DQ_CLOSE'].notna(),
        data['S_DQ_CLOSE'] / data['VWAP'] - 1,
        0
    )
    data['RETURN'] = np.where(
        data['S_DQ_PRECLOSE'].gt(0) & data['S_DQ_CLOSE'].notna(),
        data['S_DQ_CLOSE'] / data['S_DQ_PRECLOSE'] - 1,
        0
    )
    data = data.reset_index(drop=True)
    return data


def Load_Industry(conn, localfile=False):
    '''
    函数名称：Load_Industry
    函数功能：从sql server读取所属行业数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：所属行业数据[AShare_WINDCODE,IND_CODE, ENTRY_DT,REMOVE_DT,S_INFO_WINDCODE]
    '''
    if localfile:
        data = pd.read_csv(localfile)
    else:
        sql='select S_INFO_WINDCODE AShare_WINDCODE, substring(citics_ind_code, 1, 4) as IND_CODE, ENTRY_DT,REMOVE_DT from AShareIndustriesClassCITICS'
        data1 = pd.read_sql(sql, conn)       #此sql语句截取citics_ind_code列中的前四个字符串命名为IND_CODE
        data2 = pd.read_csv('Output/data_factor.csv')
        data2 = data2[['S_INFO_WINDCODE','AShare_WINDCODE']] #读取正股代码
        # 去除重复行
        data2 = data2.drop_duplicates()

        data = pd.merge(data1,data2,on='AShare_WINDCODE',how='left')
        data.to_csv('Output/Industry.csv')
    data['ENTRY_DT'] = pd.to_datetime([str(i) for i in data['ENTRY_DT']]).strftime('%Y%m%d')
    data['REMOVE_DT'].fillna('21000101', inplace=True)
    data['REMOVE_DT'] = pd.to_datetime([str(int(i)) for i in data['REMOVE_DT']]).strftime('%Y%m%d')
    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_Suspend(conn, localfile):
    '''
    函数名称：Load_Suspend
    函数功能：从sql server读取交易状态（是否停牌）数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：交易状态（是否停牌）数据:交易0、停牌1、N（上市）、XD（除息）、XR（除权）、DR（除权除息）

    '''
    if localfile:
        data = pd.read_csv(localfile)
    else:
        sql="SELECT  CbondEODPrices.S_INFO_WINDCODE,TRADE_DT,   CASE WHEN S_DQ_TRADESTATUS = '交易' THEN 0     WHEN S_DQ_TRADESTATUS = '停牌' THEN 1   WHEN S_DQ_TRADESTATUS = '上市' THEN 2    ELSE   3    END AS S_DQ_TRADESTATUSCODE FROM CBONDEODPRICES join CbondDescription on CbondEODPrices.S_INFO_Windcode=CbondDescription.S_INFO_Windcode where CbondDescription.B_INFO_FULLNAME like '%可转换%' "
        data = pd.read_sql(sql, conn)
        
        # data.TRADE_DT = data.TRADE_DT.apply(str)
        data.to_csv('Output/Suspend.csv')
    data['TRADE_DT'] = pd.to_datetime([str(i) for i in data['TRADE_DT']]).strftime('%Y%m%d')
    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_ListDate(conn, localfile):
    '''
    函数名称：Load_ListDate
    函数功能：从sql server读取上市日期数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：上市日期数据
    '''
    if localfile:
        data = pd.read_csv(localfile)
    else:
        sql = "select S_INFO_WINDCODE,B_INFO_LISTDATE '转债上市日期' from CbondDESCRIPTION where B_INFO_FULLNAME like '%可转换%'"
        data = pd.read_sql(sql, conn)
        data.to_csv('Output/ListDate.csv')

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data

def Load_Balance(localfile):
    '''
    函数名称：Load_Balance
    函数功能：从csv读取余额数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：余额数据
    '''
    if localfile:
        data = pd.read_csv(localfile)
    else:
        data = pd.read_csv('Output/data_factor.csv')
        data = data[['S_INFO_WINDCODE','TRADE_DT','当前余额']]
        data.to_csv('Output/Balance.csv')
    data['TRADE_DT'] = pd.to_datetime([str(i) for i in data['TRADE_DT']]).strftime('%Y%m%d')
    data.sort_values(by=['TRADE_DT','S_INFO_WINDCODE'], inplace=True)
    data.drop_duplicates(subset=['TRADE_DT', 'S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data

def Load_Rating(localfile):
    '''
    函数名称：Load_Rating
    函数功能：从csv读取评级数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：评级数据
    '''
    if localfile:
        data = pd.read_csv(localfile)
    else:
        data = pd.read_csv('Output/data_factor.csv')
        data = data[['S_INFO_WINDCODE','TRADE_DT','当前评级']]
        data.to_csv('Output/Rating.csv')
    data['TRADE_DT'] = pd.to_datetime([str(i) for i in data['TRADE_DT']]).strftime('%Y%m%d')
    data.sort_values(by=['TRADE_DT','S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data

def Load_CbondType(localfile):
    '''
    函数名称：Load_CbondType
    函数功能：从csv读取转债类型
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：转债类型
    '''
    if localfile:
        data = pd.read_csv(localfile)
    else:
        data = pd.read_csv('Output/data_factor.csv')
        data = data[['S_INFO_WINDCODE','TRADE_DT','转债类型']]
        data.to_csv('Output/CbondType.csv')
    data['TRADE_DT'] = pd.to_datetime([str(i) for i in data['TRADE_DT']]).strftime('%Y%m%d')
    data.sort_values(by=['TRADE_DT','S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data




def Load_Turnover(localfile):
    '''
    函数名称：Load_Turnover
    函数功能：从csv读取换手率数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：评级数据
    '''
    if localfile:
        data = pd.read_csv(localfile)
    else:
        data = pd.read_csv('Output/data_factor.csv')
        data = data[['S_INFO_WINDCODE','TRADE_DT','前20个交易日换手率加和']]
        data.to_csv('Output/Turnover.csv')
    data['TRADE_DT'] = pd.to_datetime([str(i) for i in data['TRADE_DT']]).strftime('%Y%m%d')
    data.sort_values(by=['TRADE_DT','S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data

def Load_Return(conn, Calender ):

    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]

    sql = "select S_DQ_PCTCHANGE , TRADE_DT from AINDEXEODPRICES where S_INFO_WINDCODE = \'000905.SH\' AND TRADE_DT >=\'" + StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
    Return = pd.read_sql(sql, conn)
    Return.rename(columns={'S_DQ_PCTCHANGE': 'RETURN'}, inplace=True)
    Return['RETURN'] = Return['RETURN'] / 100
    Return.set_index('TRADE_DT', inplace=True)
    Return.sort_index(inplace = True)
    return Return

def LoadRiskFreeReturn(filename, Calender):
    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]
    data = pd.read_excel(filename, skiprows=1)
    data['Date'] = list(datetime.strftime(x, '%Y%m%d') for x in data['Date'])
    Return = data[(data['Date'] >= StartDate) & (data['Date'] <= EndDate)].rename(columns={'无风险利率': 'RETURN'})
    Return = Return[['Date', 'RETURN']].set_index('Date')
    Return.sort_index(inplace=True)
    return Return



# ================================== 因子预处理 ==================================
def Mad_Standard(group, CbondType):
    '''
    函数名称：Mad_Standard
    函数功能：去极值和标准化，groupby内部函数
    输入参数：group：单日因子值
    输出参数：Result：处理后因子值
    '''
    # MAD 去极值：先用截面中位数和MAD识别异常值。
    ColumnName=group.columns[2]
    median = np.nanmedian(group[ColumnName])
    diff_median = ((group[ColumnName] - median).abs()).quantile(0.5)
    upper_limit = median + 3 * diff_median
    lower_limit = median - 3 * diff_median
    
    # 找到超出极值范围的索引
    upper_outliers_index = (group[ColumnName] > upper_limit)
    lower_outliers_index = (group[ColumnName] < lower_limit)

    group[ColumnName+'_Mad'] = group[ColumnName]

    # 按照研报把极值均匀插值到3~3.5倍MAD的区间
    group.loc[upper_outliers_index, ColumnName+'_Mad'] = interpolate_outliers(group.loc[upper_outliers_index, [ColumnName+'_Mad']], median + 3 * diff_median, median + 3.5 * diff_median)
    group.loc[lower_outliers_index, ColumnName+'_Mad'] = interpolate_outliers(group.loc[lower_outliers_index, [ColumnName+'_Mad']], median - 3.5 * diff_median, median - 3 * diff_median)
    
    group=pd.merge(group, CbondType, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')

    # 标准化口径：在偏债型 / 平衡型 / 偏股型三类转债内部做分组 ZScore。
    group = group.groupby('转债类型').apply(Standard).reset_index(drop=True)

    Result = group[['TRADE_DT', 'S_INFO_WINDCODE',ColumnName, ColumnName+'_Mad_Stand']]
    return Result

def Standard(data):
    ColumnName=data.columns[2]
    data[ColumnName+'_Mad_Stand'] = (data[ColumnName+'_Mad'] - np.nanmean(data[ColumnName+'_Mad'])) / np.nanstd(data[ColumnName+'_Mad'], ddof=1)
    return data



def interpolate_outliers(outliers, a, b):
    ColumnName = outliers.columns[0]
    if len(outliers) == 0:
        return outliers
    if len(outliers) == 1:
        return pd.Series([(a + b) / 2], index=outliers.index)
    outliers = outliers.copy()
    outliers['rank'] = outliers[[ColumnName]].rank(ascending=True, method='average', axis=0, na_option='keep')
    outliers['new_' + ColumnName] = a + (b - a) * (outliers['rank'] - 1) / (len(outliers) - 1)
    return outliers['new_' + ColumnName]

def Get_Residual(group, MarketCap, Industry):
    '''
    函数名称：Get_Residual
    函数功能：对因子进行 行业中性化，groupby内部函数
    输入参数：group：单日因子值；Industry：所属行业数据
    输出参数：Result：处理后因子值
    '''
    # 把市值部分去掉,行业也不需要
    # 中性化
    group['Factor_Neu']=np.nan
    # group = pd.merge(group, MarketCap, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')
    # 先取出[ENTRY_DT,REMOVE_DT]包含当日的所属行业数据
    Industry_group=pd.merge(Industry,group[['TRADE_DT','S_INFO_WINDCODE']],on='S_INFO_WINDCODE',how='left')
    
    Industry_group=Industry_group[(Industry_group.TRADE_DT>=Industry_group.ENTRY_DT)&(Industry_group.TRADE_DT<=Industry_group.REMOVE_DT)]
    group = pd.merge(group, Industry_group[['TRADE_DT','S_INFO_WINDCODE','IND_CODE']], on=['TRADE_DT','S_INFO_WINDCODE'], how='left')

    # 对不为nan的部分进行线性回归取残差
    NotNanIndexs = np.logical_not(pd.isnull(group['Factor_Mad_Stand'])).values & np.logical_not(pd.isnull(group['S_VAL_MV_Mad_Stand'])).values & np.logical_not(pd.isnull(group['IND_CODE'])).values
    if any(NotNanIndexs):
        X = np.concatenate([group.loc[NotNanIndexs,'S_VAL_MV_Mad_Stand'].values.reshape(-1, 1),pd.get_dummies(group.loc[NotNanIndexs,'IND_CODE']).values],axis=1)
        Y = group.loc[NotNanIndexs,'Factor_Mad_Stand'].values.reshape(-1, 1)
        reg = LR()
        reg.fit(X, Y)
        residual = Y - reg.predict(X)
        group.loc[NotNanIndexs,'Factor_Neu'] = residual.flatten()
    Result = group[['TRADE_DT', 'S_INFO_WINDCODE', 'Factor', 'Factor_Neu']]
    return Result


# ================================== 分组与样本筛选 ==================================
def Cal_Stratify(data, GroupNum):
    '''
    函数名称：Cal_Stratify
    函数功能：获取因子值分组
    输入参数：data：因子值；GroupNum：组数
    输出参数：Result：因子分组
    '''
    # 怎么分组呢？
    # 按照ep分成3组，分别计算收益率
    data_Rank = data.groupby('TRADE_DT').apply(Get_Rank).reset_index(drop=True)
    length = data_Rank.groupby('TRADE_DT')[['RANK']].count()  # 计算每月共有多少只股票
    # bins记录分组的边界，是分组的依据（左闭右开）
    bins = pd.DataFrame(length['RANK'].apply(lambda _: [math.ceil(_ / GroupNum * i) for i in range(GroupNum + 1)]))
    #print(bins)
    GroupLabels = [str(i) for i in range(GroupNum, 0, -1)]  # 最高因子值对应最高组号
    data_Rank = data_Rank.groupby('TRADE_DT').apply(Get_Group, bins, GroupLabels).reset_index(drop=True)
    Result = data_Rank[['TRADE_DT', 'S_INFO_WINDCODE', 'GROUP']]
    return Result


def Get_Rank(group):
    '''
    函数名称：Get_Rank
    函数功能：排序，groupby内部函数
    输入参数：group：待排序值
    输出参数：group：添加了一列序号
    '''
    group['RANK'] = group[['Factor_Mad_Stand']].rank(ascending=False, method='min', axis=0, na_option='keep')
    return group

def Get_Group(group, bins, GroupLabels):
    '''
    函数名称：Get_Group
    函数功能：分组，groupby内部函数
    输入参数：group：待排序值；bins分组边界；GroupLabels：各组标签
    输出参数：group：添加了一列组标签
    '''
    raw_bins = bins.loc[group.TRADE_DT.values[0], 'RANK']
    if raw_bins[1] == 0:
        group['GROUP'] = 0
        return group
    # 样本数少于组数时 math.ceil 会产生重复边界（如 [0,1,2,2]），去重后裁剪 labels
    unique_bins = list(dict.fromkeys(raw_bins))
    n_intervals = len(unique_bins) - 1
    if n_intervals <= 0:
        group['GROUP'] = 0
        return group
    adjusted_labels = GroupLabels[:n_intervals]
    group['GROUP'] = pd.cut(group.RANK, unique_bins, labels=adjusted_labels)
    return group


def Cal_Sift(data, ListDate, Suspend, Balance, Rating, Turnover, ExitActive=None, trading_dates=None):
    '''
    函数名称：Cal_Sift
    函数功能：对股票池进行筛选（1：不为ST，2：不为停牌，3：上市满6个月）
    输入参数：data：股票池；ST：ST日期数据；Suspend：交易状态（是否停牌）数据；ListDate：上市日期数据
    输出参数：data_Sift：筛选后的股票池
    '''
    # 样本池筛选顺序与研报一致：
    # 1. 剔除停牌
    # 2. 剔除上市不足 10 天
    # 3. 剔除余额低于 2 亿
    # 4. 剔除主体评级低于 A
    # 5. 剔除近 20 日换手率过高
    data_tmp1 = Sift_Suspend(data, Suspend)
    data_tmp2 = Sift_ExitActive(data_tmp1, ExitActive)
    data_tmp3 = Sift_ListDate(data_tmp2, ListDate, trading_dates=trading_dates)
    data_tmp4 = Sift_Balance(data_tmp3, Balance)
    data_tmp5 = Sift_Rating(data_tmp4, Rating)
    data_Sift = Sift_Turnover(data_tmp5, Turnover)
    data_Sift = data_Sift.sort_values(by=['TRADE_DT','S_INFO_WINDCODE']).reset_index(drop=True)
    return data_Sift


def Sift_ExitActive(data, ExitActive):
    if ExitActive is None or len(ExitActive) == 0:
        return data

    data = pd.merge(data, ExitActive, on=['TRADE_DT', 'S_INFO_WINDCODE'], how='left')
    data['EXIT_ACTIVE'] = data['EXIT_ACTIVE'].fillna(False).astype(bool)
    data = data[~data['EXIT_ACTIVE']]
    return data[['TRADE_DT', 'S_INFO_WINDCODE', 'Factor']]

def Sift_Turnover(data, Turnover):
    '''
    函数名称：Sift_Turnover
    函数功能：根据前20个交易日换手率筛选股票池
    输入参数：data：股票池；Turnover：前20个交易日换手率数据；
    输出参数：Result：筛选后的股票池
    '''
    data = pd.merge(data, Turnover, on=['TRADE_DT', 'S_INFO_WINDCODE'], how='left')    
    # 剔除前20个交易日换手率加和大于1的
    data = data[data['前20个交易日换手率加和'] <= 1]

    Result = data[['TRADE_DT', 'S_INFO_WINDCODE','Factor']]
    return Result

def Sift_Rating(data, Rating):
    '''
    函数名称：Sift_Rating
    函数功能：根据评级筛选股票池
    输入参数：data：股票池；Rating：余额数据；
    输出参数：Result：筛选后的股票池
    '''
    Rating.drop_duplicates(subset=['TRADE_DT', 'S_INFO_WINDCODE'], inplace=True)

    data = pd.merge(data, Rating, on=['TRADE_DT', 'S_INFO_WINDCODE'], how='left')    
    # 指定需要剔除的评级范围
    exclude_values = ['CCC', 'CC', 'C', 'BBB+', 'BBB-', 'BBB', 'BB+', 'BB-', 'BB', 'B+', 'B', 'B-', 'A-']

    # 剔除评级低于A的，保留评级高于等于A的
    data = data[(~data['当前评级'].isin(exclude_values))]

    Result = data[['TRADE_DT', 'S_INFO_WINDCODE','Factor']]
    return Result


def Sift_Balance(data, Balance):
    '''
    Sift_Balance
    函数功能：根据余额筛选股票池
    输入参数：data：股票池；Balance：余额数据；
    输出参数：Result：筛选后的股票池
    '''
    Balance = Balance.drop_duplicates(subset=['TRADE_DT', 'S_INFO_WINDCODE'])
    data = pd.merge(data, Balance, on=['TRADE_DT', 'S_INFO_WINDCODE'], how='left')
    Result = data[data['当前余额'] >= 2]
    Result = Result[['TRADE_DT', 'S_INFO_WINDCODE', 'Factor']]
    return Result


def Sift_Suspend(data, Suspend):
    '''
    函数名称：Sift_Suspend
    函数功能：根据是否处于停牌筛选股票池
    输入参数：data：股票池；Suspend：交易状态（是否停牌）数据；
    输出参数：Result：筛选后的股票池
    '''
    data = pd.merge(data, Suspend, on=['TRADE_DT', 'S_INFO_WINDCODE'], how='left')
    data = data[data['S_DQ_TRADESTATUSCODE'] !=1 ]
    Result = data[['TRADE_DT', 'S_INFO_WINDCODE','Factor']]
    return Result


def Sift_ListDate(data, ListDate, trading_dates=None):
    '''
    函数名称：Sift_ListDate
    函数功能：根据是否满足上市超过10个交易日筛选股票池
    输入参数：data：股票池；ListDate：上市日期数据；trading_dates：交易日序列（提供时按交易日计算，否则按自然日）
    输出参数：Result：筛选后的股票池
    '''
    data = pd.merge(data, ListDate, on=['S_INFO_WINDCODE'], how='left')
    if trading_dates is not None:
        sorted_td = np.array(sorted(str(d) for d in trading_dates))
        tdt = data['TRADE_DT'].astype(str).values
        ldt = data['转债上市日期'].apply(lambda x: str(int(x)) if pd.notnull(x) else '').values
        deltas = []
        for t, l in zip(tdt, ldt):
            if not l:
                deltas.append(0)
                continue
            i_l = int(np.searchsorted(sorted_td, l, side='left'))
            i_t = int(np.searchsorted(sorted_td, t, side='right')) - 1
            deltas.append(max(0, i_t - i_l))
        data['DaysDelta'] = deltas
    else:
        Date = [datetime.strptime(i, "%Y%m%d") for i in data['TRADE_DT']]
        ListDate_col = [datetime.strptime(str(int(i)), "%Y%m%d") for i in data['转债上市日期']]
        data['DaysDelta'] = [(Date[i] - ListDate_col[i]).days for i in range(len(Date))]
    data = data[(data['DaysDelta'] > 10)]
    Result = data[['TRADE_DT', 'S_INFO_WINDCODE','Factor']]
    return Result

def Fillna_WithIndustryMean(group, Industry):
    # 缺失值处理：按当日所属中信一级行业的截面中位数填充。
    # 行业本身缺失的样本会被剔除，避免后续标准化混入无行业标签样本。
    group['isnull']=group['Factor'].isnull()
    Industry_group=pd.merge(Industry,group[['TRADE_DT','S_INFO_WINDCODE']],on='S_INFO_WINDCODE',how='left')
    
    Industry_group=Industry_group[(Industry_group.TRADE_DT>=Industry_group.ENTRY_DT)&(Industry_group.TRADE_DT<=Industry_group.REMOVE_DT)]
    group = pd.merge(group, Industry_group[['TRADE_DT','S_INFO_WINDCODE','IND_CODE']], on=['TRADE_DT','S_INFO_WINDCODE'], how='left')
    group.dropna(subset=['IND_CODE'], inplace=True)

    industry_medians = group.groupby('IND_CODE')['Factor'].transform('median')
    group['Factor'] = group['Factor'].fillna(industry_medians)
    group = group[['TRADE_DT', 'S_INFO_WINDCODE','Factor']]
    
    return group


# ================================== 持仓构建与回测统计 ==================================
def Cal_Position(Hold):
    '''
    函数名称：Cal_Position
    函数功能：根据持仓股票池计算下期仓位
    输入参数：Hold：持仓股票池
    输出参数：Position：仓位
    '''
    # 每个调仓日内做等权配置，得到调仓持仓权重。
    Position = Hold[['TRADE_DT', 'S_INFO_WINDCODE']].drop_duplicates().copy()
    if len(Position) == 0:
        Position['POSITION'] = pd.Series(dtype=float)
        return Position
    Position['POSITION'] = 1.0
    Position['POSITION'] = Position.groupby('TRADE_DT')['POSITION'].transform(lambda x: x / x.sum())
    Position = Position.sort_values(by=['TRADE_DT', 'S_INFO_WINDCODE']).reset_index(drop=True)
    return Position


def M2D(group, Position):
    '''
    函数名称：M2D
    函数功能：Back_Testing的内部函数，将月频策略转为日频收益率
            首先需要对资产的日频收益率进行再平衡：根据月频仓位计算每个月内的各资产日频净值，再反推每个月内的各资产日频收益率，以进行仓位再平衡（若只有一个资产，这一步不改变任何值，有多个资产时才起作用）
    输入参数：group：单月的日频各标的收益率；Position：仓位
    输出参数：Result：仓位对应的日频收益率
    '''
    Period = group.TRADE_PERIOD.values[0]
    PositionPeriod = Position[Position.TRADE_DT == Period]
    NetValueTmp = pd.DataFrame(index=group.TRADE_DT.unique(), columns=['NETVALUE'])
    group = pd.merge(group[['TRADE_DT', 'S_INFO_WINDCODE', 'RETURN']], PositionPeriod[['S_INFO_WINDCODE', 'POSITION']],on='S_INFO_WINDCODE', how='right')
    # 若某券在持有期内无行情（如建仓后停牌），收益按 0 处理。
    group['TRADE_DT'] = group['TRADE_DT'].fillna(group.TRADE_DT.unique()[0])
    group['RETURN'] = group['RETURN'].fillna(0)

    if (group['POSITION'] == 0).all():
        NetValueTmp.loc[:] = np.ones([len(NetValueTmp), 1])
    else:
        PositionValue = pd.pivot(group, index='TRADE_DT', columns='S_INFO_WINDCODE', values='POSITION').iloc[0].values
        groupPivot = pd.pivot(group, index='TRADE_DT', columns='S_INFO_WINDCODE', values='RETURN')
        NetValueTmp['NETVALUE'] = (PositionValue * np.cumprod(groupPivot + 1)).sum(axis=1).values

    NetValueTmp['RETURN'] = NetValueTmp['NETVALUE'] / NetValueTmp['NETVALUE'].shift(1) - 1
    # 第一天的组合收益率即第一天所有股票收益率均值
    NetValueTmp['RETURN'].values[0] = group[group.TRADE_DT == min(group.TRADE_DT)]['RETURN'].mean()
    Result = NetValueTmp[['RETURN']]
    return Result


def PerfStatis(Return, frequency):
    '''
    函数名称：PerfStatis
    函数功能：Backtest的内部函数，用于统计绩效指标
    输入参数：Return：收益率；frequency：频率("M","W","D")
    输出参数：区间收益率 区间波动率 年化收益率 年化波动率 夏普比率 最大回撤
    '''
    if frequency == 'M':
        N = 12
    elif frequency == 'W':
        N = 52
    elif frequency == 'D':
        N = 250
    ET = len(Return)  # 有效长度
    # 计算净值
    Return['NETVALUE'] = np.cumprod(Return['RETURN'] + 1)
    FinalNetValue = Return['NETVALUE'].values[-1] if ET > 0 else np.nan
    # 计算区间收益率、区间波动率、年化收益率和年化波动率
    IntervalRet = FinalNetValue - 1
    IntervaStd = np.std(Return['RETURN'].values)
    AnnualRet = (FinalNetValue ** (N / ET) - 1) if ET > 0 and FinalNetValue >= 0 else np.nan
    AnnualStd = np.std(Return['RETURN'].values) * np.sqrt(N)
    # 计算最大回撤
    Maxdrawdown = min(Return['NETVALUE'] / Return['NETVALUE'].cummax() - 1)
    return Return, IntervalRet, IntervaStd, AnnualRet, AnnualStd, Maxdrawdown


def Cal_Excess(StrategyReturn, BenchReturn):
    '''
    函数名称：Cal_Excess
    函数功能：计算超额收益
    输入参数：StrategyReturn：策略收益；BenchReturn：基准收益
    输出参数：ExcessReturn：超额收益
    '''
    ExcessReturn = pd.DataFrame(index=BenchReturn.index, columns=['RETURN'])
    ExcessReturn['NETVALUE'] = StrategyReturn['NETVALUE'] / BenchReturn['NETVALUE']
    ExcessReturn['RETURN'] = ExcessReturn['NETVALUE'] / ExcessReturn['NETVALUE'].shift(1) - 1
    ExcessReturn['RETURN'].values[0] = 0
    return ExcessReturn


def Cal_Winper(Return):
    '''
    函数名称：Cal_Winper
    函数功能：计算胜率
    输入参数：Return：收益率数据
    输出参数：Winper：胜率
    '''
    Denominator = len(Return[Return['RETURN'] != 0])
    Winper = len(Return[Return['RETURN'] > 0]) / Denominator if Denominator > 0 else np.nan
    return Winper


def Cal_TurnOver(Position):
    '''
    函数名称：Cal_TurnOver
    函数功能：计算换手率
    输入参数：Position：仓位数据
    输出参数：TurnOverRate：换手率
    '''
    # 年化单边换手率，供“绩效统计”sheet 使用。
    if len(Position) == 0:
        return 0
    
    Position_Pivot = pd.pivot(Position, index='TRADE_DT', columns='S_INFO_WINDCODE', values='POSITION')
    Position_Pivot = Position_Pivot.sort_index().fillna(0)
    PrevPosition = Position_Pivot.shift(1).fillna(0)
    DeltaPosition = Position_Pivot - PrevPosition
    BuyTurnOver = DeltaPosition.clip(lower=0).sum(axis=1)

    TradeDates = pd.to_datetime(Position_Pivot.index.astype(str), format='%Y%m%d', errors='coerce')
    if len(TradeDates) > 1 and TradeDates.notna().all():
        AvgDays = TradeDates.to_series().diff().dt.days.dropna().mean()
        PeriodsPerYear = 365.0 / AvgDays if AvgDays > 0 else len(BuyTurnOver)
    else:
        PeriodsPerYear = len(BuyTurnOver)

    return BuyTurnOver.mean() * PeriodsPerYear


def Cal_AvgSingleSideTurnOver(Position):
    '''
    函数名称：Cal_AvgSingleSideTurnOver
    函数功能：计算每次调仓的平均单边换手率（百分比）
    输入参数：Position：仓位数据
    输出参数：AvgSingleSideTurnOver：平均单边换手率(%)
    '''
    # 每次调仓的平均单边换手率(%)，供“因子表现”sheet 使用。
    if len(Position) == 0:
        return np.nan

    Position_Pivot = pd.pivot(Position, index='TRADE_DT', columns='S_INFO_WINDCODE', values='POSITION')
    Position_Pivot = Position_Pivot.sort_index().fillna(0)
    PrevPosition = Position_Pivot.shift(1).fillna(0)
    BuyTurnOver = (Position_Pivot - PrevPosition).clip(lower=0).sum(axis=1)

    if len(BuyTurnOver) > 1:
        BuyTurnOver = BuyTurnOver.iloc[1:]

    if len(BuyTurnOver) == 0:
        return np.nan

    return BuyTurnOver.mean()


def Back_Testing(Position, Price, BenchReturn, RiskFreeReturn, cost_rate=0.0015):
    '''
    函数名称：Back_Testing
    函数功能：回测函数
    输入参数：Position：仓位数据；Price：股票价格数据；BenchReturn：基准收益率；RiskFreeReturn：无风险收益率；StartDate：起始日期；EndDate：截止日期
    输出参数：NetValue：净值；Perform：绩效统计
    '''
    # 组合收益已经在日频收益序列中扣除了双边交易成本。
    StrategyReturn = Cal_Daily_Return(Position, Price, cost_rate=cost_rate)
    BenchReturn = BenchReturn.copy()
    RiskFreeReturn = RiskFreeReturn.copy()

    CommonIndex = StrategyReturn.index.intersection(BenchReturn.index).intersection(RiskFreeReturn.index)
    StrategyReturn = StrategyReturn.loc[CommonIndex].copy()
    BenchReturn = BenchReturn.loc[CommonIndex].copy()
    RiskFreeReturn = RiskFreeReturn.loc[CommonIndex].copy()

    TurnOverRate = Cal_TurnOver(Position)
    RiskfreeReturn, RiskfreeIntervalRet, RiskfreeIntervaStd, RiskfreeAnnualRet, RiskfreeAnnualStd, RiskfreeMaxdrawdown = PerfStatis(RiskFreeReturn.copy(), 'D')
    StrategyReturn, StrategyIntervalRet, StrategyIntervaStd, StrategyAnnualRet, StrategyAnnualStd, StrategyMaxdrawdown = PerfStatis(StrategyReturn.copy(), 'D')
    BenchReturn, BenchIntervalRet, BenchIntervaStd, BenchAnnualRet, BenchAnnualStd, BenchMaxdrawdown = PerfStatis(BenchReturn.copy(), 'D')
    ExcessReturn = Cal_Excess(StrategyReturn, BenchReturn)
    ExcessReturn, ExcessIntervalRet, ExcessIntervaStd, ExcessAnnualRet, ExcessAnnualStd, ExcessMaxdrawdown = PerfStatis(ExcessReturn, 'D')
    StrategyWinper = Cal_Winper(ExcessReturn)

    Perform = pd.DataFrame(index=['绩效统计'], columns=['年化收益(%)', '基准年化收益(%)', '超额年化收益(%)', '年化波动(%)', '基准年化波动(%)', '超额年化波动(%)', '最大回撤(%)', '基准最大回撤(%)', '超额最大回撤(%)', '夏普比率', '基准夏普比率', '信息比率', '收益回撤比', '基准收益回撤比', '超额收益回撤比', '胜率(%)', '换手率(年均)'])
    Perform.loc['绩效统计', ['年化收益(%)', '基准年化收益(%)', '超额年化收益(%)', '年化波动(%)', '基准年化波动(%)', '超额年化波动(%)', '最大回撤(%)', '基准最大回撤(%)', '超额最大回撤(%)', '收益回撤比', '基准收益回撤比', '超额收益回撤比', '胜率(%)', '换手率(年均)']] = [
        StrategyAnnualRet, BenchAnnualRet, ExcessAnnualRet, StrategyAnnualStd, BenchAnnualStd, ExcessAnnualStd,
        StrategyMaxdrawdown, BenchMaxdrawdown, ExcessMaxdrawdown,
        -(StrategyAnnualRet / StrategyMaxdrawdown), -(BenchAnnualRet / BenchMaxdrawdown), -(ExcessAnnualRet / ExcessMaxdrawdown),
        StrategyWinper, TurnOverRate
    ]
    Perform.loc['绩效统计', '夏普比率'] = (StrategyAnnualRet - RiskfreeAnnualRet) / StrategyAnnualStd
    Perform.loc['绩效统计', '基准夏普比率'] = (BenchAnnualRet - RiskfreeAnnualRet) / BenchAnnualStd
    Perform.loc['绩效统计', '信息比率'] = ExcessAnnualRet  / ExcessAnnualStd

    NetValue = pd.DataFrame(index=BenchReturn.index, columns=['BenchNetValue', 'StrategyNetValue', 'ExcessNetValue'])
    NetValue['BenchNetValue'] = BenchReturn['NETVALUE']
    NetValue['StrategyNetValue'] = StrategyReturn['NETVALUE']
    NetValue['ExcessNetValue'] = ExcessReturn['NETVALUE']

    return NetValue, Perform


def _reallocate_cash_to_existing_positions(portfolio_value, cash, quotes, cost_rate):
    if cash <= 0 or len(portfolio_value) == 0:
        return max(cash, 0.0)

    tradable_codes = [
        code for code in portfolio_value.keys()
        if code in quotes.index and bool(quotes.at[code, 'CAN_BUY']) and not bool(quotes.at[code, 'FORCED_SELL'])
    ]
    if len(tradable_codes) == 0:
        return max(cash, 0.0)

    available_buy_value = cash / (1 + cost_rate)
    if available_buy_value <= 0:
        return max(cash, 0.0)

    buy_amount = available_buy_value / len(tradable_codes)
    for code in tradable_codes:
        portfolio_value[code] = portfolio_value.get(code, 0.0) + buy_amount

    cash -= buy_amount * (1 + cost_rate) * len(tradable_codes)
    return max(cash, 0.0)


def Cal_Daily_Return(Position, Price, cost_rate=0):
    """按“昨收→VWAP→收盘”的口径展开组合日收益，并在 VWAP 调仓时计入交易成本。"""
    if len(Position) == 0:
        return pd.DataFrame(columns=['RETURN'])

    PositionTmp = Position[['TRADE_DT', 'S_INFO_WINDCODE', 'POSITION']].copy()
    PositionTmp['TRADE_DT'] = PositionTmp['TRADE_DT'].astype(str)
    RebalanceDates = np.array(sorted(PositionTmp['TRADE_DT'].unique()))

    PriceTmp = Price.copy()
    PriceTmp['TRADE_DT'] = PriceTmp['TRADE_DT'].astype(str)
    PriceTmp = PriceTmp[PriceTmp['TRADE_DT'] >= RebalanceDates[0]].copy()
    PriceTmp = PriceTmp.sort_values(by=['TRADE_DT', 'S_INFO_WINDCODE'])
    if len(PriceTmp) == 0:
        return pd.DataFrame(columns=['RETURN'])

    price_by_date = {
        trade_dt: group.set_index('S_INFO_WINDCODE')
        for trade_dt, group in PriceTmp.groupby('TRADE_DT', sort=True)
    }
    target_by_date = {
        trade_dt: group.set_index('S_INFO_WINDCODE')['POSITION']
        for trade_dt, group in PositionTmp.groupby('TRADE_DT', sort=True)
    }

    portfolio_value = {}
    cash = 1.0
    total_value = 1.0
    result = []

    for trade_dt in sorted(price_by_date.keys()):
        quotes = price_by_date[trade_dt]
        prev_total_value = total_value
        forced_sell_codes = [
            code for code in portfolio_value.keys()
            if code in quotes.index and bool(quotes.at[code, 'FORCED_SELL'])
        ]
        needs_midday_trade = (trade_dt in target_by_date) or (len(forced_sell_codes) > 0)

        if needs_midday_trade:
            for code in list(portfolio_value.keys()):
                if code in quotes.index:
                    portfolio_value[code] *= (1 + quotes.at[code, 'PRE_TO_VWAP_RETURN'])

            total_value = cash + sum(portfolio_value.values())

            if len(forced_sell_codes) > 0:
                for code in forced_sell_codes:
                    sell_amount = portfolio_value.pop(code, 0.0)
                    if sell_amount > 0:
                        cash += sell_amount * (1 - cost_rate)

            total_value = cash + sum(portfolio_value.values())

            if trade_dt in target_by_date:
                target_weights = target_by_date[trade_dt]
                target_value = {code: total_value * weight for code, weight in target_weights.items()}

                for code in list(portfolio_value.keys()):
                    current_value = portfolio_value.get(code, 0.0)
                    desired_value = target_value.get(code, 0.0)
                    if current_value > desired_value + 1e-12:
                        can_sell = (code in quotes.index) and bool(quotes.at[code, 'CAN_SELL'])
                        if can_sell:
                            sell_amount = current_value - desired_value
                            portfolio_value[code] = desired_value
                            cash += sell_amount * (1 - cost_rate)
                    if portfolio_value.get(code, 0.0) <= 1e-12:
                        portfolio_value.pop(code, None)

                buy_needs = []
                total_buy_need = 0.0
                for code, desired_value in target_value.items():
                    current_value = portfolio_value.get(code, 0.0)
                    if desired_value > current_value + 1e-12:
                        can_buy = (code in quotes.index) and bool(quotes.at[code, 'CAN_BUY'])
                        if can_buy:
                            need = desired_value - current_value
                            buy_needs.append((code, need))
                            total_buy_need += need

                if total_buy_need > 0 and cash > 0:
                    available_buy_value = cash / (1 + cost_rate)
                    buy_scale = min(1.0, available_buy_value / total_buy_need)
                    for code, need in buy_needs:
                        buy_amount = need * buy_scale
                        portfolio_value[code] = portfolio_value.get(code, 0.0) + buy_amount
                        cash -= buy_amount * (1 + cost_rate)
            elif len(forced_sell_codes) > 0:
                cash = _reallocate_cash_to_existing_positions(portfolio_value, cash, quotes, cost_rate)

            for code in list(portfolio_value.keys()):
                if code in quotes.index:
                    portfolio_value[code] *= (1 + quotes.at[code, 'VWAP_TO_CLOSE_RETURN'])
        else:
            for code in list(portfolio_value.keys()):
                if code in quotes.index:
                    portfolio_value[code] *= (1 + quotes.at[code, 'RETURN'])

        cash = max(cash, 0.0)
        total_value = cash + sum(portfolio_value.values())
        daily_return = total_value / prev_total_value - 1 if prev_total_value != 0 else 0
        result.append((trade_dt, daily_return))

    StrategyReturn = pd.DataFrame(result, columns=['TRADE_DT', 'RETURN']).set_index('TRADE_DT')
    return StrategyReturn[['RETURN']]


# ================================== 因子表现统计 ==================================
def Build_Ranked_Factor(Factor):
    """按标准化后的因子值从高到低排序，供 TOP/BOTTOM 组合与 RankIC 使用。"""
    RankedFactor = Factor.copy()
    RankedFactor['RANK'] = RankedFactor.groupby('TRADE_DT')['Factor_Mad_Stand'].rank(
        ascending=False,
        method='min',
        na_option='keep'
    )
    RankedFactor['COUNT'] = RankedFactor.groupby('TRADE_DT')['S_INFO_WINDCODE'].transform('count')
    return RankedFactor


def _build_rank_slice_hold(RankedFactor, slice_name):
    def _select(group):
        count = int(group['COUNT'].iloc[0]) if len(group) > 0 else 0
        if count == 0:
            return group.iloc[0:0]

        if slice_name == 'TOP30':
            upper = min(30, count)
            mask = group['RANK'] <= upper
        elif slice_name == 'TOP31_60':
            lower, upper = 31, min(60, count)
            mask = (group['RANK'] >= lower) & (group['RANK'] <= upper)
        elif slice_name == 'BOTTOM30':
            lower = max(count - 29, 1)
            mask = group['RANK'] >= lower
        elif slice_name == 'BOTTOM60_31':
            lower = max(count - 59, 1)
            upper = max(count - 30, 0)
            mask = (group['RANK'] >= lower) & (group['RANK'] <= upper)
        else:
            raise ValueError(f'Unknown slice name: {slice_name}')

        return group.loc[mask, ['TRADE_DT', 'S_INFO_WINDCODE']]

    Hold = RankedFactor.groupby('TRADE_DT', group_keys=False).apply(_select).reset_index(drop=True)
    return Hold.drop_duplicates().reset_index(drop=True)


def Cal_ForwardBondReturn(Factor, Price):
    if len(Factor) == 0:
        return pd.DataFrame(columns=['TRADE_DT', 'S_INFO_WINDCODE', 'FWD_RETURN'])

    PeriodList = np.array(sorted(Factor['TRADE_DT'].astype(str).unique()))
    if len(PeriodList) < 2:
        return pd.DataFrame(columns=['TRADE_DT', 'S_INFO_WINDCODE', 'FWD_RETURN'])

    PriceTmp = Price[['TRADE_DT', 'S_INFO_WINDCODE', 'RETURN', 'PRE_TO_VWAP_RETURN', 'VWAP_TO_CLOSE_RETURN']].copy()
    PriceTmp['TRADE_DT'] = PriceTmp['TRADE_DT'].astype(str)
    close_return = PriceTmp.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='RETURN').sort_index()
    pre_to_vwap = PriceTmp.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='PRE_TO_VWAP_RETURN').sort_index()
    vwap_to_close = PriceTmp.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='VWAP_TO_CLOSE_RETURN').sort_index()

    forward_rows = []
    factor_dates = Factor['TRADE_DT'].astype(str)
    for idx in range(len(PeriodList) - 1):
        start_dt = PeriodList[idx]
        end_dt = PeriodList[idx + 1]
        codes = Factor.loc[factor_dates == start_dt, 'S_INFO_WINDCODE'].drop_duplicates().tolist()
        if len(codes) == 0:
            continue

        start_leg = 1 + vwap_to_close.reindex(index=[start_dt], columns=codes).fillna(0).iloc[0]
        end_leg = 1 + pre_to_vwap.reindex(index=[end_dt], columns=codes).fillna(0).iloc[0]
        middle_dates = close_return.index[(close_return.index > start_dt) & (close_return.index < end_dt)]
        if len(middle_dates) > 0:
            middle_leg = (1 + close_return.reindex(index=middle_dates, columns=codes).fillna(0)).prod(axis=0)
        else:
            middle_leg = pd.Series(1.0, index=codes)

        fwd_return = start_leg * middle_leg * end_leg - 1
        forward_rows.extend(
            [(start_dt, code, fwd_return.loc[code]) for code in codes]
        )

    return pd.DataFrame(forward_rows, columns=['TRADE_DT', 'S_INFO_WINDCODE', 'FWD_RETURN'])


def Cal_RankIC(Factor, Price):
    """计算调仓截面的 Spearman RankIC。"""
    ForwardReturn = Cal_ForwardBondReturn(Factor, Price)
    ICData = pd.merge(
        Factor[['TRADE_DT', 'S_INFO_WINDCODE', 'Factor_Mad_Stand']],
        ForwardReturn,
        on=['TRADE_DT', 'S_INFO_WINDCODE'],
        how='left'
    )

    def _calc(group):
        group = group[['Factor_Mad_Stand', 'FWD_RETURN']].dropna()
        if len(group) < 2:
            return np.nan
        if group['Factor_Mad_Stand'].nunique() < 2 or group['FWD_RETURN'].nunique() < 2:
            return np.nan
        return group['Factor_Mad_Stand'].corr(group['FWD_RETURN'], method='spearman')

    RankIC = ICData.groupby('TRADE_DT').apply(_calc)
    RankIC.name = 'RankIC'
    return RankIC.dropna()


def Cal_LongShort_Daily_Return(LongPosition, ShortPosition, Price, cost_rate=0.0015):
    LongReturn = Cal_Daily_Return(LongPosition, Price, cost_rate=cost_rate)
    ShortReturn = Cal_Daily_Return(ShortPosition, Price, cost_rate=cost_rate)

    if len(LongReturn) == 0 and len(ShortReturn) == 0:
        return pd.DataFrame(columns=['RETURN'])

    CommonIndex = LongReturn.index.union(ShortReturn.index)
    Result = pd.DataFrame(index=CommonIndex, columns=['RETURN'], dtype=float)
    Result['RETURN'] = LongReturn.reindex(CommonIndex, fill_value=0)['RETURN'] - ShortReturn.reindex(CommonIndex, fill_value=0)['RETURN']
    Result.sort_index(inplace=True)
    return Result


def _safe_backtest_metrics(Position, Price, BenchReturn, RiskFreeReturn, cost_rate):
    if len(Position) == 0:
        return np.nan, np.nan, np.nan

    _, Perform = Back_Testing(Position, Price, BenchReturn, RiskFreeReturn, cost_rate=cost_rate)
    if len(Perform) == 0:
        return np.nan, np.nan, np.nan

    PerformRow = Perform.iloc[0]
    return (
        PerformRow.get('超额年化收益(%)', np.nan),
        PerformRow.get('信息比率', np.nan),
        Cal_AvgSingleSideTurnOver(Position)
    )


def _safe_annual_return(Return):
    if len(Return) == 0:
        return np.nan
    _, _, _, AnnualRet, _, _ = PerfStatis(Return.copy(), 'D')
    return AnnualRet


def Cal_FactorPerformance(Factor, Price, BenchReturn, RiskFreeReturn, cost_rate=0.0015):
    """
    生成研报风格的“因子表现”表。

    说明：
    1. TOP30 / Bottom30 始终保持字面含义，不因方向而互换。
    2. 多头超额收益率 / 多头信息比率 会根据方向自动选择更优的一侧。
    """
    RankedFactor = Build_Ranked_Factor(Factor)
    RankICSeries = Cal_RankIC(RankedFactor, Price)

    RankICMean = RankICSeries.mean() if len(RankICSeries) > 0 else np.nan
    RankICStd = RankICSeries.std(ddof=1) if len(RankICSeries) > 0 else np.nan
    ICIR = RankICMean / RankICStd if pd.notna(RankICStd) and RankICStd != 0 else np.nan

    FactorGroup = Cal_Stratify(Factor, 3)
    HighGroupHold = FactorGroup[FactorGroup['GROUP'] == '3'][['TRADE_DT', 'S_INFO_WINDCODE']].reset_index(drop=True)
    LowGroupHold = FactorGroup[FactorGroup['GROUP'] == '1'][['TRADE_DT', 'S_INFO_WINDCODE']].reset_index(drop=True)
    HighGroupPosition = Cal_Position(HighGroupHold)
    LowGroupPosition = Cal_Position(LowGroupHold)

    HighGroupExcessRet, HighGroupInfoRatio, HighGroupTurnOver = _safe_backtest_metrics(
        HighGroupPosition,
        Price,
        BenchReturn,
        RiskFreeReturn,
        cost_rate
    )
    LowGroupExcessRet, LowGroupInfoRatio, LowGroupTurnOver = _safe_backtest_metrics(
        LowGroupPosition,
        Price,
        BenchReturn,
        RiskFreeReturn,
        cost_rate
    )

    Top30Hold = _build_rank_slice_hold(RankedFactor, 'TOP30')
    Top31_60Hold = _build_rank_slice_hold(RankedFactor, 'TOP31_60')
    Bottom60_31Hold = _build_rank_slice_hold(RankedFactor, 'BOTTOM60_31')
    Bottom30Hold = _build_rank_slice_hold(RankedFactor, 'BOTTOM30')

    Top30Position = Cal_Position(Top30Hold)
    Top31_60Position = Cal_Position(Top31_60Hold)
    Bottom60_31Position = Cal_Position(Bottom60_31Hold)
    Bottom30Position = Cal_Position(Bottom30Hold)

    Top30ExcessRet, _, _ = _safe_backtest_metrics(Top30Position, Price, BenchReturn, RiskFreeReturn, cost_rate)
    Top31_60ExcessRet, _, _ = _safe_backtest_metrics(Top31_60Position, Price, BenchReturn, RiskFreeReturn, cost_rate)
    Bottom60_31ExcessRet, _, _ = _safe_backtest_metrics(Bottom60_31Position, Price, BenchReturn, RiskFreeReturn, cost_rate)
    Bottom30ExcessRet, _, _ = _safe_backtest_metrics(Bottom30Position, Price, BenchReturn, RiskFreeReturn, cost_rate)

    if pd.notna(HighGroupExcessRet) and pd.notna(LowGroupExcessRet):
        Direction = 1 if HighGroupExcessRet >= LowGroupExcessRet else -1
    elif pd.notna(Top30ExcessRet) and pd.notna(Bottom30ExcessRet):
        Direction = 1 if Top30ExcessRet >= Bottom30ExcessRet else -1
    elif pd.notna(RankICMean) and RankICMean != 0:
        Direction = 1 if RankICMean > 0 else -1
    else:
        Direction = np.nan

    if len(RankICSeries) > 0 and pd.notna(Direction):
        RankICWinRate = (RankICSeries * Direction > 0).mean()
    else:
        RankICWinRate = np.nan

    if Direction == 1:
        LongPosition = HighGroupPosition
        ShortPosition = LowGroupPosition
        LongExcessRet = HighGroupExcessRet
        LongInfoRatio = HighGroupInfoRatio
        LongTurnOver = HighGroupTurnOver
    elif Direction == -1:
        LongPosition = LowGroupPosition
        ShortPosition = HighGroupPosition
        LongExcessRet = LowGroupExcessRet
        LongInfoRatio = LowGroupInfoRatio
        LongTurnOver = LowGroupTurnOver
    else:
        LongPosition = pd.DataFrame(columns=['TRADE_DT', 'S_INFO_WINDCODE', 'POSITION'])
        ShortPosition = pd.DataFrame(columns=['TRADE_DT', 'S_INFO_WINDCODE', 'POSITION'])
        LongExcessRet = np.nan
        LongInfoRatio = np.nan
        LongTurnOver = np.nan

    LongShortReturn = Cal_LongShort_Daily_Return(LongPosition, ShortPosition, Price, cost_rate=cost_rate)
    LongShortAnnualRet = _safe_annual_return(LongShortReturn)

    Result = pd.DataFrame(
        {
            '方向': [Direction],
            'RankIC均值': [RankICMean],
            'IC_IR(未年化)': [ICIR],
            'RankIC胜率': [RankICWinRate],
            '多头超额收益率': [LongExcessRet],
            '多头信息比率': [LongInfoRatio],
            '多头平均换手率（单边）': [LongTurnOver],
            '多空年化收益率': [LongShortAnnualRet],
            'TOP30组合超额收益率': [Top30ExcessRet],
            'TOP31-60组合超额收益率': [Top31_60ExcessRet],
            'Bottom60-31组合超额收益率': [Bottom60_31ExcessRet],
            'Bottom30组合超额收益率': [Bottom30ExcessRet],
        },
        index=['因子表现']
    )
    return Result, RankICSeries




START_DATE = "2024-01-11"  # start date for historical data
RSI_TIME_WINDOW = 7  # number of days

import requests
import pandas as pd
import warnings
import datetime as dt
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas_datareader.data as web
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

## URLS and names
urls = ["https://www.cryptodatadownload.com/cdd/Bitfinex_EOSUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_BTCUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_ETHUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_LTCUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_BATUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_OMGUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_DAIUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_ETCUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_NEOUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_TRXUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_XLMUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_XMRUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_XVGUSD_d.csv",
        ]
crypto_names = ["EOS Coin (EOS)",
                "Bitcoin (BTC)",
                "Ethereum (ETH)",
                "Litecoin (LTC)",
                "Basic Attention Token (BAT)",
                "OmiseGO (OMG)",
                "Dai (DAI)",
                "Ethereum Classic (ETC)",
                "Neo (NEO)",
                "TRON (TRX)",
                "Stellar (XLM)",
                "Monero (XMR)",
                "Verge (XVG)"
                ]


## Data download and loading
def df_loader(urls, start_date="2021-01-01"):
    filenames = []
    all_df = pd.DataFrame()
    for idx, url in enumerate(urls):
        req = requests.get(url, verify=False)
        url_content = req.content
        filename = url[48:]
        csv_file = open(filename, 'wb')
        csv_file.write(url_content)
        csv_file.close()
        filename = filename[:-9]
        filenames.append(filename)
    for file in filenames:
        df = pd.read_csv(file + "USD_d.csv", header=1, parse_dates=["date"])
        df = df[df["date"] > start_date]
        df.index = df.date
        df.drop(labels=[df.columns[0], df.columns[1], df.columns[8]], axis=1, inplace=True)
        all_df = pd.concat([all_df, df], ignore_index=False)

    return all_df, filenames


def computeRSI(data, time_window):
    diff = data.diff(1).dropna()
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[diff < 0]
    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def computeMA(data, window):
    return data.rolling(window=window, min_periods=1).mean()


all_df, filenames = df_loader(urls, start_date=START_DATE)

crypto_df = []
latest_common_date = None
earliest_common_date = None
for file in filenames:
    symbol = file + "/USD"
    temp_df = pd.DataFrame(all_df[all_df["symbol"] == symbol])

    # 检查索引类型
    print(f"\nChecking index for {symbol}:")
    print(f"Index dtype: {temp_df.index.dtype}")

    # 检查是否所有索引都是日期格式
    if not isinstance(temp_df.index, pd.DatetimeIndex):
        print(f"Warning: Index for {symbol} is not DatetimeIndex")
        print("Sample of problematic indices:")
        print(temp_df.index[:5].tolist())  # 打印前5个索引值

    if latest_common_date is None and earliest_common_date is None:
        latest_common_date = temp_df.index.max()
        earliest_common_date = temp_df.index.min()
        print(f"Latest date: {temp_df.index.max()}")
        print(f"Earliest date: {temp_df.index.min()}")
    else:
        print(f"Latest date: {temp_df.index.max()}")
        print(f"Earliest date: {temp_df.index.min()}")
        latest_common_date = min(temp_df.index.max(), latest_common_date)
        earliest_common_date = max(temp_df.index.min(), earliest_common_date)
for file in filenames:
    symbol = file + "/USD"
    temp_df = pd.DataFrame(all_df[all_df["symbol"] == symbol])
    temp_df = temp_df[(temp_df.index <= latest_common_date) & (temp_df.index >= earliest_common_date)]

    temp_df.drop(columns=["symbol"], inplace=True)

    # 对数值列进行插值处理
    numeric_columns = ['open', 'high', 'low', 'close', 'Volume USD']
    temp_df[numeric_columns] = temp_df[numeric_columns].interpolate(method='time')

    print(temp_df[numeric_columns].isnull().sum())

    # 如果某列缺失值过多（比如超过20%），可能需要特殊处理
    missing_threshold = 0.2
    missing_ratio = temp_df[numeric_columns].isnull().sum() / len(temp_df)
    if (missing_ratio > missing_threshold).any():
        print(f"Warning: High missing ratio in {file}")

    temp_df["close_rsi"] = computeRSI(temp_df['close'], time_window=RSI_TIME_WINDOW)
    temp_df["ma_7"] = computeMA(temp_df['close'], window=7)
    temp_df["ma_25"] = computeMA(temp_df['close'], window=25)
    temp_df["ma_99"] = computeMA(temp_df['close'], window=99)
    temp_df["high_rsi"] = 30
    temp_df["low_rsi"] = 70
    exec('%s = temp_df.copy()' % file.lower())
    crypto_df.append(temp_df)
## plot
fig = make_subplots(
    rows=5,
    cols=4,
    shared_xaxes=True,
    specs=[
        [{"rowspan": 2, "colspan": 4}, None, None, None],  # 蜡烛图行
        [None, None, None, None],  # 被蜡烛图占用的行
        [{"rowspan": 1, "colspan": 2}, None, {"colspan": 2}, None],  # RSI和成交量行
        [{"rowspan": 1, "colspan": 4, "type": "domain"}, None, None, None],  # Treemap行
        [{"rowspan": 1, "colspan": 1, "type": "domain"}, {"rowspan": 1, "colspan": 1, "type": "domain"}, {"rowspan": 1, "colspan": 2}, None],  # 饼图和热力图行
    ],
    row_heights=[0.25, 0.15, 0.15, 0.2, 0.25]  # 总和为1
)
date_buttons = [
    {'step': "all", 'label': "All time"},  # 显示所有时间范围的数据
    {'count': 1, 'step': "year", 'stepmode': "backward", 'label': "Last Year"},  # 显示最近一年的数据
    {'count': 1, 'step': "year", 'stepmode': "todate", 'label': "Current Year"},  # 显示今年至今的数据
    {'count': 1, 'step': "month", 'stepmode': "backward", 'label': "Last 2 Months"},  # 显示最近一个月的数据
    {'count': 1, 'step': "month", 'stepmode': "todate", 'label': "Current Month"},  # 显示本月至今的数据
    {'count': 7, 'step': "day", 'stepmode': "todate", 'label': "Current Week"},  # 显示本周至今的数据（7天）
    {'count': 4, 'step': "day", 'stepmode': "backward", 'label': "Last 4 days"},  # 显示最近4天的数据
    {'count': 1, 'step': "day", 'stepmode': "backward", 'label': "Today"},  # 显示今天的数据
]
buttons = []
i = 0
j = 0
COUNT = 8
# 蜡烛
# 成交量柱状图
# 价格线图
# 最低价线
# 最高价线
# RSI指标线
# RSI低线
# RSI高线

vis = [False] * len(crypto_names) * COUNT
vis.append(True)
vis.append(True)
vis.append(True)
vis.append(True)
for df in crypto_df:
    for k in range(COUNT):
        vis[j + k] = True
    buttons.append({'label': crypto_names[i],
                    'method': 'update',
                    'args': [{'visible': vis},
                             {'title': crypto_names[i] + ' Charts and Indicators'}
                             ]}
                   )
    i += 1
    j += COUNT
    vis = [False] * len(crypto_names) * COUNT
    vis.extend([True] * 4)
for df in crypto_df:
    print(df.index)
    fig.add_trace(
        go.Candlestick(x=df.index,
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'],
                       name='current_stock_price',
                       showlegend=True,
                       increasing_line_color='#26A69A',  # 上涨时的颜色（绿色）
                       decreasing_line_color='#EF5350',  # 下跌时的颜色（红色）
                       increasing_fillcolor='#26A69A',  # 上涨时的填充颜色
                       decreasing_fillcolor='#EF5350'  # 下跌时的填充颜色
                       ),
        row=1,
        col=1)
    fig.add_trace(
        go.Bar(x=df.index,
               y=df["Volume USD"],
               name='Volume USD',
               showlegend=True,
               marker_color='aqua'),
        row=3,
        col=1)

    # fig.add_trace(
    #     go.Scatter(x=df.index, y=df['close'],
    #                mode='lines',
    #                name='close_Price',
    #               showlegend =True,
    #                line=dict(color="red", width=4)),
    #     row=1,
    #     col=2)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['close_rsi'],
                   mode='lines',
                   name='RSI',
                   showlegend=True,
                   line=dict(color="aquamarine", width=4)),
        row=3,
        col=3)
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['low_rsi'],
                   fill='tonexty',
                   mode='lines',
                   name='RSI_low',
                   showlegend=False,
                   line=dict(width=2, color='aqua', dash='dash')),
        row=3,
        col=3)
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['high_rsi'],
                   fill='tonexty',
                   mode='lines',
                   name='RSI_high',
                   showlegend=False,
                   line=dict(width=2, color='aqua', dash='dash')),
        row=3,
        col=3)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ma_7'],
                   mode='lines',
                   name='MA7',
                   line=dict(color="orange", width=1)),
        row=1, col=1)

    fig.add_trace(
        go.Scatter(x=df.index, y=df['ma_25'],
                   mode='lines',
                   name='MA25',
                   line=dict(color="purple", width=1)),
        row=1, col=1)

    fig.add_trace(
        go.Scatter(x=df.index, y=df['ma_99'],
                   mode='lines',
                   name='MA99',
                   line=dict(color="cyan", width=1)),
        row=1, col=1)

fig.update_xaxes(
    tickfont=dict(size=15, family='monospace', color='#B8B8B8'),
    tickmode='array',
    ticklen=6,
    showline=False,
    showgrid=True,
    gridcolor='#595959',
    ticks='outside')
fig.update_layout(
    spikedistance=100,
    xaxis_rangeslider_visible=False,
    hoverdistance=1000)
fig.update_xaxes(
    showspikes=True,
    spikesnap="cursor",
    spikemode="across"
)
fig.update_yaxes(
    showspikes=True,
    spikesnap='cursor',
    spikemode="across"
)
fig.update_yaxes(
    tickfont=dict(size=15, family='monospace', color='#B8B8B8'),
    tickmode='array',
    showline=False,
    ticksuffix='$',
    showgrid=True,
    gridcolor='#595959',
    ticks='outside')
fig.update_layout(
    autosize=True,
    font_family='monospace',
    xaxis=dict(
        rangeselector=dict(
            buttons=date_buttons,
            y=1.1,
            x=0,
            font=dict(color='#FFFFFF', size=12),
            bgcolor='#000000'
        ),
    ),
    xaxis2=dict(),
    xaxis3=dict(),
    updatemenus=[dict(type='dropdown',
                      x=1,
                      y=1.1,
                      showactive=True,
                      active=2,
                      buttons=buttons,
                      font=dict(color='#FFFFFF')
                      )],
    title=dict(text='<b>Cryptocurrencies  Dashboard<b>',
               font=dict(color='#FFFFFF', size=22),
               x=0.50),
    font=dict(color="blue"),
    annotations=[
        dict(text="<b>Volume Traded<b>",
             font=dict(size=20, color="#ffffff"),
             showarrow=False,
             x=0.14,
             y=-0.53,
             xref='paper', yref="paper",
             align="left"),
        dict(text="<b>Relative Strength Index (RSI)<b>",
             font=dict(size=20, color="#ffffff"),
             showarrow=False,
             x=0.94,
             y=-0.53,
             xref='paper', yref="paper",
             align="left")
    ],
    template="plotly_dark"
    # Options include "plotly", "ggplot2", "seaborn", "simple_white", "plotly_white", "presentation", "xgridoff", "ygridoff", "gridon", "gridoff", "none", and "plotly_dark"
)
for i in range(0, 13 * COUNT):
    fig.data[i].visible = False
for i in range(COUNT):
    fig.data[i].visible = True
fig.layout["xaxis"]["rangeslider"]["visible"] = False
fig.layout["xaxis2"]["rangeslider"]["visible"] = False
fig.layout["xaxis3"]["rangeslider"]["visible"] = False
fig.layout['xaxis']['rangeselector']['visible'] = True
fig.update_xaxes(matches='x')
# fig.layout["xaxis4"]["rangeslider"]["visible"] = True
# fig.layout["xaxis3"]["rangeslider"]["borderwidth"] = 4
# fig.layout["xaxis4"]["rangeslider"]["borderwidth"] = 4
# fig.layout["xaxis3"]["rangeslider"]["bordercolor"] = "aqua"
# fig.layout["xaxis4"]["rangeslider"]["bordercolor"] = "aqua"
# fig.layout["yaxis4"]["ticksuffix"] = ""
# fig.layout["yaxis4"]["range"] = [10,100]
# fig.show()


# 创建网格布局的方块图
def create_market_overview(crypto_df, filenames):
    # Prepare data for treemap
    values = []  # Market values
    changes = []  # Price changes
    labels = []  # Crypto symbols
    colors = []  # Colors based on price change

    for idx, df in enumerate(crypto_df):
        if idx >= 8:  # Still limiting to 4 cryptocurrencies
            break
        if idx == 1 or idx == 2:
            continue
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100

        # Get trading volume as market value
        market_value = df['Volume USD'].iloc[-1]

        symbol = filenames[idx].split('/')[-1]

        values.append(market_value)
        changes.append(price_change)
        labels.append(f"{symbol}<br>${current_price:.2f}<br>{price_change:+.2f}%")
        colors.append('#3C6E3C' if price_change > 0 else '#4E3636')  # 使用深绿色和深红色
        print(price_change)
        print(market_value)
        print(labels)

    # Add Treemap
    fig.add_trace(
        go.Treemap(
            labels=labels,
            parents=[''] * len(labels),
            values=values,
            textinfo='label',
            marker=dict(
                colors=colors,
                line=dict(width=2, color='#1a1a1a')  # 深色边框
            ),
            hoverinfo='label',
            textfont=dict(size=14, color='white'),
        ),
        row=4,
        col=1
    )
    # Pie
    fig.add_trace(
        go.Pie(
        labels=labels,
        values=values,
        hoverinfo='label',
        textfont=dict(size=14, color='white'),
        hole=0.5,  # Adjust hole size for a donut chart effect
        marker=dict(line=dict(color='#000000', width=2)),  # Add border to pie slices
        legend="legend2"
    ),
    row=5, col=1
)
    fig.add_trace(
    go.Pie(
        labels=labels,
        values=values,
        hoverinfo='label',
        textfont=dict(size=14, color='white'),
        hole=0.5,  # Adjust hole size for a donut chart effect
        marker=dict(line=dict(color='#000000', width=2)),  # Add border to pie slices
        legend = "legend2"
    ),
    row=5, col=2
)
    fig.update_layout(
        annotations=[
            *fig.layout.annotations,  # 保留现有的 annotations
            # 第一个饼图的中心文字
            dict(
                text='Volume<br>Distribution',
                x=0.2125/2-0.025,  # 中心位置的x坐标
                y=0.152/2,   # 中心位置的y坐标
                showarrow=False,
                font=dict(size=12, color='white'),
                xref='paper',
                yref='paper'
            ),
            # 第二个饼图的中心文字
            dict(
                text='Market<br>Share',
                x=(0.2625+0.475)/2,  # 中心位置的x坐标
                y=0.152/2,   # 中心位置的y坐标
                showarrow=False,
                font=dict(size=12, color='white'),
                xref='paper',
                yref='paper'
            )
        ],
        legend2=dict(
            font=dict(size=8, color='white'),
            x=-0.02,  # 默认全局图例在右侧
            y=0.01,
            xanchor="left",
            yanchor="middle"
        ),
        legend=dict(
            font=dict(size=8, color='white'),
            x=1.02,  # 默认全局图例在右侧
            y=0.99,
            xanchor="left",
            yanchor="top"
        )
    )
    fig.update_layout({
        'xaxis5': {'showgrid': False},  # 第5行第2列的x轴
        'yaxis5': {'showgrid': False}  # 第5行第2列的y轴
    })
    # Add heatmap with adjusted colorbar height
    correlation_matrix = all_df.pivot_table(index='date', columns='symbol', values='close').corr()
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='Viridis',
            name='Correlation Heatmap',
            hoverongaps=False,
            hovertemplate='<b>x: %{x}</b><br>y: %{y}<br>Correlate: %{z}<br><extra></extra>',
            colorbar=dict(
                lenmode='fraction',
                len=0.3,  # Set this value to match the height of the heatmap
                x=1.05,  # Adjust the x position of the colorbar
                y=0.1,  # Adjust the y position of the colorbar
                xanchor='left',
                yanchor='middle'
            )
        ),
        row=5, col=3
    )
fig.layout['yaxis4']['ticksuffix'] = ''
fig.layout['yaxis4']['ticks'] = 'outside'
# 在fig.show()之前调用这个函数
create_market_overview(crypto_df, filenames)
print(fig.layout)
print(fig.data)
fig.data[104].visible = True
fig.show()
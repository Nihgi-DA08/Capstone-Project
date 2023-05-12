import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
import ipywidgets as widgets
import sklearn.metrics as met
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import Dropdown
from ipywidgets import interactive
from pandas.api.types import CategoricalDtype
from sklearn.linear_model import LinearRegression

COL_NAME_HTL = 'Hotel'
COL_NAME_LEAD_TIME = 'Lead Time'
COL_NAME_ARR_DATE_D = 'Arrival Date Day'
COL_NAME_ARR_DATE_M = 'Arrival Date Month'
COL_NAME_ARR_DATE_Y = 'Arrival Date Year'
COL_NAME_ARR_DATE = 'Arrival Date'
COL_NAME_WKDAY = 'Weekday'
COL_NAME_ADT = 'Adults'
COL_NAME_CHLDN = 'Children'
COL_NAME_CNTRY = 'Country'
COL_NAME_MKT_SEG = 'Market Segment'
COL_NAME_AGT = 'Agent'
COL_NAME_CUST_TYPE = 'Customer Type'
COL_NAME_AVG_DLY_RATE = 'Average Daily Rate'
COL_NAME_CNTRY_NAME = 'Country Name'

month_ord = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
wkday_ord = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

df = pd.read_csv('https://media.githubusercontent.com/media/Nihgi-DA08/Capstone-Project/main/data_valid.csv')
cntry_codes_df = pd.read_excel('country_codes_list.xlsx')
merged_df = pd.merge(df,
                     cntry_codes_df,
                     on=COL_NAME_CNTRY,
                     how='left')

# Create dropdown list
def crt_ddl(col_name):
    list = df[col_name].unique().tolist()
    list.append('All')
    return list

df[COL_NAME_ARR_DATE_Y] = df[COL_NAME_ARR_DATE_Y].astype(str)

merged_df[COL_NAME_ARR_DATE_Y] = merged_df[COL_NAME_ARR_DATE_Y].astype(str)

# Univariate mod dataframe
def uv_mod_df(year, mkt_seg, cust_type):
    fltr_df = df if year == 'All' else df[df[COL_NAME_ARR_DATE_Y] == year]
    fltr_df = fltr_df if mkt_seg == 'All' else fltr_df[fltr_df[COL_NAME_MKT_SEG] == mkt_seg]
    return fltr_df if cust_type == 'All' else fltr_df[fltr_df[COL_NAME_CUST_TYPE] == cust_type]

# Univariate Hotel
def uv_htl(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_HTL].value_counts()
    return go.Pie(labels=cnts.index,
                  values=cnts.values,
                  hole=.5,
                  name='Hotel')

# Univariate Lead Time
def uv_lead_time(year, mkt_seg, cust_type):
    return go.Histogram(x=uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_LEAD_TIME],
                        nbinsx=30,
                        name='Day')

# Univariate Arrival Date Day
def uv_arr_date_day(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_ARR_DATE_D].value_counts().sort_index()
    return go.Bar(x=cnts.index,
                  y=cnts.values,
                  name='Day')

# Univariate Weekday
def uv_wkday(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_WKDAY].astype(CategoricalDtype(categories=wkday_ord,
                                                                                       ordered=True)).value_counts().sort_index()
    return go.Bar(x=cnts.index.str[:3],
                  y=cnts.values,
                  hovertext=wkday_ord,
                  name='Weekday')

# Univariate Arrival Date Month
def uv_arr_date_month(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_ARR_DATE_M].astype(CategoricalDtype(categories=month_ord,
                                                                                            ordered=True)).value_counts().sort_index()
    return go.Bar(x=cnts.index.str[:3],
                  y=cnts.values,
                  hovertext=month_ord,
                  name='Month')

# Univariate Arrival Date Year
def uv_arr_date_year(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_ARR_DATE_Y].value_counts().sort_index()
    return go.Pie(labels=cnts.index,
                  values=cnts.values,
                  hole=.5,
                  name='Year')

# Univariate Country
def uv_cntry(year, mkt_seg, cust_type):
    new_df = pd.merge(cntry_codes_df,
                      uv_mod_df(year, mkt_seg, cust_type).groupby([COL_NAME_CNTRY]).size().reset_index(name='Count'),
                      how='inner')
    return go.Choropleth(locations=new_df[COL_NAME_CNTRY],
                         z=new_df['Count'],
                         hovertext=new_df[COL_NAME_CNTRY_NAME],
                         colorscale='reds',
                         showscale=False,
                         name='Country')

# Univariate Adults
def uv_adt(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_ADT].value_counts().sort_index()
    return go.Pie(labels=cnts.index,
                  values=cnts.values,
                  hole=.5,
                  rotation=315,
                  name='N.O. Adults')

# Univariate Children
def uv_chldn(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_CHLDN].value_counts().sort_index()
    return go.Pie(labels=cnts.index,
                  values=cnts.values,
                  hole=.5,
                  rotation=270,
                  name='N.O. Children')

# Univariate Market Segment
def uv_mrk_seg(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_MKT_SEG].value_counts().sort_index()
    return go.Pie(labels=cnts.index,
                  values=cnts.values,
                  hole=.5,
                  rotation=315,
                  name='Name')

# Univariate Agent
def uv_agt(year, mkt_seg, cust_type):
    return go.Histogram(x=uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_AGT],
                        name='ID')

# Univariate Customer Type
def uv_cust_type(year, mkt_seg, cust_type):
    cnts = uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_CUST_TYPE].value_counts().sort_index()
    return go.Pie(labels=cnts.index,
                  values=cnts.values,
                  hole=.5,
                  rotation=270,
                  name='Type')

# Average Daily Rate
def uv_avg_dly_rate(year, mkt_seg, cust_type):
    return go.Histogram(x=uv_mod_df(year, mkt_seg, cust_type)[COL_NAME_AVG_DLY_RATE],
                        nbinsx=50,
                        name='USD')

# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div(children='Nihgi (team 6) Capstone Project'),
    html.Hr(),
    html.Div(children=[
        html.Div(children=[
            html.Label('Year'),
            dcc.Dropdown(crt_ddl(COL_NAME_ARR_DATE_Y), value='All', id='controls-and-dropdown-year')
        ], style={'padding': 10, 'flex': 1}),
        html.Div(children=[
            html.Label('Market Segment'),
            dcc.Dropdown(crt_ddl(COL_NAME_MKT_SEG), value='All', id='controls-and-dropdown-ms')
        ], style={'padding': 10, 'flex': 1}),
        html.Div(children=[
            html.Label('Customer Type'),
            dcc.Dropdown(crt_ddl(COL_NAME_MKT_SEG), value='All', id='controls-and-dropdown-ct')
        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Graph(figure={}, id='controls-and-graph')
])

# Add controls to build the interaction
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='controls-and-dropdown-year', component_property='value')
)
def update_graph(year='All', mkt_seg='All', cust_type='All'):
    figs = sp.make_subplots(7, 4,
                            specs=[[{'type': 'pie', 'rowspan': year == 'All' and 1 or 2, 'colspan': 1}, {'type': 'pie'}, {'type': 'pie'}, {}],
                                   [{'type': 'pie'}, {}, {}, {}],
                                   [{'rowspan': 1, 'colspan': mkt_seg == 'All'and 1 or 2}, {'type': 'pie'}, {'type': cust_type == 'All' and 'pie' or 'histogram', 'rowspan': 1, 'colspan': cust_type == 'All' and 1 or 2}, {}],
                                   [{'type': 'choropleth', 'rowspan': 4, 'colspan': 4}, None, None, None],
                                   [None, None, None, None],
                                   [None, None, None, None],
                                   [None, None, None, None]],
                            subplot_titles=[COL_NAME_HTL, COL_NAME_ADT, COL_NAME_CHLDN, COL_NAME_AVG_DLY_RATE,
                                            year == 'All' and COL_NAME_ARR_DATE_Y or '', COL_NAME_ARR_DATE_M, COL_NAME_WKDAY, COL_NAME_ARR_DATE_D,
                                            COL_NAME_LEAD_TIME, mkt_seg == 'All' and COL_NAME_MKT_SEG or '', cust_type == 'All' and COL_NAME_CUST_TYPE or COL_NAME_AGT, cust_type == 'All' and COL_NAME_AGT or ''])
    figs.add_trace(uv_htl(year, mkt_seg, cust_type), 1, 1)
    figs.add_trace(uv_adt(year, mkt_seg, cust_type), 1, 2)
    figs.add_trace(uv_chldn(year, mkt_seg, cust_type), 1, 3)
    figs.add_trace(uv_avg_dly_rate(year, mkt_seg, cust_type), 1, 4)
    if year == 'All':
        figs.add_trace(uv_arr_date_year(year, mkt_seg, cust_type), 2, 1)
    figs.add_trace(uv_arr_date_month(year, mkt_seg, cust_type), 2, 2)
    figs.add_trace(uv_wkday(year, mkt_seg, cust_type), 2, 3)
    figs.add_trace(uv_arr_date_day(year, mkt_seg, cust_type), 2, 4)
    figs.add_trace(uv_lead_time(year, mkt_seg, cust_type), 3, 1)
    if mkt_seg == 'All':
        figs.add_trace(uv_mrk_seg(year, mkt_seg, cust_type), 3, 2)
    if cust_type == 'All':
        figs.add_trace(uv_cust_type(year, mkt_seg, cust_type), 3, 3)
    figs.add_trace(uv_agt(year, mkt_seg, cust_type), 3, cust_type == 'All' and 4 or 3)
    figs.add_trace(uv_cntry(year, mkt_seg, cust_type), 4, 1)
    figs.update_layout(height=1000,
                       title_text='EDA Univariate Analysis for Portugal Hotel Booking',
                       showlegend=False)
    return figs

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

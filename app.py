import streamlit as st
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import vnquant.data as dt

	

st.set_page_config(page_title = "Nguyen Quang Bac's Stock Portfolio Optimizer", layout = "wide")
st.header("Nguyen Quang Bac's Stock Portfolio Optimizer")

col1, col2 = st.columns(2)

with col1:
	start_date = st.date_input('Start Date',datetime(2010,1,1))
	
with col2:
	end_date = st.date_input('End Date') # it defaults to current date
MCK_string = st.text_input('Nhập mã chứng khoán mà không có dấu cách ví dụ: TCB,SSI,VHC,VHM,HBC,FPT,HPG','')
MCK = MCK_string.split(',')
loader = dt.DataLoader(MCK, str(start_date),str(end_date), minimal=True, data_source = "cafe")
data= loader.download()
data=data.stack()
data=data.reset_index()
data1 = data.pivot_table(values = 'adjust', index = 'date', columns = 'Symbols').dropna()



mu = expected_returns.mean_historical_return(data1)
S = risk_models.sample_cov(data1)
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
performance = ef.portfolio_performance(verbose=True)

#HRP
from pypfopt import hierarchical_portfolio
returns = expected_returns.returns_from_prices(data1, log_returns=False)
hierarchical_portfolio.HRPOpt(returns,S)
hrp = hierarchical_portfolio.HRPOpt(returns,risk_models.sample_cov(data1))
weight_hrp = hrp.optimize()
hrp_performance = hrp.portfolio_performance(verbose=True)
#min Cvar
from pypfopt import efficient_frontier
cvar = efficient_frontier.EfficientCVaR(mu,returns,beta = 0.95,weight_bounds=(0, 1),verbose=False)
mincvar=cvar.min_cvar()
cvar_performance = cvar.portfolio_performance(verbose=True)

form = st.form(key="my_form")
submit1 = form.form_submit_button(label="Get data")
if submit1:
    st.write(data1)
    st.subheader("Optimized Max Sharpe Portfolio")
    st.write(cleaned_weights)
    st.write(performance)
 
    st.subheader("HRP")
    st.write(weight_hrp)
    st.write(hrp_performance)
  
    st.subheader("Min CVaR")
    st.write(cvar_performance)
   

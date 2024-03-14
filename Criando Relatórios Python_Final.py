#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # *Criando relatórios usando a FinQuant*

# Vamos trabalhar algumas bibliotecas, nomeadamente:
# 
# * FinQuant
# * QuantStats
# * RiskFolio
# 
# E por último, vamos colocar isso num PDF com a biblioteca FPDF.

# # *Arrumando Bugs para baixar a riskfolio-lib*

# In[131]:


#1º Passo
# !pip uninstall matplotlib
# !pip install matplotlib

# 2º passo
# exit()

# 3º passo
# !pip install cvxpy

# 4º passo
# !pip install pybind11


# ## *1. Instalações e Importações*

# In[132]:


#!pip install finquant


# In[133]:


#!pip install riskfolio-lib


# In[134]:


#!pip install pybind11


# In[135]:


#!pip install cvxpy


# In[136]:


#!pip install FPDF


# In[137]:


# Importações
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import riskfolio as rp
from fpdf import FPDF
from finquant.portfolio import build_portfolio
warnings.filterwarnings('ignore')


# ## *2. In sample data*

# In[138]:


#Período
inicio = '2017-01-01'
fim = '2022-08-30'

#Seleção dos ativos da carteira
ativos = ['PRIO3.SA','VALE3.SA', 'ITUB3.SA','HYPE3.SA', 'TAEE11.SA' ,'WEGE3.SA', 'IVVB11.SA','O']

#Peso da carteira anteriormente
peso_in = np.array([0.10,0.10,0.15,0.15,0.15,0.15,0.10,0.10])


# ## *3.Download dos dados*

# In[139]:


carteira = yf.download(ativos, start=inicio, end=fim)['Adj Close']
carteira.head()


# ## *4. Calculo dos parâmetros*

# In[140]:


#Retornos
retorno_carteira = carteira.pct_change().dropna()

#Covariância
cov_in = retorno_carteira.cov()


# In[141]:


# Pesos da carteira
pesos_in = pd.DataFrame(data={'pesos_in':peso_in},index=ativos)
pesos_in


# ## *5. Retorno out of sample*

# In[142]:


out_inicio = '2022-09-01'
out_fim = '2024-09-11'

#Download dos dados
carteira_out = yf.download(ativos, start = out_inicio, end = out_fim)['Adj Close']

#Calculo Retorno
retorno_out = carteira_out.pct_change().dropna()

#Matriz covariancia out-of-sample
cov_out = retorno_out.cov()

display(retorno_out.head())


# In[143]:


# Dendrograma dos ativos
ax = rp.plot_dendrogram(returns=retorno_carteira,
                      codependence='pearson',
                      linkage='single',
                      k=None,
                      max_k=10,
                      leaf_order=True,
                      ax=None)


# ## *6. Modelo de otimização*
# 
# <br>
# 
# - Marcos López de Prado. Building diversified portfolios that outperform out of sample. The Journal of Portfolio Management, 42(4):59–69, 2016. URL: https://jpm.pm-research.com/content/42/4/59, arXiv:https://jpm.pm-research.com/content/42/4/59.full.pdf, doi:10.3905/jpm.2016.42.4.059.
# 
# <br>
# 
# - Copyright (c) 2020-2022, Dany Cajas All rights reserved.  
#     fonte: https://riskfolio-lib.readthedocs.io/en/latest/index.html

# In[144]:


pd.options.display.float_format = '{:.4%}'.format

portfolio = rp.HCPortfolio(returns=retorno_carteira)

model='HRP' 
codependence = 'pearson'
rm = 'MV' 
rf = 0 # 
linkage = 'single' 

leaf_order = True 

pesos = portfolio.optimization(model=model,
                      codependence=codependence,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      leaf_order=leaf_order)
display(pesos)


# In[145]:


#Retorno out of sample
fig_2, ax_2 = plt.subplots(figsize=(1,1))

rp.plot_series(returns=retorno_out, w=pesos, cmap='tab20', height=6, width=10,
                    ax=None)
plt.savefig('cum_ret.png');


# In[146]:


#Gráfico de composição dos pesos antes
fig_2, ax_2 = plt.subplots(figsize=(6,2))

rp.plot_pie(w=pesos_in, title='Portfolio', height=6, width=10,
                 cmap="tab20", ax=None)
plt.savefig('pf_weights_in.png');


# In[147]:


#Gráfico de composição dos novos pesos da carteira otimizada
fig_3, ax_3 = plt.subplots(figsize=(6,2))

rp.plot_pie(w=pesos, title='Portfolio', height=6, width=10,
                 cmap="tab20", ax=None)
plt.savefig('pf_weights_out.png');


# In[148]:


#Parametros do portfolio otimizado
media_retorno = portfolio.mu
covariancia = portfolio.cov
retornos = portfolio.returns


# ### *Contribuição de risco por ativo*

# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYQAAAAzCAYAAACXBX+qAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAACB/SURBVHhe7d1lbC3V1wbwwd3d3d3d3SV44AuQQIIkkBA+EAh8JIQQSIDgEPzi7u7u7u7u+n/57bfrZnJy2s6c0ytc1pNM2s6Z2XutZ+neM20n+t8/qBKJRCLxn8fEA18TiUQi8R9HFoREIpFIFGRBSCQSiURBFoREIpFIFGRBSCQSiURBFoREIpFIFGRBSCQSiURBFoREIpFIFGRBSCQSiURB/qbyMPj777/L8fPPP1e///57OTf11FOXI5FIJCYkZEEYBr/88kv1ww8/VPfff3/1+uuvVxNNNFG19tprVxtssMHAFYlEIjFhIAvCMPjiiy+qd955p7rpppuqp556qvr++++rlVZaqVpzzTUHrkgkEolxiyWWWKJaeeWVB37qHVkQhoFVwQMPPFA99thj1dNPP1299tprZetokkkmGbgikUgkxi0OOuig6oQTThj4qXdkQRgGCsGoUaOqKaaYoppqqqmq6aabrpp00kkHPk0kEolxjxVWWKFab731Bn7qHVkQBkHQcvvtt1ennHJKtckmm5RnB8suu2w17bTTls/aIsb0gPq7776r/vrrr9EPrT2knmGGGarJJpusr9XHn3/+Wf32229la+uPP/4o58yroM0888ylmE088b/j5bLg69dff62+/fbboptzeFOcg69+CrTxjMse5gl7sMGMM85YTTnllGWOROK/gCwIgyBosTo46qijqiOOOKLaYYcdqllmmaXnBBHJxzbU3XffXQqDh9a+KjTrrrtuNfvss/dccEDi/PDDD6sHH3yw+uqrr0bPOf/881dbbbVVkX/yyScfuHr8Rsj+7rvvVvfcc0/RLfhadNFFq4033riabbbZqumnn37gjvYwvuLJHuZRFIyv2Gy00UbVwgsvXM0666wDVycSEzby9xAGgS5UAvrxxx/Lm0W2iiSJfrp3Cc4h4Xz++ecl8c8111wlKb333nvVM888U33zzTcDV/cGq4OffvqprA6Mv+CCC5bu99VXX63efvvtvscf28CXJO3hvm593nnnLec++uij6oknnqi+/PLLgSt7g7FwhRcrAwXAasp8zz//fPX+++8PXDn2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)

# In[149]:


#Gráfico de contribuição de medida de risco por ativo carteira as is
fig_4, ax_4 = plt.subplots(figsize=(6,2))

rp.plot_risk_con(w=pesos_in, cov=cov_in, returns=retorno_carteira, rm=rm,
                      rf=0, alpha=0.05, color="tab:blue", height=6,
                      width=10, t_factor=252, ax=None)
plt.savefig('risk_cont_in.png');


# In[150]:


cov_out


# In[151]:


#Gráfico de contribuição de medida de risco por ativo carteira as os

fig_5, ax_5 = plt.subplots(figsize=(6,2))

rp.plot_risk_con(w=pesos, cov=cov_out, returns=retorno_out, rm=rm,
                      rf=0, alpha=0.05, color="tab:blue", height=6,
                      width=10, t_factor=252, ax=None)
plt.savefig('risk_cont_out.png');


# ### *Histograma de retornos do portfólio*

# In[152]:


#Histograma dos retornos do portfolio in sample
fig_6, ax_6 = plt.subplots()

rp.plot_hist(returns=retorno_carteira, w=pesos_in, alpha=0.05, bins=50, height=6,
                  width=10, ax=None)
plt.savefig('pf_returns_in.png');


# In[153]:


# Histograma dos retornos do portfolio out sample
fig_7, ax_7 = plt.subplots()

rp.plot_hist(returns=retorno_out, w=pesos, alpha=0.05, bins=50, height=6,
                  width=10, ax=None);
plt.savefig('pf_returns_out.png')


# ### Tabela de medidas de risco

# In[154]:


# Carteira in sample
fig_8, ax_8 = plt.subplots(figsize=(6,2))
rp.plot_table(returns=retorno_carteira, w=pesos_in, MAR=0, alpha=0.05, ax=None)
plt.savefig('table_in.png');


# In[155]:


# Carteira out sample
fig_9, ax_9 = plt.subplots(figsize=(6,2))
rp.plot_table(returns=retorno_out, w=pesos, MAR=0, alpha=0.05, ax=None)
plt.savefig('table_out.png');


# ## *7. Construindo o Relatório em PDF*
# 
# - Biblioteca utilizada FPDF for Python
#     https://pyfpdf.readthedocs.io/en/latest/index.html

# In[109]:


# 1. Setup básico do PDF

#Criamos o pdf
pdf = FPDF()

#Adicionamos uma nova página
pdf.add_page()

#Setup de fonte 
pdf.set_font('Arial', 'B', 16)

# 2. Layout do pdf

## Título
pdf.cell(40, 10, 'Diagnóstico da sua Carteira')

## Quebra de linha
pdf.ln(20)

# 3. Tabela performance
pdf.cell(20, 7, 'Como sua carteira performou de {} até {}'.format(inicio,fim))
pdf.ln(8)
pdf.image('table_in.png', w=180, h=200)
pdf.ln(60)

# 4. Tabela peformance out-of-sample
pdf.cell(20, 7, 'Como sua carteira performou de {} até {}'.format(out_inicio,out_fim))
pdf.ln(8)
pdf.image('table_out.png', w=180, h=200)
pdf.ln(60)

# 5. Retorno Acumulado Carteira
pdf.cell(20, 7, 'Retorno Acumulado da Carteira de {} até {}'.format(out_inicio,out_fim))
pdf.ln(8)
pdf.image('cum_ret.png', w=120, h=70)
pdf.ln(10)
         
# 6. Pesos         
pdf.cell(20, 7, 'Pesos Carteira Atual')
pdf.ln(8)
pdf.image('pf_weights_in.png', w=100, h=60)
pdf.ln(10)

pdf.cell(20, 7, 'Pesos Carteira Otimizada')
pdf.ln(8)
pdf.image('pf_weights_out.png', w=100, h=60)
pdf.ln(30)
        
# 7. Contribuição de risco por ativo
pdf.cell(20, 7, 'Contribuição de risco por ativo de {} até {}'.format(inicio,fim))
pdf.ln(15)
pdf.image('risk_cont_in.png',w=150, h=80)
pdf.ln(20)
pdf.cell(20, 7, 'Contribuição de risco por ativo de {} até {}'.format(out_inicio,out_fim))
pdf.ln(15)
pdf.image('risk_cont_out.png',w=150, h=80)
pdf.ln(80)         
  
# 8. Histograma de retornos
pdf.cell(20, 7, 'Histograma de retornos de {} até {}'.format(inicio,fim))
pdf.ln(15)
pdf.image('pf_returns_in.png', w=150, h=80)
pdf.ln(20)

pdf.cell(20, 7, 'Histograma de retornos de {} até {}'.format(out_inicio,out_fim))
pdf.ln(15)
pdf.image('pf_returns_out.png', w=150, h=80)
pdf.ln(20)

pdf.add_page()

pdf.cell(20, 7, 'Fronteira Eficiente')
pdf.ln(15)
pdf.image('efficientfrontier.png', w=150, h=80)
pdf.ln(20)
         
# 9. Disclaimer
pdf.set_font('Times', '', 6)
pdf.cell(5, 2, 'Relatório construído com a biblioteca RiskFolio https://riskfolio-lib.readthedocs.io/en/latest/index.html')

# 10. Output do PDF file
pdf.output('diagnostico_de_carteira.pdf', 'F')


# # *7.1 Construindo o Relatório em PDF pt 2*
# 
# <br>
# 
# - Explorando a biblioteca finquant

# In[86]:


names = ['^BVSP', 'VALE3.SA', 'PETR4.SA', 'WEGE3.SA',
         'ABEV3.SA', 'HGLG11.SA']


# In[87]:


start_date = '2018-01-01'
end_date = '2022-01-01'


# In[88]:


pf = build_portfolio(names = names,
                     start_date = start_date,
                     end_date = end_date,
                     data_api = 'yfinance')


# In[89]:


pf.data


# In[90]:


pf.properties()


# In[91]:


pf.plot_stocks()


# In[92]:


pf.comp_stock_volatility()


# In[112]:


pf.comp_cumulative_returns().plot();
plt.savefig('comp_cumulative_returns.png')


# In[113]:


pf.comp_volatility() * 100


# In[114]:


pf.comp_expected_return() * 100


# In[115]:


pf.ef_maximum_sharpe_ratio()


# In[116]:


pf.ef_minimum_volatility()


# Análise Técnica

# In[117]:


from finquant.moving_average import compute_ma, ema, sma


# In[118]:


petro = pf.get_stock('PETR4.SA').data


# In[119]:


petro


# In[120]:


# Estratégia de cruzamento de 3 médias móveis
spans = [10, 50, 100]


# In[121]:


ma = compute_ma(petro, ema, spans, plot = True)
plt.savefig('band_moving_avareges_petro_sma.png')


# In[122]:


ma


# In[123]:


from finquant.moving_average import plot_bollinger_band


# In[124]:


span = 20


# In[125]:


plot_bollinger_band(petro, sma, span)
plt.savefig('bollinger_band_petro_sma.png')


# Fronteira Eficiente
# 
# - 10 mil iterações do método de Monte Carlo

# In[126]:


opt_w, opt_res = pf.mc_optimisation(num_trials = 10000)


# In[127]:


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

figure(figsize = (15,8), dpi = 80)

# Plotando resultados da simulação
pf.mc_plot_results()
# Plotando a curva da fronteira eficiente 
pf.ef_plot_efrontier()
# Plotando os portfólios
pf.ef_plot_optimal_portfolios()
# Plotando os ativos isoladamente 
pf.plot_stocks()

plt.savefig('efficientfrontier.png')


# In[128]:


opt_w


# In[129]:


opt_res


# Novo relatório com todas as imagens

# In[157]:


# 1. Setup básico do PDF

#Criamos o pdf
pdf = FPDF()

#Adicionamos uma nova página
pdf.add_page()

#Setup de fonte 
pdf.set_font('Arial', 'B', 16)

# 2. Layout do pdf

## Título
pdf.cell(40, 10, 'Diagnóstico da sua Carteira')

## Quebra de linha
pdf.ln(20)

# 3. Tabela performance
pdf.cell(20, 7, 'Como sua carteira performou de {} até {}'.format(inicio,fim))
pdf.ln(8)
pdf.image('table_in.png', w=180, h=200)
pdf.ln(60)

# 4. Tabela peformance out-of-sample
pdf.cell(20, 7, 'Como sua carteira performou de {} até {}'.format(out_inicio,out_fim))
pdf.ln(8)
pdf.image('table_out.png', w=180, h=200)
pdf.ln(60)

# 5. Retorno Acumulado Carteira
pdf.cell(20, 7, 'Retorno Acumulado da Carteira de {} até {}'.format(out_inicio,out_fim))
pdf.ln(8)
pdf.image('cum_ret.png', w=120, h=70)
pdf.ln(10)
         
# 6. Pesos         
pdf.cell(20, 7, 'Pesos Carteira Atual')
pdf.ln(8)
pdf.image('pf_weights_in.png', w=100, h=60)
pdf.ln(10)

pdf.cell(20, 7, 'Pesos Carteira Otimizada')
pdf.ln(8)
pdf.image('pf_weights_out.png', w=100, h=60)
pdf.ln(30)
        
# 7. Contribuição de risco por ativo
pdf.cell(20, 7, 'Contribuição de risco por ativo de {} até {}'.format(inicio,fim))
pdf.ln(15)
pdf.image('risk_cont_in.png',w=150, h=80)
pdf.ln(20)
pdf.cell(20, 7, 'Contribuição de risco por ativo de {} até {}'.format(out_inicio,out_fim))
pdf.ln(15)
pdf.image('risk_cont_out.png',w=150, h=80)
pdf.ln(80)         
  
# 8. Histograma de retornos
pdf.cell(20, 7, 'Histograma de retornos de {} até {}'.format(inicio,fim))
pdf.ln(15)
pdf.image('pf_returns_in.png', w=150, h=80)
pdf.ln(20)

pdf.cell(20, 7, 'Histograma de retornos de {} até {}'.format(out_inicio,out_fim))
pdf.ln(15)
pdf.image('pf_returns_out.png', w=150, h=80)
pdf.ln(20)

pdf.add_page()

pdf.cell(20, 7, 'Comp. Cumulative Returns')
pdf.ln(15)
pdf.image('comp_cumulative_returns.png', w=150, h=80)
pdf.ln(20)

pdf.cell(20, 7, 'Band Moving Avareges - PETR4')
pdf.ln(15)
pdf.image('band_moving_avareges_petro_sma.png', w=150, h=80)
pdf.ln(20)

pdf.add_page()

pdf.cell(20, 7, 'Bollinger Bands SMA - PETR4')
pdf.ln(15)
pdf.image('bollinger_band_petro_sma.png', w=150, h=80)
pdf.ln(20)

pdf.cell(20, 7, 'Fronteira Eficiente')
pdf.ln(15)
pdf.image('efficientfrontier.png', w=150, h=80)
pdf.ln(20)

# 9. Disclaimer
pdf.set_font('Times', '', 6)
pdf.cell(5, 2, 'Relatório construído com a biblioteca RiskFolio https://riskfolio-lib.readthedocs.io/en/latest/index.html')

# 10. Output do PDF file
pdf.output('diagnostico_de_carteira.pdf', 'F')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





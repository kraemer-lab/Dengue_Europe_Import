# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import pytensor.tensor as pt
import scipy as sp

np.random.seed(33)

#####plotting parameters
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.titlesize': 12})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

## load map
wmap = gpd.read_file("./data/wmap/world-administrative-boundaries.shp")
wmap['country'] = wmap.iso_3166_1_

##### Load and Clean data all
df0 = pd.read_csv("./data/dengue_clean_data_travel.csv")
df0['year'] = [d[:4] for d in df0.term]
df0 = df0[df0.year > '2011']
df0['infection_country'] = df0.country.values
df0 = df0.dropna()

# ## load and add passenger info
# wpass = pd.read_csv("./data/world_passengers_imputed.csv")

# fdata = pd.read_csv("./data/euro_air_passengers.csv")
# fdata = fdata[['TIME_PERIOD', 'OBS_VALUE', 'geo']]
# fdata.columns = ['year', 'passengers', 'reporting_country']
# fdata = fdata[fdata.year.isin(np.arange(2012, 2022))]
# fdata = fdata[fdata.reporting_country.isin(df0.reporting_country.unique())]
# fdata['year'] = fdata.year.values.astype("str")
# fdata = fdata.groupby(['year', 'reporting_country'], as_index=False).sum()

# df0 = pd.merge(df0, fdata, how='left',on=['reporting_country','year'])
# df0.reset_index(drop=True, inplace=True)

# wpass['infection_country'] = wpass.country
# wpass['passengers_i'] = wpass.passengers
# wpass['year'] = wpass.year.values.astype("str")
# wpass = wpass[['infection_country', 'year', 'passengers_i']]
# df0 = pd.merge(df0, wpass, how='left',on=['infection_country','year'])
# df0['passengers_i'] = df0['passengers_i'].fillna(0)
# df0['passengers_r'] = df0.passengers
# df0 = df0.dropna()
# df0.reset_index(inplace=True, drop=True)

# df0['passengers_i'] = df0['passengers_i'].fillna(0) 
# passen = []
# for i in range(len(df0)):
#     c = df0['cases'].values[i]
#     p = df0['passengers_i'].values[i]
#     if p < c:
#         passen.append(c*2)
#     else:
#         passen.append(p)
# df0['passengers_i'] = passen

df_risk = pd.read_csv("./data/dengue_travel_europe_2015-2019_cleaned.csv")
df_risk = df_risk[df_risk.country.isin(df0.country.unique())]

df0 = df0[df0.country.isin(df_risk.country.unique())]
df0.reset_index(inplace=True, drop=True)

### select time periods to calibrate
# selection = ['MX','BR','VN','MV','PH','CU','LK','ID','IN','TH']
# selection = ['ID','IN','TH']
# df0 = df0[df0.infection_country.isin(selection)]

df_rank = df0.groupby('dyad', as_index=False).sum()
df_rank = df_rank.sort_values('imported')
ra_look = dict(zip(df_rank.imported.unique().astype('int'), range(len(df_rank.imported.unique()))))
df_look = pd.DataFrame({'imported':ra_look.keys(), 'dyad_ranks':ra_look.values()})
df_rank = pd.merge(df_rank, df_look, on='imported', how='left')
df_rank_d = df_rank[['dyad','dyad_ranks']]

df_rank = df0.groupby('reporting_country', as_index=False).sum()
df_rank = df_rank.sort_values('imported')
ra_look = dict(zip(df_rank.imported.unique().astype('int'), range(len(df_rank.imported.unique()))))
df_look = pd.DataFrame({'imported':ra_look.keys(), 'rep_ranks':ra_look.values()})
df_rank = pd.merge(df_rank, df_look, on='imported', how='left')
df_rank_r = df_rank[['reporting_country','rep_ranks']]
df_rank_r['rep_ranks'] = df_rank_r.rep_ranks.values

# df_rank = df0.groupby('reporting_country', as_index=False).sum()
# df_rank = df_rank.sort_values('imported')
# ra_look = dict(zip(df_rank.imported.unique().astype('int'), range(len(df_rank.imported.unique()))))
# df_look = pd.DataFrame({'imported':ra_look.keys(), 'rep_ranks':ra_look.values()})
# df_rank = pd.merge(df_rank, df_look, on='imported', how='left')
# df_rank_r = df_rank[['reporting_country','rep_ranks']]
# df_rank_r['rep_ranks'] = df_rank_r.rep_ranks.values-1

df0 = pd.merge(df0, df_rank_d, on='dyad', how='left')
df0 = pd.merge(df0, df_rank_r, on='reporting_country', how='left')

# df0 = df0.sort_values('dyad_ranks', ascending=True)

df = df0.copy()

df = df0[df0.term < "2019-1st"]
# df = df.sort_values('dyad_ranks', ascending=True)

df_unobs = df0[df0.term > '2018-4th'] 
# df_unobs = df_unobs.sort_values('dyad_ranks', ascending=True)

#df.reset_index(inplace=True, drop=True)

###### Prepare relevant variables for model
r_look = dict(zip(df.reporting_country.unique(), range(len(df.reporting_country.unique()))))
rcountry_idx = df.reporting_country.replace(r_look).values #reporting country index

e_look = dict(zip(df.infection_country.unique(), range(len(df.infection_country.unique()))))
ecountry_idx = df.infection_country.replace(e_look).values #exporting country index

d_look = dict(zip(df.dyad.unique(), range(len(df.dyad.unique()))))
dyad_idx = df.dyad.replace(d_look).values # dyad porting country index

t_look = dict(zip(df.term.unique(), range(len(df.term.unique()))))
term_idx = df.term.replace(t_look).values #reporting country index

d_rank_idx = df.dyad_ranks.values.astype('int') 
r_rank_idx = df.rep_ranks.values.astype('int') 

d_e = []
d_r = []
for i in range(len(df.dyad.unique())):
    d = df.dyad.unique()[i]
    d_e.append(e_look[d[:2]])
    d_r.append(r_look[d[3:]])
    
D = np.array([d_e, d_r])

Dlen = len(df.dyad.unique())
E = len(df.infection_country.unique()) #exporting country
R = len(df.reporting_country.unique())
Tlen = len(df.term.unique())
DRlen = len(df.dyad_ranks.unique())
Rlen = len(df.rep_ranks.unique())
terms = np.arange(len(df.term.unique()))
terms_exp = np.array([np.arange(len(df.term.unique())) for i in range(E)]).T
imported = df.imported.values

#(lam_e/(lam_p-lam_e))*lam_p
    
x = df['travellers_ee'].values # number of passengers from exporting country to Europe
x_z = (x - x.mean())/x.std()

# a = np.flip(np.sort(np.histogram(3, bins=Dlen, range=(0,10))[1])[1:])
# a = np.ones(Dlen)
ad = np.array(np.repeat(1, DRlen-1))
ar = np.array(np.repeat(1, Rlen))

coords = {"term":df.term.unique(),
          "dyad":df.dyad.unique(),
          "country_e":df.infection_country.unique(),
          "country_r":df.reporting_country.unique(),
          "location":df.index.values,
          "feature":['exporting','reporting']}

##### Build up Model 
with pm.Model(coords=coords) as mod:
    t_idx = pm.ConstantData("term_idx", term_idx, dims="location")
    d_idx = pm.ConstantData("dyad_idx", dyad_idx, dims="location")
    e_idx = pm.ConstantData("ecountry_idx", ecountry_idx, dims="location")
    # r_idx = pm.ConstantData("rcountry_idx", rcountry_idx, dims="location")
    dr_idx = pm.ConstantData("d_rank_idx", d_rank_idx, dims="location")
    r_idx = pm.ConstantData("r_rank_idx", r_rank_idx, dims="location")
    D = pm.ConstantData("D", D.T, dims=("dyad", "feature"))
    T = pm.ConstantData("T", terms, dims=("term"))
    xn = pm.ConstantData("xn", x_z, dims="location")

    sigma = pm.HalfNormal("sigma", 1)
    l = pm.HalfNormal("l", 1)
    K = pm.gp.cov.ExpQuad(input_dim=1, ls=l) * sigma**2
    latent_t = pm.gp.Latent(cov_func=K,)
    tau = latent_t.prior("tau", T[:,None], dims="term")
    
    # beta = pm.Normal("beta", 0, 1)
    # delta_z = pm.Dirichlet("delta_z", a=ad, shape=DRlen-1)
    # delta_r = pt.concatenate([pt.zeros(1), delta_z])
    # delta = beta*pt.cumsum(delta_r)
    
    eta = pm.Normal("eta", 0, 1)
    rho_z = pm.Dirichlet("rho_z", a=ar, shape=Rlen)
    rho_r = pt.concatenate([pt.zeros(1), rho_z])
    rho = eta*pt.cumsum(rho_r)
    
    delta_l = pm.Normal("delta_l", 0, 1)
    delta_z = pm.Normal("delta_z", 0, 1, dims="dyad")
    delta_s = pm.HalfNormal("delta_s", 1)
    delta = pm.Deterministic("delta", delta_l + delta_s*delta_z, dims="dyad")
    
    epsi_l = pm.Normal("epsi_l", 0, 1)
    epsi_z = pm.Normal("epsi_z", 0, 1, dims="country_e")
    epsi_s = pm.HalfNormal("epsi_s", 1)
    epsi = pm.Deterministic("epsi", epsi_l + epsi_s*epsi_z, dims="country_e")
    
    
    lam = pm.Deterministic("lam", tau[t_idx] + delta[d_idx] + epsi[e_idx]*xn + rho[r_idx])

    alpha = pm.HalfNormal("alpha", 1)

    y = pm.NegativeBinomial('y', mu=pm.math.exp(lam), alpha=alpha, observed=imported)
    
    # y = pm.Poisson("y", mu=pm.math.exp(lam), observed=imported)
    

with mod:
    idata = pm.sample(100, init=100, target_accept=0.95, chains=4, cores=12, 
                      nuts_sampler='numpyro', random_seed=33)

az.to_netcdf(idata, "idata_model.nc")

summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("idata_summary.csv")    

az.plot_trace(idata, kind='rank_vlines', var_names=['l', 'sigma',  'eta', 'epsi_l', 'epsi_s'])
plt.tight_layout()
plt.savefig("rank_plots.png", dpi=300)
plt.close()


## Load inference data
idata = az.from_netcdf("idata_model.nc")

with mod:
    pred0 = pm.sample_posterior_predictive(idata, var_names=["y",'lam', 'alpha'], predictions=True)
pred_y0 = az.extract(pred0, group="predictions")['y'].values


########################### Predictions ##############################
######################################################################

#df_unobs.reset_index(drop=True, inplace=True)

r_look_u = dict(zip(df_unobs.reporting_country.unique(), range(len(df_unobs.reporting_country.unique()))))
rcountry_u_idx = df_unobs.reporting_country.replace(r_look_u).values #reporting country index

e_look_u = dict(zip(df_unobs.infection_country.unique(), range(len(df_unobs.infection_country.unique()))))
ecountry_u_idx = df_unobs.infection_country.replace(e_look_u).values #exporting country index

d_look_u = dict(zip(df_unobs.dyad.unique(), range(len(df_unobs.dyad.unique()))))
dyad_u_idx = df_unobs.dyad.replace(d_look_u).values # dyad porting country index

t_look_u = dict(zip(df_unobs.term.unique(), range(len(df_unobs.term.unique()))))
term_u_idx = df_unobs.term.replace(t_look_u).values #reporting country index
 

terms_u = np.arange(len(df_unobs.term.unique()))

d_e_u = []
d_r_u = []
for i in range(len(df_unobs.dyad.unique())):
    d = df_unobs.dyad.unique()[i]
    d_e_u.append(e_look_u[d[:2]])
    d_r_u.append(r_look_u[d[3:]])
    
D_u = np.array([d_e_u, d_r_u])

DRulen = len(df.dyad_ranks.unique())
Rulen = len(df.rep_ranks.unique())
Eu = len(df_unobs.country.unique())
Ru = len(df_unobs.reporting_country.unique())

du_rank_idx = df_unobs.dyad_ranks.values.astype('int') 
ru_rank_idx = df_unobs.rep_ranks.values.astype('int') 

adu = np.array(np.repeat(1, DRulen-1))
aru = np.array(np.repeat(1, Rulen))

x_u = df_unobs['travellers'].values 
x_u_z = (x_u - x_u.mean())/x_u.std()

imported_u = df_unobs.imported.values

coords = {"term_u":df_unobs.term.unique(),
          "dyad_u":df_unobs.dyad.unique(),
          "country_e_u":df_unobs.infection_country.unique(),
          "country_r_u":df_unobs.reporting_country.unique(),
          "location_u":df_unobs.index.values,
          "rank_u":df_unobs.dyad_ranks.unique()}

##### Build up Model for predictions
with mod:
    mod.add_coords(coords)
    
    t_u_idx = pm.ConstantData("term_u_idx", term_u_idx, dims="location_u")
    d_u_idx = pm.ConstantData("dyad_u_idx", dyad_u_idx, dims="location_u")
    e_u_idx = pm.ConstantData("ecountry_u_idx", ecountry_u_idx, dims="location_u")
    #r_u_idx = pm.ConstantData("rcountry_u_idx", rcountry_u_idx, dims="location_u")
    dr_u_idx = pm.ConstantData("dr_u_idx", du_rank_idx, dims="location_u")
    r_u_idx = pm.ConstantData("r_u_idx", ru_rank_idx, dims="location_u")
    D_u = pm.ConstantData("D_u", D_u.T, dims=("dyad_u", "feature"))
    T_u = pm.ConstantData("T_u", terms_u, dims="term_u")
    xun = pm.ConstantData("x_u_z", x_u_z, dims="location_u")
    
    tau_u = latent_t.conditional("tau_u", T_u[:,None], dims="term_u")
    
    # delta_z_u = pm.Dirichlet("delta_z_u", a=adu, shape=DRulen-1)
    # delta_r_u = pt.concatenate([pt.zeros(1), delta_z_u])
    # delta_u = beta*pt.cumsum(delta_r_u)
    
    delta_u_z = pm.Normal("delta_u_z", 0, 1, dims="dyad")
    delta_u = pm.Deterministic("delta_u", delta_l + delta_s*delta_u_z, dims="dyad")
    
    rho_z_u = pm.Dirichlet("rho_z_u", a=aru, shape=Rulen)
    rhor_u = pt.concatenate([pt.zeros(1), rho_z_u])
    rho_u = eta*pt.cumsum(rhor_u)
    
    epsi_u_z = pm.Normal("epsi_u_z", 0, 1, dims="country_e_u")
    epsi_u = pm.Deterministic("epsi_u", epsi_l + epsi_s*epsi_u_z, dims="country_e_u")

    lam_u = pm.Deterministic("lam_u", tau_u[t_u_idx]  + delta_u[d_u_idx] +
                                     epsi_u[e_u_idx]*xun + rho_u[r_u_idx])
    
    y_u = pm.NegativeBinomial('y_u', mu=pm.math.exp(lam_u), alpha=alpha)
    
    # y_u = pm.Poisson("y_u", mu=pm.math.exp(lam_u))
    

###### Sample and plot predictions
with mod:
    pred = pm.sample_posterior_predictive(idata, var_names=["y_u", 'lam_u', 'alpha'], predictions=True)

pred_y_u = az.extract(pred, group="predictions")['y_u'].values


pred_y = np.concatenate([pred_y0, pred_y_u])

#similarity index 
def SI(a,b):
    return 2*np.minimum(a,b.T).sum()/(a.sum() + b.sum())

lam = np.concatenate([az.extract(pred0.predictions)['lam'].values, az.extract(pred.predictions)['lam_u'].values])
df0['y_m'] = pred_y.mean(axis=1) #pred_y.mean(axis=1)
df0['y_s'] = pred_y.std(axis=1)**2
# df0['y_s'] = ((np.exp(lam)**2 / az.extract(pred0.predictions)['alpha'].values) + np.exp(lam)).mean(axis=1)

df_term = df0[['term', 'imported', 'y_m', 'y_s']].groupby("term", as_index=False).sum()
df_term['y_s'] = np.sqrt(df_term.y_s.values) 
df_term['y_l'] = df_term.y_m.values -  df_term.y_s.values
df_term['y_u'] = df_term.y_m.values +  df_term.y_s.values
sit = SI(df_term.imported.values, df_term.y_m.values).round(2)

df_r_country = df0[['reporting_country', 'imported', 'y_m', 'y_s']].groupby(["reporting_country"], as_index=False, sort=False).sum()
df_r_country['y_s'] = np.sqrt(df_r_country.y_s.values)
df_r_country['y_l'] = df_r_country.y_m.values -  df_r_country.y_s.values
df_r_country['y_u'] = df_r_country.y_m.values +  df_r_country.y_s.values
sic = SI(df_r_country.imported.values, df_r_country.y_m.values).round(2)
df_r_country = df_r_country.sort_values('imported', ascending=False)

df_country = df0[['country', 'imported', 'y_m', 'y_s']].groupby(["country"], as_index=False, sort=False).sum()
df_country['y_s'] = np.sqrt(df_country.y_s.values)
df_country['y_l'] = df_country.y_m.values - df_country.y_s.values
df_country['y_u'] = df_country.y_m.values + df_country.y_s.values
sice = SI(df_country.imported.values, df_country.y_m.values).round(2)
df_country = df_country.sort_values('imported', ascending=False)

df_exp_term = df0[['term', 'country', 'imported', 'y_m']].groupby(["term","country"], as_index=False).sum()
si_exp = [SI(df0[df0.country==c].imported.values, df0[df0.country==c].y_m.values) for c in df0.country.unique()]
sim_exp = pd.DataFrame({'country':df0.country.unique(), 'SI':si_exp})
sim_exp = pd.merge(df_country, sim_exp, on='country', how='right')
sim_exp.to_csv("similarity_exporting.csv", index=False)

df_rep_term = df0[['term', 'reporting_country', 'imported', 'y_m']].groupby(["term","reporting_country"], as_index=False).sum()
si_rep = [SI(df0[df0.reporting_country==c].imported.values, df0[df0.reporting_country==c].y_m.values) for c in df0.reporting_country.unique()]
sim_rep = pd.DataFrame({'reporting_country':df0.reporting_country.unique(), 'SI':si_rep})
sim_rep = pd.merge(df_r_country, sim_rep, on='reporting_country', how='right')
sim_rep.to_csv("similarity_reporting.csv", index=False)

fig, ax = plt.subplots(3,1, figsize=(27,16))
ax[0].axvspan(28, 39, ymin=0, ymax=1, alpha=0.2, color='grey', label='Unobserved period')
ax[0].plot(np.arange(len(df_term)), df_term.imported, "-o", color='k', lw=3, label="Observed")
ax[0].plot(np.arange(len(df_term)), df_term.y_m, "--o", color='dodgerblue', linestyle="--", lw=3, label="Predicted mean")
ax[0].fill_between(np.arange(len(df_term)), df_term.y_l, df_term.y_u, color='orangered', alpha=0.2, label="SD")
ax[0].plot(np.arange(len(df_term)), np.repeat(np.nan, len(df_term)), color='w', label="SI: "+str(sit))
ax[0].legend()
ax[0].text(-2, 3400, "C", size=35)
ax[0].grid(alpha=0.3)
ax[0].set_ylabel("Count", size=16)
ax[0].set_xticks(np.arange(len(df_term)), df_term.term, rotation=45)
ax[0].set_xlabel("Year-Term", size=16, labelpad=10)
ax[0].set_title("Dengue importations aggregated by term", size=20)
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_ylim(ymin=0, ymax=3000)
ax[0].set_xlim(xmin=-0.5, xmax=len(df_term))
ax[1].plot(np.arange(len(df_r_country)), df_r_country.imported, "o", color='k', ms=12, label="Observed")
ax[1].plot(np.arange(len(df_r_country)), df_r_country.y_m, "o", color='crimson', ms=12, label="Predicted mean")
ax[1].vlines(np.arange(len(df_r_country)), df_r_country.y_l, df_r_country.y_u, color='crimson', linewidth=4, alpha=0.7, label="SD")
ax[1].plot(np.arange(len(df_term)), np.repeat(np.nan, len(df_term)), color='w', label="SI: "+str(sic))
for i in range(len(df_r_country)):
    ax[1].text(np.arange(len(df_r_country))[i]-0.1, df_r_country.y_u.values[i]+200, str(sim_rep.SI.values.round(2)[i]), size=12)
ax[1].legend()
ax[1].text(-1.35, 7000, "D", size=35)
ax[1].grid(alpha=0.3)
ax[1].set_xticks(np.arange(len(df_r_country)), df_r_country.reporting_country)
ax[1].set_xlabel("Country", size=16)
ax[1].set_ylabel("Count", size=16)
ax[1].set_title("Dengue importations aggregated by reporting country", size=20)
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_ylim(ymin=0, ymax=6000)
ax[1].set_xlim(xmin=-0.5, xmax=len(df_r_country))
ax[2].plot(np.arange(len(df_country)), df_country.imported, "o", color='k', ms=12, label="Observed")
ax[2].plot(np.arange(len(df_country)), df_country.y_m, "o", color='crimson', ms=12, label="Predicted mean")
ax[2].vlines(np.arange(len(df_country)), df_country.y_l, df_country.y_u, color='crimson', linewidth=4, alpha=0.7, label="SD")
ax[2].plot(np.arange(len(df_term)), np.repeat(np.nan, len(df_term)), color='w', label="SI: "+str(sice))
for i in range(len(df_country)):
    ax[2].text(np.arange(len(df_country))[i]-0.1, df_country.y_u.values[i]+500, str(sim_exp.SI.values.round(1)[i]), size=10)
ax[2].legend()
ax[2].text(-4, 4500, "E", size=35)
ax[2].grid(alpha=0.3)
ax[2].set_xticks(np.arange(len(df_country)), df_country.country, size=10)
ax[2].set_xlabel("Country", size=16)
ax[2].set_ylabel("Count", size=16)
ax[2].set_title("Dengue importations aggregated by exporting country", size=20)
ax[2].spines[['right', 'top']].set_visible(False)
ax[2].set_ylim(ymin=0, ymax=4000)
ax[2].set_xlim(xmin=-0.5, xmax=len(df_country))
plt.tight_layout()
plt.subplots_adjust(hspace=1)
plt.savefig("posterior_predictives_extended.png", dpi=300)
plt.show()
plt.close()

df0.to_csv("data_summary_all.csv", index=False)


################## Plot on Map ########################

##plot observed
wmap['country'] = wmap.country.replace("GR", "EL")
no_eu = wmap[~wmap.country.isin(df0.reporting_country.unique())]
eu = wmap[wmap.country.isin(df0.reporting_country.unique())]


out_eu = df0[['infection_country', 'year', 'imported']]
out_eu['country'] = out_eu["infection_country"].values
out_eu = out_eu[~out_eu.infection_country.isin(df.reporting_country.unique())]
out_eu = out_eu.groupby(["country","year"], as_index=False).sum()
out_eu['y_m_log'] = np.log(out_eu.imported.values)
out_eu_2019 = out_eu[out_eu.year=="2019"]
out_eu_2020 = out_eu[out_eu.year=="2020"]
out_eu_2021 = out_eu[out_eu.year=="2021"]

out_eu_2019 = pd.merge(wmap, out_eu_2019, how='left', on='country')
out_eu_2020 = pd.merge(wmap, out_eu_2020, how='left', on='country')
out_eu_2021 = pd.merge(wmap, out_eu_2021, how='left', on='country')

left_out = df0[~df0.infection_country.isin(wmap.country.unique())]
left_out['country'] = left_out["infection_country"].values
left_out = left_out[['country', 'year', 'imported']]
left_out = left_out.groupby(["country","year"], as_index=False).sum()
left_out['y_m_log'] = np.log(left_out.imported.values)
left_out_2019 = left_out[left_out.year=="2019"]
left_out_2020 = left_out[left_out.year=="2020"]
left_out_2021 = left_out[left_out.year=="2021"]

lefts2019 = left_out_2019.country.values
lefts2020 = left_out_2020.country.values
lefts2021 = left_out_2021.country.values


vmax = np.max([out_eu_2019.y_m_log.max(), out_eu_2020.y_m_log.max(), out_eu_2021.y_m_log.max()])
vmax = np.round(vmax, 0) 

cmap = plt.get_cmap('plasma')


fig, ax = plt.subplots(3, 1, figsize=(12,10))
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0])
rep_coun = Line2D([0], [0], label='reporting country', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[0].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
out_eu_2019.plot(column='y_m_log', cmap="plasma", ax=ax[0], vmin=0, vmax=vmax)
ax[0].axis("off")
ax[0].text(-200, 90, "A", size=20)
ax[0].set_title("Observed dengue importations to Europe in 2019", size=12)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1])
hands1 = [Line2D([0],[0], label=lefts2020[i], marker='s', ms=10, mfc=cmap(left_out_2020.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2020))]
ax[1].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
#leg1.set_title(title="Extraterritorial", prop={'size':8})
ax[1].axis("off")
ax[1].set_title("Observed dengue importations to Europe in 2020", size=12)
out_eu_2020.plot(column='y_m_log', cmap="plasma", ax=ax[1], vmin=0, vmax=vmax)

im = ax[2].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma', 
                  vmin=0, vmax=vmax, origin="lower")
im.set_visible(False)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("bottom", size="5%", pad=0.05)
   
plt.colorbar(im, cax=cax, orientation='horizontal', label="observed importations (log)")

wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[2])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[2])
out_eu_2021.plot(column='y_m_log', cmap="plasma", ax=ax[2], vmin=0, vmax=vmax) 
hands2 = [Line2D([0],[0], label=lefts2021[i], marker='s', ms=10, mfc=cmap(left_out_2021.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2021))]
ax[2].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
#leg2.set_title(title="Extraterritorial", prop={'size':8})
ax[2].axis("off")
ax[2].set_title("Observed dengue importations to Europe in 2021", size=12)

plt.tight_layout()
plt.savefig("importation_to_europe_observed.png", dpi=600, bbox_inches='tight')



###### Plot predictions
wmap['country'] = wmap.country.replace("GR", "EL")
no_eu = wmap[~wmap.country.isin(df0.reporting_country.unique())]
eu = wmap[wmap.country.isin(df0.reporting_country.unique())]


out_eu = df0[['infection_country', 'year', 'y_m']]
out_eu['country'] = out_eu["infection_country"].values
out_eu = out_eu[~out_eu.infection_country.isin(df.reporting_country.unique())]
out_eu = out_eu.groupby(["country","year"], as_index=False).sum()
out_eu['y_m_log'] = np.log(out_eu.y_m.values)
out_eu_2019 = out_eu[out_eu.year=="2019"]
out_eu_2020 = out_eu[out_eu.year=="2020"]
out_eu_2021 = out_eu[out_eu.year=="2021"]

out_eu_2019 = pd.merge(wmap, out_eu_2019, how='left', on='country')
out_eu_2020 = pd.merge(wmap, out_eu_2020, how='left', on='country')
out_eu_2021 = pd.merge(wmap, out_eu_2021, how='left', on='country')

left_out = df0[~df0.infection_country.isin(wmap.country.unique())]
left_out['country'] = left_out["infection_country"].values
left_out = left_out[['country', 'year', 'y_m']]
left_out = left_out.groupby(["country","year"], as_index=False).sum()
left_out['y_m_log'] = np.log(left_out.y_m.values)
left_out_2019 = left_out[left_out.year=="2019"]
left_out_2020 = left_out[left_out.year=="2020"]
left_out_2021 = left_out[left_out.year=="2021"]

lefts2019 = left_out_2019.country.values
lefts2020 = left_out_2020.country.values
lefts2021 = left_out_2021.country.values


fig, ax = plt.subplots(3, 1, figsize=(12,10))
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0])
rep_coun = Line2D([0], [0], label='reporting country', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[0].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
out_eu_2019.plot(column='y_m_log', cmap="plasma", ax=ax[0], vmin=0, vmax=vmax)
ax[0].axis("off")
ax[0].text(-200, 90, "B", size=20)
ax[0].set_title("Predicted dengue importations to Europe in 2019", size=12)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1])
hands1 = [Line2D([0],[0], label=lefts2020[i], marker='s', ms=10, mfc=cmap(left_out_2020.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2020))]
ax[1].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
#leg1.set_title(title="Extraterritorial", prop={'size':8})
ax[1].axis("off")
ax[1].set_title("Predicted dengue importations to Europe in 2020", size=12)
out_eu_2020.plot(column='y_m_log', cmap="plasma", ax=ax[1], vmin=0, vmax=vmax)

im = ax[2].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma', 
                  vmin=0, vmax=vmax, origin="lower")
im.set_visible(False)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("bottom", size="5%", pad=0.05)
   
plt.colorbar(im, cax=cax, orientation='horizontal', label="predicted importations (log)")

wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[2])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[2])
out_eu_2021.plot(column='y_m_log', cmap="plasma", ax=ax[2], vmin=0, vmax=vmax) 
hands2 = [Line2D([0],[0], label=lefts2021[i], marker='s', ms=10, mfc=cmap(left_out_2021.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2021))]
ax[2].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
#leg2.set_title(title="Extraterritorial", prop={'size':8})
ax[2].axis("off")
ax[2].set_title("Predicted dengue importations to Europe in 2021", size=12)

plt.tight_layout()
plt.savefig("importation_to_europe_predicted.png", dpi=600, bbox_inches='tight')


### combine images
from PIL import Image
plots = ["importation_to_europe_observed.png",
         "importation_to_europe_predicted.png",
         "posterior_predictives_extended.png"]

im1,im2,im3 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2, im1.height+im3.height), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im1.width, 0))
dst.paste(im3, (100, im1.height))

dst.save("calibration_plots.png")

dst.save("calibration_plots.pdf")
# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

np.random.seed(33)

#####plotting parameters
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.titlesize': 24})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

## load map
wmap = gpd.read_file("./data/wmap/world-administrative-boundaries.shp")
wmap['country'] = wmap.iso_3166_1_

##### Load and Clean data all
df0 = pd.read_csv("./data/chik_clean_data_travel.csv")
df0['year'] = [d[:4] for d in df0.term]
df0 = df0[df0.year > '2011']
df0['infection_country'] = df0.country.values
df0 = df0.dropna()
df0 = df0.sort_values('imported', ascending=False)

df_risk = pd.read_csv("./data/dengue_travel_europe_2015-2019_cleaned.csv")
df_risk = df_risk[df_risk.country.isin(df0.country.unique())]

df0 = df0[df0.country.isin(df_risk.country.unique())]
df0.reset_index(inplace=True, drop=True)

### select time periods to calibrate
# selection = ['MX','BR','VN','MV','PH','CU','LK','ID','IN','TH']
# selection = ['ID','IN','TH']
# df0 = df0[df0.infection_country.isin(selection)]


df = df0.copy()

df = df0[df0.term < "2019-1st"]
df.reset_index(drop=True, inplace=True)
#df = df.sort_values(['term', 'country'])

df_unobs = df0[df0.term > '2018-4th'] 
df_unobs.reset_index(drop=True, inplace=True)
df_unobs = df_unobs.set_index(np.arange(len(df_unobs)) + len(df))
#df_unobs = df_unobs.sort_values(['term', 'country'])

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
terms = np.arange(len(df.term.unique()))
terms_exp = np.array([np.arange(len(df.term.unique())) for i in range(E)]).T
imported = df.imported.values

    
x = df['travellers_ee'].values # number of passengers from exporting country to Europe
x_z = (x - x.mean())/x.std()


coords = {"term":df.term.unique(),
          "dyad":df.dyad.unique(),
          "country_e":df.infection_country.unique(),
          "country_r":df.reporting_country.unique(),
          "location":df.index.values}

##### Build up Model 
with pm.Model(coords=coords) as mod:
    t_idx = pm.ConstantData("term_idx", term_idx, dims="location")
    d_idx = pm.ConstantData("dyad_idx", dyad_idx, dims="location")
    e_idx = pm.ConstantData("ecountry_idx", ecountry_idx, dims="location")
    r_idx = pm.ConstantData("rcountry_idx", rcountry_idx, dims="location")
    T = pm.ConstantData("T", terms, dims=("term"))
    xn = pm.ConstantData("xn", x_z, dims="location")

    sigma = pm.HalfNormal("sigma", 1)
    l = pm.HalfNormal("l", 1)
    K = pm.gp.cov.ExpQuad(input_dim=1, ls=l) * sigma**2
    latent_t = pm.gp.Latent(cov_func=K,)
    tau = latent_t.prior("tau", T[:,None], dims="term")

    delta_l = pm.Normal("delta_l", 0, 1)
    delta_z = pm.Normal("delta_z", 0, 1, dims="dyad")
    delta_s = pm.HalfNormal("delta_s", 1)
    delta = pm.Deterministic("delta", delta_l + delta_s*delta_z, dims="dyad")
    
    epsi_l = pm.Normal("epsi_l", 0, 1)
    epsi_z = pm.Normal("epsi_z", 0, 1, dims="country_e")
    epsi_s = pm.HalfNormal("epsi_s", 1)
    epsi = pm.Deterministic("epsi", epsi_l + epsi_s*epsi_z, dims="country_e")

    lam = pm.Deterministic("lam", tau[t_idx] + delta[d_idx] + epsi[e_idx]*xn )

    alpha = pm.HalfNormal("alpha", 1)

    y = pm.NegativeBinomial('y', mu=pm.math.exp(lam), alpha=alpha, observed=imported)
    

# with mod:
#     idata = pm.sample(3000, init=3000, target_accept=0.95, chains=4, cores=12, 
#                       nuts_sampler='numpyro', random_seed=33)

# az.to_netcdf(idata, "idata_model.nc")

# summ = az.summary(idata, hdi_prob=0.9)
# summ.to_csv("idata_summary.csv")    

# az.plot_trace(idata, kind='rank_vlines', var_names=['l', 'sigma', 'delta_s', 'epsi_s', 'alpha'])
# plt.tight_layout()
# plt.savefig("rank_plots.png", dpi=300)
# plt.close()

# az.plot_energy(idata)
# plt.savefig('energy.png', dpi=300)
# plt.close()

## Load inference data
idata = az.from_netcdf("idata_model.nc")

with mod:
    pred0 = pm.sample_posterior_predictive(idata, var_names=["y",'lam', 'alpha'], random_seed=33)
pred_y0 = az.extract(pred0, group="posterior_predictive")['y'].values


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

Eu = len(df_unobs.country.unique())
Ru = len(df_unobs.reporting_country.unique())

x_u = df_unobs['travellers'].values 
x_u_z = (x_u - x_u.mean())/x_u.std()

imported_u = df_unobs.imported.values

coords = {"term_u":df_unobs.term.unique(),
          "dyad_u":df_unobs.dyad.unique(),
          "country_e_u":df_unobs.infection_country.unique(),
          "country_r_u":df_unobs.reporting_country.unique(),
          "location_u":df_unobs.index.values}

##### Build up Model for predictions
with mod:
    mod.add_coords(coords)
    
    t_u_idx = pm.ConstantData("term_u_idx", term_u_idx, dims="location_u")
    d_u_idx = pm.ConstantData("dyad_u_idx", dyad_u_idx, dims="location_u")
    e_u_idx = pm.ConstantData("ecountry_u_idx", ecountry_u_idx, dims="location_u")
    r_u_idx = pm.ConstantData("rcountry_u_idx", rcountry_u_idx, dims="location_u")
    T_u = pm.ConstantData("T_u", terms_u, dims="term_u")
    xun = pm.ConstantData("x_u_z", x_u_z, dims="location_u")
    
    tau_u = latent_t.conditional("tau_u", T_u[:,None], dims="term_u")
    
    delta_u_z = pm.Normal("delta_u_z", 0, 1, dims="dyad")
    delta_u = pm.Deterministic("delta_u", delta_l + delta_s*delta_u_z, dims="dyad")
    
    epsi_u_z = pm.Normal("epsi_u_z", 0, 1, dims="country_e_u")
    epsi_u = pm.Deterministic("epsi_u", epsi_l + epsi_s*epsi_u_z, dims="country_e_u")

    lam_u = pm.Deterministic("lam_u", tau_u[t_u_idx] + delta_u[d_u_idx] +
                                     epsi_u[e_u_idx]*xun )
    
    y_u = pm.NegativeBinomial('y_u', mu=pm.math.exp(lam_u), alpha=alpha)
    
    

###### Sample and plot predictions
with mod:
    pred = pm.sample_posterior_predictive(idata, var_names=["y_u", 'lam_u', 'alpha'], predictions=True, random_seed=33)

pred_y_u = az.extract(pred, group="predictions")['y_u'].values


pred_y = np.concatenate([pred_y0, pred_y_u])

#similarity index 
def SI(a,b):
    return 2*np.minimum(a,b.T).sum()/(a.sum() + b.sum())

lam = np.concatenate([az.extract(pred0.posterior_predictive)['lam'].values, az.extract(pred.predictions)['lam_u'].values])
df0['y_m'] = pred_y.mean(axis=1) #pred_y.mean(axis=1)
#df0['y_s'] = pred_y.std(axis=1)**2
df0['y_s'] = ((np.exp(lam)**2 / az.extract(pred0.posterior_predictive)['alpha'].values) + np.exp(lam)).mean(axis=1)

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
df_exp_term = df0[['term', 'country', 'imported', 'y_m']].groupby(["term","country"], as_index=False).sum()
si_exp = [SI(df0[df0.country==c].imported.values, df0[df0.country==c].y_m.values) for c in df0.country.unique()]
sim_exp = pd.DataFrame({'country':df0.country.unique(), 'SI':si_exp})
sim_exp = pd.merge(df_country, sim_exp, on='country', how='right')
sim_exp.to_csv("similarity_exporting.csv", index=False)
df_country = df_country.sort_values('imported', ascending=False)
df_country = df_country[df_country.imported > 10]


df_rep_term = df0[['term', 'reporting_country', 'imported', 'y_m']].groupby(["term","reporting_country"], as_index=False).sum()
si_rep = [SI(df0[df0.reporting_country==c].imported.values, df0[df0.reporting_country==c].y_m.values) for c in df0.reporting_country.unique()]
sim_rep = pd.DataFrame({'reporting_country':df0.reporting_country.unique(), 'SI':si_rep})
sim_rep = pd.merge(df_r_country, sim_rep, on='reporting_country', how='right')
sim_rep.to_csv("similarity_reporting.csv", index=False)

fig, ax = plt.subplots(3,1, figsize=(22,16))
ax[0].axvspan(28, 39, ymin=0, ymax=1, alpha=0.2, color='grey', label='Unobserved period')
ax[0].plot(np.arange(len(df_term)), df_term.imported, "-o", color='k', lw=3, label="Observed")
ax[0].plot(np.arange(len(df_term)), df_term.y_m, "--o", color='dodgerblue', linestyle="--", lw=3, label="Predicted mean")
ax[0].fill_between(np.arange(len(df_term)), df_term.y_l, df_term.y_u, color='orangered', alpha=0.2, label="SD")
ax[0].plot(np.arange(len(df_term)), np.repeat(np.nan, len(df_term)), color='w', label="SI: "+str(sit))
ax[0].legend(prop={'size': 20}, loc='upper left')
ax[0].text(-0.07, 1.2, "C", size=35, transform=ax[0].transAxes)
ax[0].grid(alpha=0.3)
ax[0].set_ylabel("Count")
ax[0].set_xticks(np.arange(len(df_term)), df_term.term, rotation=90)
ax[0].set_xlabel("Year-Quarter", labelpad=10)
ax[0].set_title("Chikungunya importations aggregated by quarter")
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_ylim(ymin=0, ymax=1000)
ax[0].set_xlim(xmin=-0.5, xmax=len(df_term))
ax[1].plot(np.arange(len(df_r_country)), df_r_country.imported, "o", color='k', ms=12, label="Observed")
ax[1].plot(np.arange(len(df_r_country)), df_r_country.y_m, "o", color='crimson', ms=12, label="Predicted mean")
ax[1].vlines(np.arange(len(df_r_country)), df_r_country.y_l, df_r_country.y_u, color='crimson', linewidth=4, alpha=0.7, label="SD")
ax[1].plot(np.arange(len(df_term)), np.repeat(np.nan, len(df_term)), color='w', label="SI: "+str(sic))
for i in range(len(df_r_country)):
    ax[1].text(np.arange(len(df_r_country))[i]-0.1, df_r_country.y_u.values[i]+500, str(sim_rep.SI.values.round(2)[i]), size=18)
ax[1].legend(prop={'size': 20})
ax[1].text(-0.07, 1.2, "D", size=35, transform=ax[1].transAxes)
ax[1].grid(alpha=0.3)
ax[1].set_xticks(np.arange(len(df_r_country)), df_r_country.reporting_country)
ax[1].set_xlabel("Country")
ax[1].set_ylabel("Count")
ax[1].set_title("Chikungunya importations aggregated by reporting country")
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_ylim(ymin=0, ymax=2500)
ax[1].set_xlim(xmin=-0.5, xmax=len(df_r_country))
ax[2].plot(np.arange(len(df_country)), df_country.imported, "o", color='k', ms=12, label="Observed")
ax[2].plot(np.arange(len(df_country)), df_country.y_m, "o", color='crimson', ms=12, label="Predicted mean")
ax[2].vlines(np.arange(len(df_country)), df_country.y_l, df_country.y_u, color='crimson', linewidth=4, alpha=0.7, label="SD")
ax[2].plot(np.arange(len(df_term)), np.repeat(np.nan, len(df_term)), color='w', label="SI: "+str(sice))
for i in range(len(df_country)):
    ax[2].text(np.arange(len(df_country))[i], df_country.y_u.values[i]+200, str(sim_exp.SI.values.round(1)[i]), size=18)
ax[2].legend(prop={'size': 20})
ax[2].text(-0.07, 1.2, "E", size=35, transform=ax[2].transAxes)
ax[2].grid(alpha=0.3)
ax[2].set_xticks(np.arange(len(df_country)), df_country.country)
ax[2].set_xlabel("Country")
ax[2].set_ylabel("Count")
ax[2].set_title("Chikungunya importations aggregated by exporting country")
ax[2].spines[['right', 'top']].set_visible(False)
ax[2].set_ylim(ymin=0, ymax=1500)
ax[2].set_xlim(xmin=-0.5, xmax=len(df_country))
plt.tight_layout()
plt.subplots_adjust(hspace=1)
plt.savefig("posterior_predictives_extended.png", dpi=300)
plt.savefig("posterior_predictives_extended.pdf", dpi=600)
plt.show()
plt.close()

#####plotting parameters
plt.rcParams.update({'font.size': 32})
plt.rcParams.update({'figure.titlesize': 34})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

fig = plt.figure(figsize=(34, 34))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 2])

ax_top = fig.add_subplot(gs[0, :])
ax_top.axvspan(28, 39, ymin=0, ymax=1, alpha=0.2, color='grey', label='Unobserved period')
ax_top.plot(np.arange(len(df_term)), df_term['imported'], "-o", color='k', lw=3, label="Observed")
ax_top.plot(np.arange(len(df_term)), df_term['y_m'], "--o", color='dodgerblue', linestyle="--", lw=3, label="Predicted mean")
ax_top.fill_between(np.arange(len(df_term)), df_term['y_l'], df_term['y_u'], color='orangered', alpha=0.2, label="SD")
ax_top.plot(np.arange(len(df_term)), np.repeat(np.nan, len(df_term)), color='w', label="SI: " + str(sit))
ax_top.legend(prop={'size': 28})
ax_top.text(-0.04, 1.1, "A", size=35, transform=ax_top.transAxes, weight='bold')
ax_top.grid(alpha=0.3)
ax_top.set_ylabel("Count")
ax_top.set_xticks(np.arange(len(df_term)))
ax_top.set_xticklabels(df_term['term'], rotation=90)
ax_top.set_xlabel("Year-Quarter", labelpad=10)
ax_top.set_title("Chikungunya importations aggregated by quarter")
ax_top.spines[['right', 'top']].set_visible(False)
ax_top.set_ylim(ymin=0, ymax=1000)
ax_top.set_xlim(xmin=-0.5, xmax=len(df_term))

ax_left = fig.add_subplot(gs[1:, 0])
ax_left.plot(df_r_country['imported'], np.arange(len(df_r_country)), "o", color='k', ms=12, label="Observed")
ax_left.plot(df_r_country['y_m'], np.arange(len(df_r_country)), "o", color='crimson', ms=12, label="Predicted mean")
ax_left.hlines(np.arange(len(df_r_country)), df_r_country['y_l'], df_r_country['y_u'], color='crimson', linewidth=4, alpha=0.7, label="SD")
ax_left.plot(np.repeat(np.nan, len(df_term)), np.arange(len(df_term)), color='w', label="SI: " + str(sic))
for i in range(len(df_r_country)):
    ax_left.text(df_r_country['y_u'].values[i] + 100, np.arange(len(df_r_country))[i], str(sim_rep.SI.values.round(2)[i]))
ax_left.legend(prop={'size': 32})
ax_left.text(-0.05, 1.01, "B", size=35, transform=ax_left.transAxes, weight='bold')
ax_left.grid(alpha=0.3)
ax_left.set_yticks(np.arange(len(df_r_country)))
ax_left.set_yticklabels(df_r_country['reporting_country'])
ax_left.set_ylabel("Country")
ax_left.set_xlabel("Count")
ax_left.set_title("Reporting country")
ax_left.spines[['right', 'top']].set_visible(False)
ax_left.set_xlim(0, 6000)
ax_left.set_ylim(-0.5, len(df_r_country) - 0.5)

ax_right = fig.add_subplot(gs[1:, 1])
ax_right.plot(df_country['imported'], np.arange(len(df_country)), "o", color='k', ms=12, label="Observed")
ax_right.plot(df_country['y_m'], np.arange(len(df_country)), "o", color='crimson', ms=12, label="Predicted mean")
ax_right.hlines(np.arange(len(df_country)), df_country['y_l'], df_country['y_u'], color='crimson', linewidth=4, alpha=0.7, label="SD")
ax_right.plot(np.repeat(np.nan, len(df_term)), np.arange(len(df_term)), color='w', label="SI: " + str(sice))
for i in range(len(df_country)):
    ax_right.text(df_country['y_u'].values[i] + 100, np.arange(len(df_country))[i], str(sim_exp.SI.values.round(2)[i]))
ax_right.legend(prop={'size': 32})
ax_right.text(-0.05, 1.01, "C", size=35, transform=ax_right.transAxes, weight='bold')
ax_right.grid(alpha=0.3)
ax_right.set_yticks(np.arange(len(df_country)))
ax_right.set_yticklabels(df_country['country'])
ax_right.set_ylabel("Country")
ax_right.set_xlabel("Count")
ax_right.set_title("Exporting country")
ax_right.spines[['right', 'top']].set_visible(False)
ax_right.set_xlim(0, 4000)
ax_right.set_ylim(-0.5, len(df_country) - 0.5)

plt.tight_layout()
plt.subplots_adjust(hspace=0.7)
plt.savefig("posterior_predictives_extended.png", dpi=300)
plt.savefig("posterior_predictives_extended.pdf", dpi=600)
plt.show()
plt.close()


df0.to_csv("data_summary_all.csv", index=False)

#####plotting parameters
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.titlesize': 24})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"


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


fig, ax = plt.subplots(3, 1, figsize=(14,8))
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[0].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_2019.plot(column='y_m_log', cmap="plasma", ax=ax[0], vmin=0, vmax=vmax)
ax[0].axis("off")
ax[0].set_title("2019", size=16)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1])
hands1 = [Line2D([0],[0], label=lefts2020[i], marker='s', ms=10, mfc=cmap(left_out_2020.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2020))]
# ax[1].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
#leg1.set_title(title="Extraterritorial", prop={'size':8})
ax[1].axis("off")
ax[1].set_title("2020", size=16)
out_eu_2020.plot(column='y_m_log', cmap="plasma", ax=ax[1], vmin=0, vmax=vmax)

plt.rcParams.update({'font.size': 16})
im = ax[2].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma',
                    vmin=0, vmax=vmax, origin="lower")
cbar_ax = fig.add_axes([0.36, 0.01, 0.3, 0.01])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="predicted importations (log)")
plt.rcParams.update({'font.size': 20})

wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[2])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[2])
out_eu_2021.plot(column='y_m_log', cmap="plasma", ax=ax[2], vmin=0, vmax=vmax) 
hands2 = [Line2D([0],[0], label=lefts2021[i], marker='s', ms=10, mfc=cmap(left_out_2021.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2021))]
# ax[2].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
#leg2.set_title(title="Extraterritorial", prop={'size':8})
ax[2].axis("off")
ax[2].set_title("2021", size=16)
fig.suptitle("Observed Importations", y=0.96, size=16)
plt.text(-0.5, 96, "A", size=19)
plt.tight_layout()
plt.savefig("importation_to_europe_observed.png", dpi=650, bbox_inches='tight')
plt.savefig("importation_to_europe_observed.pdf", dpi=650, bbox_inches='tight')


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



fig, ax = plt.subplots(3, 1, figsize=(14,8))
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[0].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_2019.plot(column='y_m_log', cmap="plasma", ax=ax[0], vmin=0, vmax=vmax)
ax[0].axis("off")
ax[0].set_title("2019", size=16)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1])
hands1 = [Line2D([0],[0], label=lefts2020[i], marker='s', ms=10, mfc=cmap(left_out_2020.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2020))]
# ax[1].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
#leg1.set_title(title="Extraterritorial", prop={'size':8})
ax[1].axis("off")
ax[1].set_title("2020", size=16)
out_eu_2020.plot(column='y_m_log', cmap="plasma", ax=ax[1], vmin=0, vmax=vmax)

plt.rcParams.update({'font.size': 16})
im = ax[2].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma',
                    vmin=0, vmax=vmax, origin="lower")
cbar_ax = fig.add_axes([0.36, 0.01, 0.3, 0.01])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="predicted importations (log)")
plt.rcParams.update({'font.size': 20})

wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[2])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[2])
out_eu_2021.plot(column='y_m_log', cmap="plasma", ax=ax[2], vmin=0, vmax=vmax) 
hands2 = [Line2D([0],[0], label=lefts2021[i], marker='s', ms=10, mfc=cmap(left_out_2021.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2021))]
# ax[2].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 8})
#leg2.set_title(title="Extraterritorial", prop={'size':8})
ax[2].axis("off")
ax[2].set_title("2021", size=16)
fig.suptitle("Predicted Importations", y=0.96, size=16)
plt.text(-0.5, 96, "B", size=19)
plt.tight_layout()
plt.savefig("importation_to_europe_predicted.png", dpi=650, bbox_inches='tight')
plt.savefig("importation_to_europe_predicted.pdf", dpi=650, bbox_inches='tight')

### combine images
from PIL import Image
plots = ["importation_to_europe_observed.png",
         "importation_to_europe_predicted.png",
         "posterior_predictives_extended.png"]

im1,im2,im3 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2+400, im1.height+im3.height), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im1.width+400, 0))
dst.paste(im3, (100, im1.height))

dst.save("calibration_plots.png")

dst.save("calibration_plots.pdf")


####### plot descriptive #######
########################################################################

wmap['country'] = wmap.country.replace("GR", "EL")
no_eu = wmap[~wmap.country.isin(df0.reporting_country.unique())]
eu = wmap[wmap.country.isin(df0.reporting_country.unique())]

out_eu = df0[['infection_country', 'year', 'travellers_ee']]
out_eu['country'] = out_eu["infection_country"].values 
out_eu = out_eu[~out_eu.infection_country.isin(df.reporting_country.unique())]
out_eu = out_eu[['country', 'travellers_ee']].groupby(["country"], as_index=False).mean()
out_eu_tot = pd.merge(wmap, out_eu, how='left', on='country')
out_eu_tot['travellers_ee'] = out_eu_tot.travellers_ee.values / 1e6
vmax_tot = out_eu_tot.travellers_ee.max()

out_eu = df0[['infection_country', 'year', 'imported']]
out_eu['country'] = out_eu["infection_country"].values
out_eu = out_eu[~out_eu.infection_country.isin(df.reporting_country.unique())]
out_eu = out_eu[['country', 'imported']].groupby(["country"], as_index=False).mean()
out_eu_imp = pd.merge(wmap, out_eu, how='left', on='country')
vmax_imp = out_eu_imp.imported.max()

fig, ax = plt.subplots(1, 2, figsize=(14,8))
im0 = ax[0].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='cool', alpha=0.5, origin="lower", vmin=0, vmax=vmax_tot)
im0.set_visible(False)
cbar_ax0 = fig.add_axes([0.05, 0.3, 0.4, 0.02])
cbar0 = fig.colorbar(im0, cax=cbar_ax0, orientation='horizontal')
cbar0.ax.tick_params(labelsize=16) 
cbar0.set_label(label="Count (mill.)", size=18)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[0].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_tot.plot(column='travellers_ee', cmap="cool", alpha=0.5, ax=ax[0], vmin=0, vmax=vmax_tot)
ax[0].axis("off")
ax[0].set_title("Average Total Travellers (approx.)", size=16)
plt.text(0, 1, "A", size=16, transform=ax[0].transAxes)
im1 = ax[1].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma', alpha=0.5, origin="lower", vmin=0, vmax=vmax_imp)
im1.set_visible(False)
cbar_ax1 = fig.add_axes([0.54, 0.3, 0.4, 0.02])
cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
cbar1.ax.tick_params(labelsize=16) 
cbar1.set_label(label="Count", size=18)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[1].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_imp.plot(column='imported', cmap="plasma", alpha=0.5, ax=ax[1])
ax[1].axis("off")
ax[1].set_title("Average Chikungunya Importations", size=16)
plt.text(0, 1, "B", size=16, transform=ax[1].transAxes)
plt.tight_layout()
plt.savefig("descriptive_trav_imp.png", dpi=650, bbox_inches='tight')
plt.savefig("descriptive_trav_imp.pdf", dpi=650, bbox_inches='tight')
plt.show()
plt.close()
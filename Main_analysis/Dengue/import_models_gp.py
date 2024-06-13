# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

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
df0 = pd.read_csv("./data/dengue_clean_data_travel.csv")
df0['year'] = [d[:4] for d in df0.term]
df0 = df0[df0.year > '2011']
df0['infection_country'] = df0.country.values
#df0 = df0.dropna()


df_risk = pd.read_csv("./data/dengue_travel_europe_2015-2019_cleaned.csv")
df_risk = df_risk[df_risk.country.isin(df0.country.unique())]

df0 = df0[df0.country.isin(df_risk.country.unique())]
df0.reset_index(inplace=True, drop=True)

### select time periods to calibrate
# selection = ['MX','BR','VN','MV','PH','CU','LK','ID','IN','TH']
# selection = ['ID','IN','TH']
# df0 = df0[df0.infection_country.isin(selection)]

df = df0.copy()
df.reset_index(inplace=True, drop=True)

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


## Load inference data
idata = az.from_netcdf("idata_model.nc")

with mod:
    pred0 = pm.sample_posterior_predictive(idata, var_names=["y",'lam', 'alpha'], predictions=True, random_seed=33)
pred_y0 = az.extract(pred0, group="predictions")['y'].values



########################### Predictions ##############################
######################################################################

fdata = pd.read_csv("./data/euro_air_passengers.csv")
fdata = fdata[['TIME_PERIOD', 'OBS_VALUE', 'geo']]
fdata.columns = ['year', 'passengers_r', 'reporting_country']
fdata22 = fdata[fdata.year==2022]   

dfu = df0[df0.year=='2019'][['travellers_ee', 'reporting_country', 'cases']].drop_duplicates(subset="reporting_country")
dfd = pd.DataFrame({'dyad':df0.dyad.unique() for i in range(3)})
dfd['reporting_country'] = [d[3:] for d in dfd.dyad]
dfy = pd.merge(dfu, dfd, how='right', on="reporting_country")
termsu = []
for i in range(4):
    if i == 0:
        termsu.append(np.repeat('2022-1st', dfy.shape[0]))
    if i == 1:
        termsu.append(np.repeat('2022-2n', dfy.shape[0]))
    if i == 2:
        termsu.append(np.repeat('2022-3rd', dfy.shape[0]))
    if i == 3:
        termsu.append(np.repeat('2022-4th', dfy.shape[0]))
termsu = np.concatenate(termsu)
df22 = pd.concat([dfy,dfy,dfy,dfy])
df22['term'] = termsu
df22['year'] = [t[:4] for t in df22.term]
df23 = df22.copy()
df23['term'] = [t.replace('2022', '2023') for t in df23.term]
df23['year'] = [t[:4] for t in df23.term]
df24 = df22.copy()
df24['term'] = [t.replace('2022', '2024') for t in df24.term]
df24['year'] = [t[:4] for t in df24.term]
df25 = df22.copy()
df25['term'] = [t.replace('2022', '2025') for t in df25.term]
df25['year'] = [t[:4] for t in df25.term]


fdata22 = fdata22[fdata22.reporting_country.isin(df22.reporting_country.unique())]
fdata22 = fdata22.drop('year', axis=1)
# df22 = df22.drop('passengers_r', axis=1)
df22 = pd.merge(df22, fdata22, on="reporting_country", how='left')

# df23['passengers_r'] = df23['passengers_r'].values*0.95

df_unobs = pd.concat([df22,df23,df24,df25])
df_unobs['infection_country'] = [d[:2] for d in df_unobs.dyad]
df_unobs['country'] = df_unobs.infection_country.values

df_unobs = df_unobs[df_unobs.dyad.isin(df0[df0.year=='2019'].dyad.unique())]

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

x_u = df_unobs['travellers_ee'].values 
x_u_z = (x_u - x_u.mean())/x_u.std()


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
    pred_u = pm.sample_posterior_predictive(idata, var_names=["y_u", 'lam_u'], predictions=True)

pred_y_u = az.extract(pred_u, group="predictions")['y_u'].values

pred_y = np.concatenate([pred_y0, pred_y_u])

df0 = pd.concat([df0, df_unobs])

#similarity index 
def SI(a,b):
    return 2*np.minimum(a,b.T).sum()/(a.sum() + b.sum())

lam = np.concatenate([az.extract(pred0.predictions)['lam'].values, az.extract(pred_u.predictions)['lam_u'].values])
df0['y_m'] = pred_y.mean(axis=1) #pred_y.mean(axis=1)
df0['y_s'] = ((np.exp(lam)**2 / az.extract(pred0.predictions)['alpha'].values) + np.exp(lam)).mean(axis=1)

df_term = df0[['term', 'imported', 'y_m', 'y_s']].groupby("term", as_index=False).sum()
df_term['y_s'] = np.sqrt(df_term.y_s.values) 
df_term['y_l'] = df_term.y_m.values -  df_term.y_s.values
df_term['y_u'] = df_term.y_m.values +  df_term.y_s.values
df_term[['imported']] = df_term[['imported']].replace({0:np.nan})
# sit = SI(df_term.imported.values, df_term.y_m.values).round(2)
#df_term['imported'] = df['imported'].replace({0:np.nan})

df_r_country = df0[['reporting_country', 'imported', 'y_m', 'y_s']].groupby(["reporting_country"], as_index=False, sort=False).sum()
df_r_country['y_s'] = np.sqrt(df_r_country.y_s.values)
df_r_country['y_l'] = df_r_country.y_m.values -  df_r_country.y_s.values
df_r_country['y_u'] = df_r_country.y_m.values +  df_r_country.y_s.values
# sic = SI(df_r_country.imported.values, df_r_country.y_m.values).round(2) 
df_r_country = df_r_country.sort_values('imported', ascending=False)

df_country = df0[['country', 'imported', 'y_m', 'y_s']].groupby(["country"], as_index=False, sort=False).sum()
df_country['y_s'] = np.sqrt(df_country.y_s.values)
df_country['y_l'] = df_country.y_m.values - df_country.y_s.values
df_country['y_u'] = df_country.y_m.values + df_country.y_s.values
# sice = SI(df_country.imported.values, df_country.y_m.values).round(2)
df_country = df_country.sort_values('imported', ascending=False)
df_country = df_country[df_country.imported > 50]

#####plotting parameters
plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'figure.titlesize': 30})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

fig, ax = plt.subplots(3,1, figsize=(24,18))
ax[0].axvspan(39+0.1, len(df_term)-0.1, ymin=0, ymax=1, alpha=0.2, color='grey', label='Unobserved period')
ax[0].plot(np.arange(len(df_term)), df_term.imported, "-o", color='k', lw=3, label="Observed")
ax[0].plot(np.arange(len(df_term)), df_term.y_m, "--o", color='dodgerblue', linestyle="--", lw=3, label="Predicted mean")
ax[0].fill_between(np.arange(len(df_term)), df_term.y_l, df_term.y_u, color='orangered', alpha=0.2, label="SD")
ax[0].legend(loc='upper left', fontsize=20)
ax[0].text(-0.09, 1.2, "B", size=35, transform=ax[0].transAxes)
ax[0].grid(alpha=0.3)
ax[0].set_ylabel("Count")
ax[0].set_xticks(np.arange(len(df_term)), df_term.term, rotation=90, size=24)
ax[0].set_xlabel("Year-Quarter", labelpad=10)
ax[0].set_title("Dengue importations aggregated by quarter")
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_ylim(ymin=0, ymax=1500)
ax[0].set_xlim(xmin=-0.5, xmax=len(df_term))
ax[1].plot(np.arange(len(df_r_country)), df_r_country.imported, "o", color='k', ms=12, label="Observed")
ax[1].plot(np.arange(len(df_r_country)), df_r_country.y_m, "o", color='crimson', ms=12, label="Predicted mean")
ax[1].vlines(np.arange(len(df_r_country)), df_r_country.y_l, df_r_country.y_u, color='crimson', linewidth=4, alpha=0.7, label="SD")
ax[1].legend(fontsize=20)
ax[1].text(-0.09, 1.2, "C", size=35, transform=ax[1].transAxes)
ax[1].grid(alpha=0.3)
ax[1].set_xticks(np.arange(len(df_r_country)), df_r_country.reporting_country)
ax[1].set_xlabel("Country")
ax[1].set_ylabel("Count")
ax[1].set_title("Dengue importations aggregated by reporting country")
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_ylim(ymin=0, ymax=7000)
ax[1].set_xlim(xmin=-0.5, xmax=len(df_r_country))
ax[2].plot(np.arange(len(df_country)), df_country.imported, "o", color='k', ms=12, label="Observed")
ax[2].plot(np.arange(len(df_country)), df_country.y_m, "o", color='crimson', ms=12, label="Predicted mean")
ax[2].vlines(np.arange(len(df_country)), df_country.y_l, df_country.y_u, color='crimson', linewidth=4, alpha=0.7, label="SD")
ax[2].legend(fontsize=20)
ax[2].text(-0.09, 1.2, "D", size=35, transform=ax[2].transAxes)
ax[2].grid(alpha=0.3)
ax[2].set_xticks(np.arange(len(df_country)), df_country.country)
ax[2].set_xlabel("Country")
ax[2].set_ylabel("Count")
ax[2].set_title("Dengue importations aggregated by exporting country")
ax[2].spines[['right', 'top']].set_visible(False)
ax[2].set_ylim(ymin=0, ymax=5000)
ax[2].set_xlim(xmin=-0.5, xmax=len(df_country))
plt.tight_layout()
plt.subplots_adjust(hspace=1.5)
plt.savefig("posterior_predictives.png", dpi=300)
plt.savefig("posterior_predictives.pdf", dpi=600)
plt.show()
plt.close()

df0.to_csv("data_summary_all.csv", index=False)

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.titlesize': 24})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

################## Plot on Map ########################

###### Plot predictions selected
wmap['country'] = wmap.country.replace("GR", "EL")
no_eu = wmap[~wmap.country.isin(df0.reporting_country.unique())]
eu = wmap[wmap.country.isin(df0.reporting_country.unique())]


out_eu = df0[['infection_country', 'year', 'y_m']]
out_eu['country'] = out_eu["infection_country"].values
out_eu = out_eu[~out_eu.infection_country.isin(df0.reporting_country.unique())]
out_eu = out_eu.groupby(["country","year"], as_index=False).sum()
out_eu['y_m_log'] = np.log(out_eu.y_m.values)
out_eu_2022 = out_eu[out_eu.year=="2022"]
out_eu_2023 = out_eu[out_eu.year=="2023"]
out_eu_2024 = out_eu[out_eu.year=="2024"]
out_eu_2025 = out_eu[out_eu.year=="2025"]

out_eu_2022 = pd.merge(wmap, out_eu_2022, how='left', on='country')
out_eu_2023 = pd.merge(wmap, out_eu_2023, how='left', on='country')
out_eu_2024 = pd.merge(wmap, out_eu_2024, how='left', on='country')
out_eu_2025 = pd.merge(wmap, out_eu_2025, how='left', on='country')

out_eu_2019 = df0[['country', 'year', 'imported']]
out_eu_2019 = out_eu_2019.groupby(["country","year"], as_index=False).sum()
out_eu_2019 = out_eu_2019[out_eu_2019.year=="2019"]
vmax = np.log(out_eu_2019.imported.max()) 
vmax = np.round(vmax, 0) 

cmap = plt.get_cmap('plasma')

left_out = df0[~df0.infection_country.isin(wmap.country.unique())]
left_out['country'] = left_out["infection_country"].values
left_out = left_out[['country', 'year', 'y_m']]
left_out = left_out.groupby(["country","year"], as_index=False).sum()
left_out['y_m_log'] = np.log(left_out.y_m.values)
left_out_2022 = left_out[left_out.year=="2022"]
left_out_2023 = left_out[left_out.year=="2023"]
left_out_2024 = left_out[left_out.year=="2024"]
left_out_2025 = left_out[left_out.year=="2025"]

lefts2022 = left_out_2022.country.values
lefts2023 = left_out_2023.country.values
lefts2024 = left_out_2024.country.values
lefts2025 = left_out_2025.country.values

fig, ax = plt.subplots(2, 2, figsize=(12,6))
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0,0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0,0])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[0,0].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_2022.plot(column='y_m_log', cmap="plasma", ax=ax[0,0], vmin=0, vmax=vmax)
ax[0,0].axis("off")
ax[0,0].set_title("2022", size=16)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0,1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0,1])
hands1 = [Line2D([0],[0], label=lefts2023[i], marker='s', ms=10, mfc=cmap(left_out_2023.y_m_log.values[i]/vmax), mec='w', ls='') 
          for i in range(len(left_out_2023))]
ax[0,1].axis("off")
ax[0,1].set_title("2023", size=16)
out_eu_2023.plot(column='y_m_log', cmap="plasma", ax=ax[0,1], vmin=0, vmax=vmax)
plt.rcParams.update({'font.size': 16})
im = ax[1,1].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma',
                    vmin=0, vmax=vmax, origin="lower")
cbar_ax = fig.add_axes([0.25, 0.01, 0.5, 0.01])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="predicted importations (log)")
plt.rcParams.update({'font.size': 20})
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1,0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1,0])
out_eu_2024.plot(column='y_m_log', cmap="plasma", ax=ax[1,0], vmin=0, vmax=vmax) 
ax[1,0].axis("off")
ax[1,0].set_title("2024", size=16)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1,1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1,1])
out_eu_2025.plot(column='y_m_log', cmap="plasma", ax=ax[1,1], vmin=0, vmax=vmax) 
ax[1,1].axis("off")
ax[1,1].set_title("2025", size=16)
plt.text(-0.01, 1.2, "A", size=18, transform=ax[0,0].transAxes)
fig.suptitle("Predicted Dengue importations to Europe", y=0.95, size=18)
plt.tight_layout()
plt.savefig("importation_to_europe_predicted.png", dpi=620, bbox_inches='tight')
plt.savefig("importation_to_europe_predicted.pdf", dpi=620, bbox_inches='tight')


### combine images
from PIL import Image
plots = ["importation_to_europe_predicted.png",
         "posterior_predictives.png"]

im1,im2 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width, im1.height+im2.height), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (100, im1.height))

dst.save("prediction_plots.png")

dst.save("prediction_plots.pdf")




##########################################################
#### Plot Risk
pred_e = pred_y #predicted exported cases
obs_p = df0['travellers_ee'].values #observed approx. total passengers

e_look = dict(zip(df0.country.unique(), range(len(df0.country.unique()))))
e_idx = df0.country.replace(e_look).values #exporting country index

R = pred_e.T*1e5/obs_p #travellers rate of infection per 100000 travellers
Ri = (pred_e.T/(obs_p - pred_e.T))*obs_p #raw risk (see Lee et al, 2021), i.e. ratio of infected and healthy travellers by total travellers 
Rn = Ri/Ri.max() #normalised risk
Rp = Ri.T/Ri.sum(axis=1) #proportion of risk per country

R_mean = np.array([R.T[e_idx==k].sum(axis=0) for k in e_look.values()])
Rp_mean = np.array([Rp[e_idx==k].sum(axis=0) for k in e_look.values()])

rp_5, rp_95 = az.hdi(Rp_mean.T, hdi_prob=0.9).T
r_5, r_95 = az.hdi(R_mean.T, hdi_prob=0.9).T

df_risk = pd.DataFrame({'country':e_look.keys(), 'idx':e_look.values()})
df_risk = df_risk.sort_values('idx')


df_risk['Rp_m'] = Rp_mean.mean(axis=1)
df_risk['Rp_5'] = rp_5
df_risk['Rp_95'] = rp_95

df_risk['R_m'] = R_mean.mean(axis=1)
df_risk['R_5'] = r_5
df_risk['R_95'] = r_95

rph = df_risk[df_risk["Rp_m"] > 0.01] #only countries with over 1% risk
rph = rph.sort_values("Rp_m", ascending=False)
rph.reset_index(inplace=True, drop=True)
names = rph.country.unique()

colors = [mpl.cm.get_cmap('autumn')(x/24) for x in range(len(names))][:len(names)]
fig, ax = plt.subplots(figsize=(10,12))
for i in range(len(rph)):
    ax.plot((rph["Rp_5"][i], rph["Rp_95"][i]), (len(names)-1-i, len(names)-1-i), color='k')
    ax.plot(rph["Rp_m"][i], len(names)-1-i, marker="o", markersize=5, markerfacecolor="w", color='k')
    ax.set_yticks(list(np.flip(np.arange(len(names)))), names, size=26)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Proportional Risk", size=28)
    ax.grid(alpha=0.5)
ax.set_title("Dengue 2012-2025", pad=20, size=30)
ax.tick_params(axis='x', which='major', labelsize=30)
line = mpl.lines.Line2D([], [], color='k', label='90% HDI')
circle = mpl.lines.Line2D([], [], color='w', marker='o', markeredgecolor='k', label='Posterior Predictive Mean')
ax.legend(handles=[circle, line], loc="lower right", fontsize=24)
plt.tight_layout()    
plt.savefig("proportional_risk.png", dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10,12))
for i in range(len(rph)):
    ax.plot((rph["R_5"][i], rph["R_95"][i]), (len(names)-1-i, len(names)-1-i), color='k')
    ax.plot(rph["R_m"][i], len(names)-1-i, marker="o", markersize=5, markerfacecolor="w", color='k')
    ax.set_yticks(list(np.flip(np.arange(len(names)))), names)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Infection Rate per 100,000 passengers")
    ax.grid(alpha=0.5)
    ax.set_title("Countries Infection Rate")
line = mpl.lines.Line2D([], [], color='k', label='95% HDI')
circle = mpl.lines.Line2D([], [], color='w', marker='o', markeredgecolor='k', label='Posterior Mean')
ax.legend(handles=[circle, line])
plt.tight_layout()    
plt.savefig("infection_rate.png", dpi=300)
plt.close()




## save summary and convergence
summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("idata_summary.csv")

lam_pos = np.exp(az.extract(idata.posterior)['lam'].values)
idx_set = set(rcountry_idx) # index_set = {0, 1, 2}
cr_pos = np.array([np.sum(lam_pos[rcountry_idx==k],axis=0) for k in idx_set])
cnames = r_look.keys()
cr_m = cr_pos.mean(axis=1)
cr_s = cr_pos.std(axis=1)
cr_5, cr_95 = az.hdi(cr_pos.T, hdi_prob=0.9).T
summ_cr = pd.DataFrame({'country':cnames, 'mean':cr_m, 'sd':cr_s, 'hdi_5':cr_5, 'hdi_95':cr_95})
summ_cr.to_csv("reporting_country_posterior_summary.csv", index=False)


lam_pos = np.exp(az.extract(idata.posterior)['lam'].values)
idx_set = set(ecountry_idx) # index_set = {0, 1, 2}
cr_pos = np.array([np.sum(lam_pos[ecountry_idx==k],axis=0) for k in idx_set])
cnames = e_look.keys()
cr_m = cr_pos.mean(axis=1)
cr_s = cr_pos.std(axis=1)
cr_5, cr_95 = az.hdi(cr_pos.T, hdi_prob=0.9).T
summ_cr = pd.DataFrame({'country':cnames, 'mean':cr_m, 'sd':cr_s, 'hdi_5':cr_5, 'hdi_95':cr_95})
summ_cr.to_csv("exporting_country_posterior_summary.csv", index=False)

lam_pos = np.exp(az.extract(idata.posterior)['lam'].values)
idx_set = set(dyad_idx) # index_set = {0, 1, 2}
cr_pos = np.array([np.sum(lam_pos[dyad_idx==k],axis=0) for k in idx_set])
cnames = d_look.keys()
cr_m = cr_pos.mean(axis=1)
cr_s = cr_pos.std(axis=1)
cr_5, cr_95 = az.hdi(cr_pos.T, hdi_prob=0.9).T
summ_cr = pd.DataFrame({'country':cnames, 'mean':cr_m, 'sd':cr_s, 'hdi_5':cr_5, 'hdi_95':cr_95})
summ_cr.to_csv("dyad_posterior_summary.csv", index=False)

tau_s = az.extract(idata.posterior)['l'].values
tsm = tau_s.mean()
tss = tau_s.std()
ts5, ts95 = az.hdi(idata, var_names='l', hdi_prob=0.9)['l'].values
del_s = az.extract(idata.posterior)['delta_s'].values
dsm = del_s.mean()
dss = del_s.std()
ds5, ds95 = az.hdi(idata, var_names='delta_s', hdi_prob=0.9)['delta_s'].values
epsi_s = az.extract(idata.posterior)['epsi_s'].values
esm = epsi_s.mean()
ess = epsi_s.std()
es5, es95 = az.hdi(idata, var_names='epsi_s', hdi_prob=0.9)['epsi_s'].values
alp = az.extract(idata.posterior)['alpha'].values
alpm = alp.mean()
alps = alp.std()
alp5, alp95 = az.hdi(idata, var_names='alpha', hdi_prob=0.9)['alpha'].values


names = ['tau length scale', 'delta scale', 'epsi scale', 'alpha']
meas = [tsm, dsm, esm, alpm]
sds = [tss, dss, ess, alps]
h5s = [ts5, ds5, es5, alp5]
h95s = [ts95, ds95, es95, alp95]
scales_summ = pd.DataFrame({'parameter':names, 'mean':meas, 'sd':sds, 'hdi_5%':h5s, 'hdi_95%':h95s})
scales_summ.to_csv("summary_scale_parameters.csv", index=False)

az.plot_energy(idata)
plt.savefig("energy_plot.png", dpi=300)
plt.close()

az.plot_trace(idata, kind='rank_vlines', var_names=['l', 'delta_s', 'epsi_s', 'alpha'])
plt.tight_layout()
plt.savefig("rank_plots.png", dpi=300)
plt.close()



out_eu_2018 = df0[['country', 'year', 'imported', 'y_m', 'y_s']]
out_eu_2018 = out_eu_2018.groupby(["country","year"], as_index=False).sum()
out_eu_2018 = out_eu_2018[out_eu_2018.year=="2018"]

out_eu_2019 = df0[['country', 'year', 'imported', 'y_m', 'y_s']]
out_eu_2019 = out_eu_2019.groupby(["country","year"], as_index=False).sum()
out_eu_2019 = out_eu_2019[out_eu_2019.year=="2019"]

out_eu_2020 = df0[['country', 'year', 'imported', 'y_m', 'y_s']]
out_eu_2020 = out_eu_2020.groupby(["country","year"], as_index=False).sum()
out_eu_2020 = out_eu_2020[out_eu_2020.year=="2020"]

out_eu_2021 = df0[['country', 'year', 'imported', 'y_m', 'y_s']]
out_eu_2021 = out_eu_2021.groupby(["country","year"], as_index=False).sum()
out_eu_2021 = out_eu_2021[out_eu_2021.year=="2021"]

out_eu_2022 = df0[['country', 'year', 'imported', 'y_m', 'y_s']]
out_eu_2022 = out_eu_2022.groupby(["country","year"], as_index=False).sum()
out_eu_2022 = out_eu_2022[out_eu_2022.year=="2022"]

out_eu_2023 = df0[['country', 'year', 'imported', 'y_m', 'y_s']]
out_eu_2023 = out_eu_2023.groupby(["country","year"], as_index=False).sum()
out_eu_2023 = out_eu_2023[out_eu_2023.year=="2023"]

out_eu_2024 = df0[['country', 'year', 'imported', 'y_m', 'y_s']]
out_eu_2024 = out_eu_2024.groupby(["country","year"], as_index=False).sum()
out_eu_2024 = out_eu_2024[out_eu_2024.year=="2024"]

out_eu_2025 = df0[['country', 'year', 'imported', 'y_m', 'y_s']]
out_eu_2025 = out_eu_2025.groupby(["country","year"], as_index=False).sum()
out_eu_2025 = out_eu_2025[out_eu_2025.year=="2025"]

summ_y_c = pd.concat([out_eu_2018,out_eu_2019,out_eu_2020,out_eu_2021,
                      out_eu_2022,out_eu_2023,out_eu_2024,out_eu_2025])
summ_y_c['y_s'] = np.sqrt(summ_y_c['y_s'].values) 
summ_y_c.rename(columns={'y_m':'predicted mean', 'y_s':'predicted SD'}, inplace=True)
summ_y_c[['imported']] = summ_y_c[['imported']].replace({0:'----'})
summ_y_c.to_csv("exporting_summary_predictions.csv", index=False)

# out_eu_25 = out_eu_2025[out_eu_2025.country.isin(out_eu_2018.country.unique())]

out_eu_2018['y_s'] = np.sqrt(out_eu_2018['y_s'].values)
out_eu_2025['y_s'] = np.sqrt(out_eu_2025['y_s'].values) 
out_comp = pd.merge(out_eu_2018, out_eu_2025, on='country', how='left')
out_comp.dropna(inplace=True)
out_comp = out_comp.round(2)
out_comp.to_csv("exported_2018_2025_comparison.csv", index=False)
# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

np.random.seed(33)

#####plotting parameters
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.titlesize': 24})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

## load map
wmap = gpd.read_file("./Dengue/data/wmap/world-administrative-boundaries.shp")
wmap['country'] = wmap.iso_3166_1_

##### Load dengue dataset
df0 = pd.read_csv("./Dengue/data/dengue_clean_data_travel.csv")
df0['year'] = [d[:4] for d in df0.term]
df0 = df0[df0.year > '2011']
df0['infection_country'] = df0.country.values
df0 = df0.dropna()
df0 = df0.sort_values('imported', ascending=False)
df_risk = pd.read_csv("./dengue/data/dengue_travel_europe_2015-2019_cleaned.csv")
df_risk = df_risk[df_risk.country.isin(df0.country.unique())]
df0 = df0[df0.country.isin(df_risk.country.unique())]
df0.reset_index(inplace=True, drop=True)
df_den = df0.copy()

##### Load chikv dataset
df0 = pd.read_csv("./Chik/data/chik_clean_data_travel.csv")
df0['year'] = [d[:4] for d in df0.term]
df0 = df0[df0.year > '2011']
df0['infection_country'] = df0.country.values
df0 = df0.dropna()
df0 = df0.sort_values('imported', ascending=False)
df_risk = pd.read_csv("./Chik/data/dengue_travel_europe_2015-2019_cleaned.csv")
df_risk = df_risk[df_risk.country.isin(df0.country.unique())]
df0 = df0[df0.country.isin(df_risk.country.unique())]
df0.reset_index(inplace=True, drop=True)
df_chik = df0.copy()

##### Load zika dataset
df0 = pd.read_csv("./Zika/data/zika_clean_data_travel.csv")
df0['year'] = [d[:4] for d in df0.term]
df0 = df0[df0.year > '2011']
df0['infection_country'] = df0.country.values
df0 = df0.dropna()
df0 = df0.sort_values('imported', ascending=False)
df_risk = pd.read_csv("./Zika/data/dengue_travel_europe_2015-2019_cleaned.csv")
df_risk = df_risk[df_risk.country.isin(df0.country.unique())]
df0 = df0[df0.country.isin(df_risk.country.unique())]
df0.reset_index(inplace=True, drop=True)
df_zika = df0.copy()


####### plot descriptive #######
########################################################################

wmap['country'] = wmap.country.replace("GR", "EL")
no_eu = wmap[~wmap.country.isin(df0.reporting_country.unique())]
eu = wmap[wmap.country.isin(df0.reporting_country.unique())]

out_eu_den = df_den[['infection_country', 'year', 'travellers_ee']]
out_eu_den['country'] = out_eu_den["infection_country"].values 
out_eu_den = out_eu_den[~out_eu_den.infection_country.isin(df_den.reporting_country.unique())]
out_eu_den = out_eu_den[['country', 'travellers_ee']].groupby(["country"], as_index=False).mean()
out_eu_den_tot = pd.merge(wmap, out_eu_den, how='left', on='country')
out_eu_den_tot['travellers_ee'] = out_eu_den_tot.travellers_ee.values / 1e6
vmax_tot = out_eu_den_tot.travellers_ee.max()

out_eu_den = df_den[['infection_country', 'year', 'imported']]
out_eu_den['country'] = out_eu_den["infection_country"].values
out_eu_den = out_eu_den[~out_eu_den.infection_country.isin(df_den.reporting_country.unique())]
out_eu_den = out_eu_den[['country', 'imported']].groupby(["country"], as_index=False).mean()
out_eu_den_imp = pd.merge(wmap, out_eu_den, how='left', on='country')
vmax_den = out_eu_den_imp.imported.max()

out_eu_chik = df_chik[['infection_country', 'year', 'imported']]
out_eu_chik['country'] = out_eu_chik["infection_country"].values
out_eu_chik = out_eu_chik[~out_eu_chik.infection_country.isin(df_chik.reporting_country.unique())]
out_eu_chik = out_eu_chik[['country', 'imported']].groupby(["country"], as_index=False).mean()
out_eu_chik_imp = pd.merge(wmap, out_eu_chik, how='left', on='country')
vmax_chik = out_eu_chik_imp.imported.max()

out_eu_zika = df_zika[['infection_country', 'year', 'imported']]
out_eu_zika['country'] = out_eu_zika["infection_country"].values
out_eu_zika = out_eu_zika[~out_eu_zika.infection_country.isin(df_zika.reporting_country.unique())]
out_eu_zika = out_eu_zika[['country', 'imported']].groupby(["country"], as_index=False).mean()
out_eu_zika_imp = pd.merge(wmap, out_eu_zika, how='left', on='country')
vmax_zika = out_eu_zika_imp.imported.max()

fig, ax = plt.subplots(2, 2, figsize=(14,8))
plt.subplots_adjust(hspace=0.5)
im0 = ax[0,0].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='cool', alpha=0.5, origin="lower", vmin=0, vmax=vmax_tot)
im0.set_visible(False)
cbar_ax0 = fig.add_axes([0.13, 0.55, 0.3, 0.01])
cbar0 = fig.colorbar(im0, cax=cbar_ax0, orientation='horizontal')
cbar0.ax.tick_params(labelsize=16) 
cbar0.set_label(label="Count (mill.)", size=14)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0,0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0,0])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[0,0].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_den_tot.plot(column='travellers_ee', cmap="cool", alpha=0.5, ax=ax[0,0], vmin=0, vmax=vmax_tot)
ax[0,0].axis("off")
ax[0,0].set_title("Average Total Travellers (approx.)", size=16)
plt.text(0, 1, "A", size=16, transform=ax[0,0].transAxes)
im1 = ax[0,1].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma', alpha=0.5, origin="lower", vmin=0, vmax=vmax_den)
im1.set_visible(False)
cbar_ax1 = fig.add_axes([0.62, 0.55, 0.3, 0.01])
cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
cbar1.ax.tick_params(labelsize=16) 
cbar1.set_label(label="Count", size=14)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[0,1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[0,1])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[0,1].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_den_imp.plot(column='imported', cmap="plasma", alpha=0.5, ax=ax[0,1])
ax[0,1].axis("off")
ax[0,1].set_title("Average Dengue Importations", size=16)
plt.text(0, 1, "B", size=16, transform=ax[0,1].transAxes)
im2 = ax[1,0].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma', alpha=0.5, origin="lower", vmin=0, vmax=vmax_chik)
im2.set_visible(False)
cbar_ax2 = fig.add_axes([0.13, 0.1, 0.3, 0.01])
cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
cbar2.ax.tick_params(labelsize=16) 
cbar2.set_label(label="Count", size=14)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1,0])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1,0])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[1,0].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_chik_imp.plot(column='imported', cmap="plasma", alpha=0.5, ax=ax[1,0], vmin=0, vmax=vmax_chik)
ax[1,0].axis("off")
ax[1,0].set_title("Average Chikungunya Importations", size=16)
plt.text(0, 1, "C", size=16, transform=ax[1,0].transAxes)
im3 = ax[1,1].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma', alpha=0.5, origin="lower", vmin=0, vmax=vmax_zika)
im3.set_visible(False)
cbar_ax3 = fig.add_axes([0.62, 0.1, 0.3, 0.01])
cbar3 = fig.colorbar(im3, cax=cbar_ax3, orientation='horizontal')
cbar3.ax.tick_params(labelsize=16) 
cbar3.set_label(label="Count", size=14)
wmap.plot(color="gainsboro", edgecolor="k", linewidth=0.1, ax=ax[1,1])
eu.plot(color="green", edgecolor="k", linewidth=0.1, ax=ax[1,1])
rep_coun = Line2D([0], [0], label='reporting', marker='s', ms=10, mec='k', mfc='g', ls='')
no_samp = Line2D([0], [0], label='no samples', marker='s', ms=10, mec='k', mfc='gainsboro', ls='')
ax[1,1].legend(handles=[rep_coun, no_samp], loc='lower left', handletextpad=0.1, prop={'size': 11})
out_eu_zika_imp.plot(column='imported', cmap="plasma", alpha=0.5, ax=ax[1,1])
ax[1,1].axis("off")
ax[1,1].set_title("Average Zika Importations", size=16)
plt.text(0, 1, "D", size=16, transform=ax[1,1].transAxes)
plt.tight_layout()
plt.savefig("descriptive_trav_imp.png", dpi=650, bbox_inches='tight')
plt.savefig("descriptive_trav_imp.pdf", dpi=650, bbox_inches='tight')
plt.show()
plt.close()





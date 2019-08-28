# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: altmetrics
#     language: python
#     name: altmetrics
# ---

# # Analysis notebook for: How much research shared on Facebook is hidden from public view?
#
# This notebook produces all results and figures in the article.
#
# Figures are plotted to the *figures/* directory.
#
# In order to re-produce the plots without interacting with the notebook use `python analysis.py`
#
# **Outline**
#
# 1. Coverage
#     1. Comparison of AES, POS, and TW
#     2. Facebook coverage in detail
#     3. Coverage by disciplines
#         1. Disciplinary breakdown of Facebook methods
# 2. Engagement Counts
#     1. Comparison of AES, POS, and TW
#     2. Facebook counts in detail
#     3. Facebook counts by discipline

# +
import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gspread_dataframe import set_with_dataframe
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib_venn import venn2, venn3, venn3_circles
from oauth2client.service_account import ServiceAccountCredentials
from scipy import stats
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

tqdm.pandas()


# -

# The following presents an implementation of _partial log binning_ following Milojević (2010)

# +
def thresh(bin_size):
    x = 1
    while True:
        diff = np.log10(x+1) - np.log10(x)
        if diff < bin_size:
            return x +1
        x = x + 1

def partial_log_binning(data_counts, bin_size=0.1):
    n_bins = 1/bin_size
    binning_threshold = thresh(bin_size)

    log_data = np.log10(data_counts)
    log_index = np.log10(log_data.index)

    logbins = np.linspace(np.log10(binning_threshold)+0.1,
                          np.log10(max(data)),
                          ((np.log10(max(data))-np.log10(binning_threshold)+0.1)//0.1)+1)

    binned_xs = []
    binned_vals = []      
    
    for i in range(1, binning_threshold+1):      
        if i in log_data.index:
            binned_vals.append(log_data.loc[i])
            binned_xs.append(np.log10(i))
    
    for b in logbins:       
            vals = (b-.05 <= log_index) & (log_index < b+.05)
            vs = data_counts[vals]
            if len(vs)>0:
                n = np.ceil(10**(b+.05) - 10**(b-.05))
                if n == 0:
                    continue
                binned_vals.append(np.log10(vs.sum()/n))
                binned_xs.append(b)
    return binned_xs, binned_vals  


# -

# # Initialization

# ## Configuration

# +
# Seaborn styles
sns.set_style("whitegrid")

# Matplotlib figure configuration fonts and figsizes
plt.rcParams.update({
    'font.family':'sans-serif',
    'font.size': 16.0,
    'text.usetex': False,
    'figure.figsize': (11.69,8.27)
})

# Color palette
cm = "Paired"
cp3 = sns.color_palette(cm, 3)
cp10 = sns.color_palette(cm, 10)

# +
### Optional ###
# Set up GSpread connection to push dataframes to Google Spreadsheets
# Instructions can be found at https://gspread.readthedocs.io/en/latest/

push_to_gspread = False

if push_to_gspread:
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-d9fa71152fe8.json', scope)

    gc = gspread.authorize(credentials)
    sh =  gc.open("PLOS Paper - Tables")
# -

# ## Load data

# +
articles_csv = "data/articles.csv"

figs = "figures/"
tables = "tables/"

save_tables = True
save_figs = True
# -

# Load data
articles = pd.read_csv(articles_csv, index_col="doi", parse_dates=['publication_date'])

# ## Helpers
#
# A few variables to quickly access slices/sets of the dataset

# +
# A few useful sets to index various slices of the articles
aes_set = set(articles['AES'].dropna().index.tolist())
aer_set = set(articles['AER'].dropna().index.tolist())
aec_set = set(articles['AEC'].dropna().index.tolist())

pos_set = set(articles['POS'].dropna().index.tolist())
tw_set = set(articles['TW'].dropna().index.tolist())

any_shares = pos_set.union(aes_set)
both_shares = pos_set.intersection(aes_set)
any_engagement = aes_set.union(aer_set).union(aec_set)
# -

# three main metrics
metrics = ['AES', 'POS', 'TW']

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# # Results
# -

# ## Table 3

# +
x = articles[metrics + ['year']].groupby("year").count()
x['All articles'] = articles.groupby("year").count()['title']

df_cross_metrics = x.copy()
df_cross_metrics.loc['All years'] = df_cross_metrics.sum(axis=0)

for i in df_cross_metrics.index.tolist():
    df_cross_metrics.loc[i, metrics] = df_cross_metrics.loc[i, metrics].map(
        lambda x: "{:,} ({:.1f}%)".format(x, 100*x/df_cross_metrics.loc[i, 'All articles'])
    )

df_cross_metrics.index.name = ""

if push_to_gspread:
    wks = sh.worksheet("Coverage - Methods")
    set_with_dataframe(wks, df_cross_metrics.reset_index())

if save_tables:
    df_cross_metrics.to_csv(tables + "table_3_coverage_aes_pos_tw.csv")
    
df_cross_metrics
# -

# ## Figure 2

# +
pdf = articles

total = len(pos_set.union(aes_set).union(tw_set))

v = venn3([tw_set, pos_set, aes_set],
      set_labels=('', '', ''),
      subset_label_formatter=lambda x: "{:,} ({:.1f})".format(x, 100*x/total));

v.get_patch_by_id('100').set_color(cp3[0])
v.get_patch_by_id('010').set_color(cp3[1])
v.get_patch_by_id('001').set_color(cp3[2])

v.get_patch_by_id('110').set_color(np.add(cp3[0], cp3[1])/2)
v.get_patch_by_id('011').set_color(np.add(cp3[1], cp3[2])/2)
v.get_patch_by_id('101').set_color(np.add(cp3[0], cp3[2])/2)

v.get_patch_by_id('111').set_color(np.add(np.add(cp3[1], cp3[0]), cp3[2]) / 3)

for text in v.set_labels:
    text.set_fontsize(10)
for text in v.subset_labels:
    text.set_fontsize(12)

plt.gca().legend(handles=[v.get_patch_by_id('100'), v.get_patch_by_id('010'), v.get_patch_by_id('001')],
                 labels=["TW", "POS", "AES"], prop={'size': 12});

if save_figs:
    plt.savefig(figs + "figure_2_coverage_aes_pos_tw.png", bbox_inches="tight")
# -

# ## Table 4
#

# +
df = articles[['AES', 'POS']].dropna(how="all")
df = pd.concat([df, articles[['year']]], join="inner", axis=1)

x = df.groupby("year").apply(lambda x: sum(~x['AES'].isna() & x['POS'].isna())).to_frame("AES")
x = pd.concat([x, df.groupby("year").apply(lambda x: sum(~x['AES'].isna() & ~x['POS'].isna())).to_frame("AES and POS")], axis=1)
x = pd.concat([x, df.groupby("year").apply(lambda x: sum(x['AES'].isna() & ~x['POS'].isna())).to_frame("POS")], axis=1)

df_overlap_coverage = x.copy()
df_overlap_coverage.loc['All years'] = df_overlap_coverage.sum(axis=0).astype(int)
df_overlap_coverage['Any FB'] = df_overlap_coverage.sum(axis=1)

cols = ["AES", "AES and POS", "POS"]
for i in [2015, 2016, 2017, 'All years']:
    df_overlap_coverage.loc[i, cols] = df_overlap_coverage.loc[i, cols].map(
        lambda x: "{:,} ({:.1f}%)".format(x, 100*x/df_overlap_coverage.loc[i, "Any FB"]))
    
df_overlap_coverage.index.name = ""

if push_to_gspread:
    wks = sh.worksheet("Coverage - Facebook")
    set_with_dataframe(wks, df_overlap_coverage.reset_index())
    
if save_tables:
    df_overlap_coverage.to_csv(tables + "table_4_coverage_facebook_methods.csv")

df_overlap_coverage

# +
# Difference between AES and POS counts
articles['diff'] = articles['AES'] - articles['POS']

# Remove articles in Arts and Humanities
base = articles[~articles.discipline.isin(["Arts", "Humanities"])]
"Removed {} articles in Arts or Humanities".format(articles[articles.discipline.isin(["Arts", "Humanities"])].shape[0])

# +
# Disciplinary analysis is based on the "discipline" column
# Other options: "grand_discipline", "specialty"
col = "discipline"

cov_disciplines = base.groupby(col)[metrics].apply(lambda x: x.count())
cov_disciplines['All articles'] = base.groupby(col)[metrics].size()
cov_disciplines = cov_disciplines.sort_values("All articles", ascending=False)

# Column names + order
cov_disciplines.index.name = "Discipline"
# -

# ## Figure 3

# +
pdf = cov_disciplines[metrics].copy()
pdf = pdf.apply(lambda x: x.map(lambda y: 100*y/cov_disciplines.loc[x.name, "All articles"]), axis=1)
# pdf = pdf.sort_values("AES", ascending=False)

pdf.index = pdf.index.map(lambda x: "{} ({:,})".format(x, cov_disciplines.loc[x, "All articles"]))
pdf = pdf.sort_values(["AES"], ascending=False)
pdf = pdf[['POS', 'AES', "TW"][::-1]]

pdf.plot(kind="barh", colormap=ListedColormap(sns.color_palette("Paired", 3)), width=.65)

h, l = plt.gca().get_legend_handles_labels()
plt.legend(h[::-1], l[::-1])

plt.ylabel("")
plt.grid(False)
plt.grid(True, axis="x", linestyle=":")

ticks = plt.gca().get_xticks()[:-1]
plt.xticks(ticks, ["{}%".format(int(x)) for x in ticks])

sns.despine(top=True, bottom=True, left=True, right=True);

if save_figs:
    plt.savefig(figs + "figure_3_coverage_disciplines.png", bbox_inches="tight")
# -

# ## Appendix A

# +
# Format + Percentages
cov_disc_formatted = cov_disciplines.copy()
cov_disc_formatted.columns = [
    "AES (coverage)",
    "POS (coverage)",
    "TW (coverage)",
    "All articles (percentage)"]

cov_disc_formatted.loc['Total'] = cov_disc_formatted.sum()

cols = cov_disc_formatted.columns[0:3]
for i in cov_disc_formatted.index:
    cov_disc_formatted.loc[i, cols] = cov_disc_formatted.loc[i, cols].map(
        lambda x: "{:,} ({:.1f}%)".format(x, 100*x/cov_disc_formatted.loc[i, "All articles (percentage)"]))

t = cov_disc_formatted["All articles (percentage)"][:-1].sum()
cov_disc_formatted["All articles (percentage)"] = cov_disc_formatted["All articles (percentage)"].map(
    lambda x: "{:,} ({:.1f}%)".format(x, 100*x/t))
    
if push_to_gspread:
    wks = sh.worksheet("Disciplines - Coverage")
    set_with_dataframe(wks, cov_disc_formatted.reset_index())

if save_tables:
    cov_disc_formatted.to_csv(tables + "appendix_A_coverage_disciplines.csv")
    
cov_disc_formatted
# -

# ## Figure 4

# +
a = aes_set.difference(pos_set)
b = aes_set.intersection(pos_set)
c = pos_set.difference(aes_set)

any_fb_counts = base.reindex(aes_set.union(pos_set))[col].value_counts()
any_fb_counts.loc['Total'] = any_fb_counts.sum()

indices = [a, b, c]

dfs = []
for ix, label in enumerate(["Only AES", "Both", "Only POS"]):
    dois = set().union(*indices[ix:ix+1])
    x = base.reindex(dois)[col].value_counts()
    dfs.append(x.to_frame(label))

pdf = pd.concat(dfs, axis=1, sort=False)
pdf.index = pdf.index.map(lambda x: "{} ({:,})".format(x, any_fb_counts.loc[x]))

pdf = pdf.apply(lambda x: x.map(lambda y: 100*y/sum(x)), axis=1)
pdf = pdf.sort_values(by="Only AES")
pdf[['Both', 'Only POS', 'Only AES']].plot(kind="barh", stacked=True, colormap=ListedColormap(sns.color_palette("Paired", 3)[::-1]))

ticks = plt.gca().get_xticks()[:-1]
plt.xticks(ticks, ["{}%".format(int(x)) for x in ticks])

plt.grid(False)
plt.grid(True, axis="x", linestyle=":")
sns.despine(top=True, bottom=True, left=True, right=True)

if save_figs:
    plt.savefig(figs + "figure_4_coverage_facebook_methods.png", bbox_inches="tight")
# -

pdf

any_fb_counts = base.reindex(aes_set.union(pos_set))[col].value_counts()
any_fb_counts.loc['Total'] = any_fb_counts.sum()

# +
a = aes_set.difference(pos_set)
b = aes_set.intersection(pos_set)
c = pos_set.difference(aes_set)

indices = [a, b, c]

dfs = []
for ix, label in enumerate(["Only AES", "Both", "Only POS"]):
    x = base.reindex(indices[ix])[col].value_counts()
    dfs.append(x.to_frame(label))

pdf = pd.concat(dfs, axis=1, sort=False)
pdf.loc['Total'] = pdf.sum()

pdf['Any FB'] = pdf.sum(axis=1)
pdf['Public/Private (%)'] = np.round((pdf['Only POS'] + pdf['Both']) / (pdf['Only AES'] + pdf['Both']) * 100, 1)
pdf['POS/AES (%)'] = np.round(pdf['Only POS'] / (pdf['Only AES']) * 100, 1)

cols = pdf.columns[0:3]
for i in pdf.index:
    pdf.loc[i, cols] = pdf.loc[i, cols].map(
        lambda x: "{:,} ({:.1f}%)".format(x, 100*x/any_fb_counts[i]))

pdf.index.name = "Discipline"

if push_to_gspread:
    wks = sh.worksheet("Disciplines - FB")
    set_with_dataframe(wks, pdf[["Any FB", "Only AES", "Both", "Only POS"]].reset_index())

if save_tables:
    pdf[["Any FB", "Only AES", "Both", "Only POS"]].to_csv(tables + "appendix_B_coverage_facebook_methods.csv")
    
pdf[["Any FB", "Only AES", "Both", "Only POS"]]

# +
binns = {}
alphas = {}

# straight line
def func(x, a, b):
    return a*x + b

for _ in ['AES', 'POS', 'TW']:
    data = articles[_].dropna()
    val_counts = data.value_counts().sort_index()
    
    x, y = partial_log_binning(val_counts, bin_size=0.1)
    
    binns[_] = x,y

    alpha, intercept = curve_fit(func, x, y)[0] # your data x, y to fit
    alphas[_] = np.abs(alpha)
# -

# ## Table 5

# +
as_data = articles['AES'].dropna()
ps_data = articles['POS'].dropna()
tw_data = articles['TW'].dropna()

summary = pd.DataFrame()

# Base stats
summary['Count'] = articles[metrics].count()
summary['Min'] = articles[metrics].min()
summary['Max'] = articles[metrics].max()

# summary['Med'] = articles[metrics].median()
summary['Geom-Mean'] = [stats.gmean(as_data),
                        stats.gmean(ps_data),
                        stats.gmean(tw_data)]

# Fitted powerlaw
summary['α'] = [alphas['AES'], alphas['POS'], alphas['TW']]
summary.index.name = ""

if push_to_gspread:
    wks = sh.worksheet("Volumne - Distributions")
    set_with_dataframe(wks, summary.reset_index().round(2))

if save_tables:
    summary.round(2).to_csv(tables + "table_5_descriptive_stats_distros.csv")
    
summary.round(2)
# -

# ## Table 6

# +
if save_tables:
    articles[metrics].fillna(0).corr("spearman").round(2).to_csv(tables + "table_6_spearman_corr.csv")
    

articles[metrics].fillna(0).corr("spearman").round(2)
# -

# ## Figure 5

# +
fig = plt.figure()
ax = plt.gca()

for ix, (m, (x,y)) in enumerate(list(binns.items())):
    data = {
        'x':x,
        'y':y
    }
    pdf = pd.DataFrame(data=data)
        
    val_counts = articles[m].dropna().value_counts()
    val_counts = np.log10(val_counts)
    val_counts.index = np.log10(val_counts.index)
    
    plt.scatter(x="index", y=m, data=val_counts.reset_index(),
                label="{}: Original".format(m), c=[cp10[2*ix]], alpha=1, s=40, marker="o",
                zorder=1)
    
    sns.regplot(x="x", y="y", data=pdf,
                label="{}: Binned".format(m), marker="1", color=cp10[2*ix+1],
                truncate=True, ci=False,
                scatter_kws={'s':200, 'zorder':2})
    
plt.legend()

plt.ylim(-4,5.1)
# plt.xlim(-.1,4)

xbins = [1, 10, 100, 1000, 10000]
plt.xticks(np.log10(xbins), ["{:,}".format(_) for _ in xbins]);

ybins = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
plt.yticks(np.log10(ybins), ["{:,}".format(_) for _ in ybins]);

min_ticks = [2,3,4,5,6,7,8,9]
min_ticks.extend([10*x for x in min_ticks[-8:]])
min_ticks.extend([10*x for x in min_ticks[-8:]])
min_ticks.extend([10*x for x in min_ticks[-8:]])
min_ticks.extend([10*x for x in min_ticks[-8:]])
min_ticks.extend([10*x for x in min_ticks[-8:]])

ax.tick_params(axis="x", which="both", bottom=True, length=3)
ax.tick_params(axis="y", which="both", bottom=True, length=3)

# locmin = ticker.LogLocator(subs=subs, numticks=200, numdecs=5)
locmin = ticker.FixedLocator(np.log10(min_ticks))

ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

ax.grid(axis="both", which="minor", linestyle=":")

ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# plt.grid(False)

plt.ylabel("Expected articles with engagement count")
plt.xlabel("Engagement counts")

sns.despine(bottom=True, top=True, left=True, right=True, ax=ax)

if save_figs:
    plt.savefig(figs + "figure_5_count_distributions.png", bbox_inches="tight")
# -

print("AES > POS: {}".format(sum(articles['diff']>0)))
print("POS > AES: {}".format(sum(articles['diff']<0)))
print("AES == POS: {}".format(sum(articles['diff']==0)))

# ## Figure 6

# +
pdf = articles[['diff', 'year']].dropna()
pdf['year'] = pdf['year'].astype(str)
pdf = pdf[pdf['diff']!=0]

pdf["Metrics"] = pdf['diff']>0
pdf.Metrics.replace(True, "AES > POS", inplace=True)
pdf.Metrics.replace(False, "POS > AES", inplace=True)

pdf["diff"] = pdf["diff"].abs()

plt.figure()
sns.boxenplot(x="year", y="diff", hue="Metrics", dodge=True, data=pdf, palette=cm, saturation=1)

medians = pdf.groupby(['year', 'Metrics'])['diff'].median().tolist()
nobs = pdf.groupby(['year', 'Metrics'])['diff'].count().tolist()
nobs = ["n: {:,}".format(x) for x in nobs]

pos = [-.2, .2, .8, 1.2, 1.8, 2.2]

for x, y, label in zip(pos, medians, nobs):
    plt.text(x, y+.4, label,
             horizontalalignment='center', color='w', weight='bold')

plt.yscale("log")
yticks = [1,2,3,5,10,100,1000,10000]
plt.yticks(yticks, ["{:,}".format(_) for _ in yticks]);
plt.ylim(1, plt.gca().get_ylim()[1]);
plt.ylabel("Absolute difference between AS and PS counts")
plt.xlabel("")

plt.grid(axis="y", linestyle=":")
plt.grid(False, axis="x")

sns.despine(left=True, right=True, top=True)

if save_figs:
    plt.savefig(figs + "figure_6_letter_value_plot_count_difference.png", bbox_inches="tight")
# -

# ## Figure 7

# +
a = base[base['diff']>0].index
b = base[base['diff']<0].index
c = base[base['diff']==0].index

indices = [a, b, c]

dfs = []
for ix, label in enumerate(["AES > POS", "AES < POS", "AES == POS"]):
    x = base.reindex(indices[ix]).copy()
    x['diff'] = np.abs(x['diff'])
    x = x.melt(id_vars="discipline", value_vars=['diff'], value_name="Absolute difference between AES and POS")
    x['type'] = label
    dfs.append(x)
df = pd.concat(dfs, axis=0, sort=False)

# +
diff_disc_counts = base[~base['diff'].isna()][col].value_counts()

pdf = df.groupby([col, "type"])['variable'].count().to_frame().reset_index()
y = df.groupby([col])['variable'].count()

pdf['%'] = pdf.apply(lambda x: 100*x.variable/y[x[col]], axis=1)

pdf = pdf.pivot(index=col, columns="type")['%']
pdf = pdf[['AES > POS', 'AES == POS', "AES < POS"]]
pdf = pdf.sort_values('AES > POS')

pdf.index = pdf.index.map(lambda x: "{} ({:,})".format(x, diff_disc_counts.loc[x]))

pdf.plot(kind="barh", stacked=True, colormap=ListedColormap(sns.color_palette("Paired", 3)[::-1]))

h, l = plt.gca().get_legend_handles_labels()
plt.legend(h, l)

ticks = plt.gca().get_xticks()[:-1]
plt.xticks(ticks, ["{}%".format(int(x)) for x in ticks])

plt.ylabel("")
plt.xlabel("")

plt.grid(False)
plt.grid(True, axis="x", linestyle=":")
sns.despine(top=True, bottom=True, left=True, right=True)

if save_figs:
    plt.savefig(figs + "figure_7_AES_POS_comparison_disciplines.png", bbox_inches="tight")
# -

# ## Supplemental material

# +
pdf = articles[metrics+["year"]]
pdf = pdf.melt(value_vars=metrics, value_name="Engagement count", id_vars="year", var_name="Metrics")

plt.figure()
sns.boxenplot(x="year", y="Engagement count", hue="Metrics", data=pdf, palette=cm)

plt.yscale("log")

yticks = [1,2,3,5,10,100,1000,10000]
plt.yticks(yticks, ["{:,}".format(_) for _ in yticks]);
plt.xlabel("")
sns.despine(top=True, bottom=True, left=True, right=True)

if save_figs:
    plt.savefig(figs + "supplemental_fig_8_metrics_over_years.png", bbox_inches="tight")

# +
pdf = df.dropna()
pdf = pdf[pdf.type != "AES == POS"]
sort_med = pdf.groupby([col, "type"])['Absolute difference between AES and POS'].median().to_frame("median").groupby(col)['median'].max()
pdf['sort'] = pdf.apply(lambda x: sort_med[x[col]], axis=1)

pdf = pdf.sort_values(["sort", col, "type"])

diff_disc_counts = base[base['diff']>0][col].value_counts()
pdf.disc = pdf[col].map(lambda x: "{} ({})".format(x, diff_disc_counts.loc[x]))


plt.figure(figsize=((11.69,10)))
sns.boxenplot(y=col, x="Absolute difference between AES and POS", hue="type", data=pdf,
              palette=cm, saturation=.9)
plt.xscale("log")

h, l = plt.gca().get_legend_handles_labels()
plt.legend(h, l)

plt.xlim(1, plt.gca().get_xlim()[1])

plt.ylabel("")
# plt.xticks(rotation=45, ha="right");

ticks = [1, 2, 3, 5, 10, 50, 100, 500, 1000, 5000, 15000]
plt.xticks(ticks, ticks)

plt.grid(axis="x", linestyle=":")
plt.grid(False, axis="y")

sns.despine(left=True, right=True, bottom=True, top=True)

if save_figs:
    plt.savefig(figs + "supplemental_fig_9_letter_value_AES_POS_diff.png", bbox_inches="tight")
# -

# # References
#
# Milojević, S. (2010). Power law distributions in information science: Making the case for logarithmic binning. Journal of the American Society for Information Science and Technology, 61(12), 2417–2425. doi: 10/bm7ck6

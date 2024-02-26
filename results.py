"""
Reproducibility code for the 
Attitudinally-positioned European sample dataset

Social Media for Democracy (SoMe4Dem project, Horizon Europe grant No. 101094752).


Jimena Royo-Letelier & Pedro Ramaciotti
médialab Sciences Po

February 2024

Data is available at:

https://doi.org/10.6084/m9.figshare.25288210.v1



"""

import os
import yaml
import pandas as pd

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)

DPI = 300
FONTSIZE = 12
LIMS = [0, 10]

legend_mps = Line2D(
    [0],
    [0],
    label='MPs',
    marker='+',
    markersize=5,
    linewidth=0,
    markeredgecolor='black',
)
legend_parties = Line2D(
    [0],
    [0],
    label='Parties',
    marker='o',
    markersize=5,
    linewidth=0,
    markeredgecolor='black',
    markerfacecolor='white',
)
legend_followers = Line2D(
    [0],
    [0],
    label='Followers',
    marker='h',
    markersize=5,
    linewidth=0,
    markeredgecolor='deepskyblue',
    markerfacecolor="deepskyblue"
)
CUSTOMLEGEND = [legend_mps, legend_parties, legend_followers]

COUNTRIES = [
    'belgium',
    'france',
    'germany',
    'italy',
    'netherlands',
    'poland',
    # 'romania',
    'slovenia',
    'spain',
]

SEED = 187
NBSPLITS = 100

IDEFIG = False
ATTFIG = True
LOGREG = False

STATS = False
SHOW = False

validation_results = np.empty(shape=(2, len(COUNTRIES))).astype(str)


nb_mps_accounts = []
nb_selected_accounts = []

llm_labels_left = []
llm_labels_right = []
llm_labels_populist = []
llm_labels_elite = []

for c, country in enumerate(COUNTRIES):

    print(f"----- {country} -----")

    vizconfig = os.path.join("vizconfigs", f"{country}.yaml")
    with open(vizconfig, "r", encoding='utf-8') as fh:
        vizparams = yaml.load(fh, Loader=yaml.SafeLoader)

    palette = vizparams['palette']

    sources = pd.read_csv(f"{country}_users.csv")
    targets = pd.read_csv(f"{country}_mps.csv")

    nb_mps_accounts.append(len(targets))
    nb_selected_accounts.append(len(sources))

    llm_labels_left.append(
        len(sources.query("labeled_left==1 & labeled_right!=1")))
    llm_labels_right.append(
        len(sources.query("labeled_right==1 & labeled_left!=1")))
    llm_labels_populist.append(
        len(sources.query("labeled_populist==1 & labeled_elite!=1")))
    llm_labels_elite.append(
        len(sources.query("labeled_elite==1 & labeled_populist!=1")))

    # I. Ideological embeddings visualizations
    if IDEFIG:

        nudges = vizparams['ideological']['nudges']
        limits = vizparams['ideological']['limits']
        cbar_rect = vizparams['ideological']['cbar_rect']
        legend_loc = vizparams['ideological']['legend_loc']

        plot_df = pd.concat([
            sources[['delta_1', 'delta_2']],
            targets[['delta_1', 'delta_2']]
            ]) \
            .reset_index() \
            .drop(columns="index") \
            .rename(columns={'delta_1': 'x', 'delta_2': 'y'})

        kwargs = {
            'x': 'x',
            'y': 'y',
            'space': 0,
            'ratio': 10,
            'height': 5,
            'color': "deepskyblue",
            'gridsize': 100,
            'kind': 'hex',
            'data': plot_df,
        }

        g = sns.jointplot(**kwargs)

        ax = g.ax_joint
        texts = []
        for party in palette:

            sample = targets[targets.party == party]
            sample = sample[['delta_1', 'delta_2', 'party']] \
                .rename(columns={'delta_1': 'x', 'delta_2': 'y'})

            if len(sample) == 0:
                continue

            ax.scatter(
                sample['x'],
                sample['y'],
                marker='+',
                s=20,
                alpha=0.5,
                color=palette[party],
                label=party
            )

            mean_group_estimated = sample[['x', 'y']].mean()

            ax.plot(
                mean_group_estimated['x'],
                mean_group_estimated['y'],
                marker='o',
                markeredgecolor='black',
                markeredgewidth=1.0,
                markersize=5,
                color=palette[party],
            )

            text = ax.text(
                mean_group_estimated['x']+nudges[party][0],
                mean_group_estimated['y']+nudges[party][1],
                party.replace("&", ""),
                color='white',
                bbox=dict(
                    boxstyle="round",
                    ec='black',
                    fc=palette[party],
                    alpha=1),
                fontsize=9)
            texts.append(text)

        xl = fr'First latent dimension $\delta_1$'
        yl = fr'Second latent dimension $\delta_2$'
        ax.set_xlabel(xl, fontsize=FONTSIZE)
        ax.set_ylabel(yl, fontsize=FONTSIZE)

        ax.legend(handles=CUSTOMLEGEND, loc=legend_loc)
        ax.tick_params(axis='x', labelsize=FONTSIZE)
        ax.tick_params(axis='x', labelsize=FONTSIZE)

        ax.set_xlim(limits['x'])
        ax.set_ylim(limits['y'])

        cbar_ax = g.fig.add_axes(cbar_rect)
        cbar = plt.colorbar(cax=cbar_ax)

        path = f"{country}_delta_1_vs_delta_2.pdf"
        plt.savefig(path, dpi=DPI)
        print(f"Figure saved at {path}.")

        if SHOW:
            plt.show()


    # II. Ideological embeddings visualizations
    if ATTFIG:

        attparams =  vizparams['attitudinal']['ches2019']['lrgen_vs_antielite_salience']
        nudges = attparams['nudges']
        limits = attparams['limits']
        cbar_rect = attparams['cbar_rect']
        legend_loc = attparams['legend_loc']

        plot_df = pd.concat([
            sources[['left_right', 'antielite']],
            targets[['left_right', 'antielite']]
            ]) \
            .reset_index() \
            .drop(columns="index") \
            .rename(columns={'left_right': 'x', 'antielite': 'y'})

        kwargs = {
            'x': 'x',
            'y': 'y',
            'color': "deepskyblue",
            'space': 2,
            'ratio': 10,
            'height': 5,
            'kind': 'hex',
            'data': plot_df,
        }

        # plot sources and targets embeddings
        g = sns.jointplot(**kwargs)

        ax = g.ax_joint

        # plot square showing CHES limits

        lowlim_x = 0
        upperlim_x = 10
        lowlim_y = 0
        upperlim_y = 10
        A = [lowlim_x, lowlim_x, upperlim_x, upperlim_x, lowlim_x]
        B = [lowlim_y, upperlim_y, upperlim_y, lowlim_y, lowlim_y]
        ax.plot(A, B, color='white', linestyle='-')
        ax.plot(A, B, color='black', linestyle='--')
        txt = ax.text(2, 10.25, f'CHES survey bounds', fontsize=12)
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=2, foreground='w')])

        # plot colored by parties targets attitudinal embeddings
        texts = []

        for party in palette:

            # plot colored by parties target embeddings
            mps_coord_att = targets[targets['party'] == party] \
                .rename(columns={'left_right': 'x', 'antielite': 'y'})

            ax.scatter(
                mps_coord_att['x'],
                mps_coord_att['y'],
                marker='+',
                s=20,
                alpha=0.5,
                color=palette[party],
                label=party
            )

            group_positions = mps_coord_att[['x', 'y']].mean()
            ax.plot(
                group_positions.iloc[0],
                group_positions.iloc[1],
                marker='o',
                markeredgecolor='black',
                markeredgewidth=1.0,
                markersize=5,
                color=palette[party],
        )

            text = ax.text(
                group_positions.iloc[0]+nudges[party][0],
                group_positions.iloc[1]+nudges[party][1],
                party.replace("&", ""),
                color='white',
                bbox=dict(
                    boxstyle="round",
                    ec='black',
                    fc=palette[party],
                    alpha=1),
                fontsize=9)
            texts.append(text)

        ax.set_xlabel('Left – Right', fontsize=FONTSIZE)
        ax.set_ylabel('Anti-elite rhetoric', fontsize=FONTSIZE)

        ax.legend(
            handles=CUSTOMLEGEND,
            loc=legend_loc,
            fontsize=FONTSIZE-2,
            framealpha=0.98
        )

        ax.tick_params(axis='x', labelsize=FONTSIZE)
        ax.tick_params(axis='x', labelsize=FONTSIZE)

        # setting lims
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        cbar_ax = g.fig.add_axes(cbar_rect)
        cbar = plt.colorbar(cax=cbar_ax)

        path = f"{country}_left_right_vs_antielite.pdf"
        plt.savefig(path, dpi=DPI)
        print(f"Figure saved at {path}.")

        if SHOW:
            plt.show()

    # III. Validations
    if LOGREG:

        VALIDATIONS = [
            ('right', 'left', 'left_right', 'Left – Right'),
            ('populist', 'elite', 'antielite', 'Anti-elite rhetoric'),
        ]

        for v, (label1, label2, attdim, dimname) in enumerate(VALIDATIONS):

            attdim1 = sources \
                .query(f"labeled_{label1}==1 & labeled_{label2}!=1") \
                .loc[:, attdim]

            attdim2 = sources \
                .query(f"labeled_{label2}==1 & labeled_{label1}!=1") \
                .loc[:, attdim]

            l1 = len(attdim1)
            l2 = len(attdim2)

            X = np.hstack([
                attdim1.values,
                attdim2.values
                ]).reshape(-1, 1)

            y = np.hstack([np.zeros_like(attdim1), np.ones_like(attdim2)]).ravel()

            # egalize samples
            model = make_pipeline(
                RandomUnderSampler(random_state=SEED),
                LogisticRegression(penalty='l2', C=1e5, class_weight='balanced')
            )

            if not len(X) > NBSPLITS:
                NBSPLITS = min(l1, l2)
                print(
                    f"""Classes size ({l1} and {l2}) are too small,
                    changing number of folds to {NBSPLITS}.""")

            cv_results = cross_validate(
                model, X, y, cv=NBSPLITS, scoring=('precision', 'recall', 'f1'),
                return_train_score=True, return_estimator=True, n_jobs=-1)

            clf_models = [pipe[-1] for pipe in cv_results['estimator']]
            clf_intercept = np.mean([clf.intercept_ for clf in clf_models])
            clf_coef = np.mean([clf.coef_ for clf in clf_models])

            precision = cv_results['train_precision'].mean()
            recall = cv_results['train_recall'].mean()
            f1 = cv_results['train_f1'].mean()
            f1_sdt = cv_results['train_precision'].std()

            validation_results[v, c] = f"{f1:.2f} ± {f1_sdt:.2f}"

            Xplot = np.sort(X.flatten())
            f = expit(Xplot * clf_coef + clf_intercept).ravel()

            if clf_intercept < 0:
                above_threshold = f > 0.5
            else:
                above_threshold = f < 0.5

            X_threshold = Xplot[above_threshold][0]

            custom_legend=[
                #densities
                Line2D([0], [0], color='white', lw=1, alpha=1, label='Users:'),
                Line2D([0], [0], color='blue', marker='o', mew=0, lw=0, alpha=0.5,
                    label=f'Labeled {label1} ({l1})'),
                Line2D([0], [0], color='red', marker='o', mew=0, lw=0, alpha=0.5,
                    label=f'Labeled {label2} ({l2})'),
                #densities
                Line2D([0], [0], color='white', lw=1, alpha=1, label='\nDensities:'),
                Line2D([0], [0], color='blue', alpha=1, label=f'Labeled {label1}'),
                Line2D([0], [0], color='red', alpha=1, label=f'Labeled {label2}\n'),
                Line2D([0], [0], color='white', lw=1, alpha=1, label='\nLogistic Reg.:'),
                Line2D([0], [0], color='k',  alpha=1, label='Model'),
                Line2D([0], [0], color='k',  linestyle=':', alpha=1,
                    label='Classification'),
                Line2D([0], [0], color='white', lw=1, alpha=1, label='cuttof'),
            ]

            fig = plt.figure(figsize=(5,  3.3))
            ax = fig.add_subplot(1,  1,  1)

            sns.kdeplot(data=attdim1.to_frame(), x=attdim, color='blue', ax=ax, common_norm=False)
            ax.plot(X[y==0], np.zeros(X[y==0].size), 'o', color='blue', alpha=0.02, ms=5, mew=1)

            sns.kdeplot(data=attdim2.to_frame(), x=attdim, color='red', ax=ax, common_norm=False)
            ax.plot(X[y==1], np.ones(X[y==1].size),  'o', color='red',  alpha=0.02, ms=5, mew=1)

            # logistic
            ax.plot(Xplot, f, color='k')
            ax.axvline(X_threshold, linestyle=':', color='k')
            ax.axhline(0.5, linestyle=':', color='k')
            ax.text(-2.3, 0.42, r'$0.5$', color='gray', fontsize=10)
            ax.text(X_threshold+0.25, -0.18, r'$%.2f$' % (X_threshold), color='gray', fontsize=10)

            # positives & negatives
            ax.text(X_threshold+0.2, 1.1, 'True pos.', color='r', fontsize=9)
            ax.text(X_threshold-3.15, 1.1, 'False neg.', color='r', fontsize=9)
            ax.text(X_threshold+0.2, -0.1, 'False pos.', color='b', fontsize=9)
            ax.text(X_threshold-3.05, -0.1, 'True neg.', color='b', fontsize=9)

            # axis
            ax.set_xlim((-2.5, 15))
            ax.set_ylim((-0.2, 1.2))
            ax.set_xlabel(dimname, fontsize=13)
            ax.set_ylabel('')
            ax.legend(handles=custom_legend, loc='center left', fontsize=8.7, bbox_to_anchor=(1, 0.5))
            ax.set_xticks([0, 2.5, 5, 7.5, 10])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            title = f
            fig.suptitle(
                t=f'{country.capitalize()}: precision=%.3f,  recall=%.3f,  F1=%.3f ' % (precision, recall, f1),
                x=0.5,
                y=0.94
            )
            plt.tight_layout()

            path = f'{country}_{attdim}_validation_f1.pdf'
            plt.savefig(path, dpi=300)
            print(f"Figure saved at {path}.")
            if SHOW:
                plt.show()




# STATS

if STATS:
    accounts_stats = [
        nb_mps_accounts,
        nb_selected_accounts,
    ]
    index = [
        '# mps accounts',
        '# users selected accounts',
    ]
    accounts_stats = pd.DataFrame(
        data=accounts_stats,
        index=index,
        columns=COUNTRIES)
    print(f"Accounts statistics:\n{accounts_stats}")
    path = 'accounts_stats.csv'
    accounts_stats.to_csv(path)
    print(f"Labelling stats saved at {path}.")
    with open("accounts_stats.tex", "w") as tf:
        tf.write(accounts_stats.to_latex())

    labelling_stats = [
        llm_labels_left,
        llm_labels_right,
        llm_labels_populist,
        llm_labels_elite,
    ]
    index = [
        '# labeled left',
        '# labeled right',
        '# labeled populist',
        '# labeled elite',
    ]
    labelling_stats = pd.DataFrame(
        data=labelling_stats,
        index=index,
        columns=COUNTRIES)
    print(f"Labelling statistics:\n{labelling_stats}")
    path = 'labelling_stats.csv'
    labelling_stats.to_csv(path)
    print(f"Labelling stats saved at {path}.")
    with open("labelling_stats.tex", "w") as tf:
        tf.write(labelling_stats.to_latex())



if LOGREG:

    validation_results = pd.DataFrame(
        data=validation_results,
        columns=COUNTRIES,
        index=['Left – Right', 'Anti-elite rhetoric'],
        dtype=str)
    path = 'validation_results.csv'
    validation_results.to_csv(path)
    print(f"Validations results saved at {path}.")
    with open("validation_results.tex", "w") as tf:
        tf.write(validation_results.to_latex())
    print(f"Validation results:\n{validation_results}")


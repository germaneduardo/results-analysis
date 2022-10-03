import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import lines, markers


def get_markers(n):
    """generate combinations of markers and linestyles"""
    l = [
        ll
        for ll in lines.lineStyles.keys()
        if ll is not None and (ll not in ("", "None", " "))
    ]
    m = [
        mm
        for mm in markers.MarkerStyle.markers.keys()
        if mm is not None and (mm not in ("", "None", " "))
    ]
    lms = list(itertools.product(m, l))
    samples = np.random.choice(len(lms), n)
    mm = []
    ll = []
    for idx in samples:
        mm.append(lms[idx][0])
        ll.append(lms[idx][1])
    return (mm, ll)


def read_results(folder, ignore_files, str_to_split_db="AllResults", db_idx=0):
    """read csv files into pandas dataframes"""

    path = pathlib.Path(folder)
    results = pd.DataFrame()
    files = []
    for i in path.glob("**/*.csv"):
        if not i.name in (ignore_files):
            partial = pd.read_csv(i.absolute())
            partial["database"] = i.name.split(str_to_split_db)[db_idx]
            results = pd.concat([results, partial], ignore_index=True)
    return results


def get_best_for(
    g,
    measures=None,
    measure_sort="F1",
    ascending=False,
    factorsModel=None,
    agg=False,
    debug=False,
):
    """Aggregates to get the best results based on the a measure and some factorsmodel"""
    test = g.copy()

    test = test[test["type"] == "test"]
    best = test.groupby(factorsModel)[measures].mean()
    best.sort_values(by=measure_sort, ascending=ascending, inplace=True)
    params = best.head(1).copy().reset_index()[factorsModel].to_dict("records")[0]
    params_train = params.copy()
    params_test = params.copy()
    params_train["type"] = "train"
    params_test["type"] = "test"
    if debug:
        print("parameters", params)
    train = g.loc[(g[list(params_train)] == pd.Series(params_train)).all(axis=1)].copy()
    test = g.loc[(g[list(params_test)] == pd.Series(params_test)).all(axis=1)].copy()
    if agg:
        tr_best = (
            train.groupby(factorsModel)[measures]
            .mean()
            .sort_values(by=measure_sort, ascending=ascending, inplace=False)
        )
        tr_best["type"] = "train"
        if debug:
            print(best.head(1))
            print(tr_best.head(1))
        return pd.concat([best.head(1), tr_best.head(1)])
    else:
        train = train.loc[:, factorsModel + measures + ["k"]]
        train["type"] = "train"
        test = test.loc[:, factorsModel + measures + ["k"]]
        test["type"] = "test"
        if debug:
            print("test \n", test.shape)
            print("train \n", train.shape)
        return pd.concat([train, test], ignore_index=True, axis=0)


def increase_lw(g, lw=2.8):
    """increase the linewidth of all the lines in seaborn plot"""
    for ax in np.ravel(g.axes):
        for l in ax.lines:
            plt.setp(l, linewidth=lw)


"""
OTHER OLDER useful methods
"""

aspect = 1.5
sns.set(font_scale=1.3, style="white")


def get_db_from_ax(ax):
    db_str = ax.title.get_text()
    db = db_str.split(" | ")[0].split("=")[1].strip()
    return db


def draw_reference(g, df_reference, measure, ls={"Train": "--", "Test": "-"}, lw=0.1):
    axs = np.ravel(g.axes)
    for ax in np.ravel(axs):
        db = get_db_from_ax(ax)
        mea_train = df_reference.loc[
            (df_reference.database == db) & (df_reference.type == "Train"), measure
        ]
        mea_test = df_reference.loc[
            (df_reference.database == db) & (df_reference.type == "Test"), measure
        ]
        if len(mea_train) > 0:
            ax.axhline(
                mea_train.values[0],
                ls=ls["Train"],
                color="k",
                lw=lw,
                label="Train",
                alpha=0.9,
            )
            ax.axhline(
                mea_test.values[0],
                ls=ls["Test"],
                color="k",
                lw=lw,
                label="Test",
                alpha=0.9,
            )
        else:
            print("referece for db " + db + " missing")


def get_result_plot(df, measure, cond, df_reference=None):

    if type(measure) == tuple:
        measure_show = measure[0]
        measure_for_best = measure[1]
    else:
        measure_show, measure_for_best = measure, measure

    data = (
        df[cond]
        .groupby(factorsIS)
        .apply(
            lambda g: ut.get_best_for(
                g, measures, measure_sort=measure_for_best, factorsModel=factorsModel
            )
        )
        .reset_index()
    )
    data["is"] = data["one_class_method"]

    g = sns.relplot(
        x="ands",
        y=measure_show,
        hue="lshMEthod",
        style="type",
        aspect=aspect,
        row="database",
        col="is",
        kind="line",
        data=data,
        legend="full",
        markers=True,
        facet_kws={"sharey": "row"},
    )
    if df_reference is not None:
        draw_reference(g, df_reference, measure)
    ut.increase_lw(g)
    plt.show()
    g.savefig(f"report/{measure}.png")
    return g


def compute_metrics(row):
    wtdAcc = 0.7 * row["sensibility"] + 0.3 * row["specificity"]
    balancedAcc = (row["sensibility"] + row["specificity"]) / 2
    mcc_den = (
        (row.tp + row.fp) * (row.tp + row.fn) * (row.tn + row.fp) * (row.tn + row.fn)
    )
    # print(mcc_den)
    mcc = ((row.tp * row.tn) - (row.fp * row.fn)) / np.sqrt(float(mcc_den))
    return pd.Series([wtdAcc, balancedAcc, mcc])


def plot_9_results(
    df_in, df_reference, measure="Gmean", measure_for_best="Gmean", aspect=2, height=8
):

    linewidth = 5
    col_order = [
        "credit_card/\nentropy",
        "credit_card/\ndrop3-boundaries",
        "credit_card/\ndrop3-one",
        "pageblocks/\nentropy",
        "pageblocks/\ndrop3-boundaries",
        "pageblocks/\ndrop3-one",
        # "gateway_credit_card/\nentropy",
        # "gateway_credit_card/\ndrop3-boundaries",
        # "gateway_credit_card/\ndrop3-one",
    ]

    bar_params = {
        "errorbar": ("ci", 95),
        "err_style": "bars",
        "err_kws": {"capsize": 10.0, "elinewidth": linewidth * 1.1},
    }

    common = {
        "data": df_in[df_in.type == "test"],
        "x": "ands",
        "y": measure,
        "style": "lshMEthod",
        "hue": "lshMEthod",
        "col": "db/IS_Method",
        "col_wrap": 3,
        "linewidth": linewidth,
        "col_order": col_order,
        "kind": "line",
        "aspect": aspect,
        "height": height,
        "facet_kws": {"sharey": "row"},
    }

    common.update(bar_params)

    gg = sns.relplot(**common, legend=True)

    leg = gg._legend

    g = sns.relplot(**common, legend=False)

    if df_reference is not None:
        # add reference
        axs = np.ravel(g.axes)
        for ax in np.ravel(axs):
            db = get_db_from_ax(ax).split("/")[0]
            mea_test = df_reference.loc[
                (df_reference.database == db) & (df_reference.type == "test"),
                ["ands", measure],
            ].copy()
            print("reference", db, mea_test.shape)
            mea_test["Baseline"] = "Baseline"
            if len(mea_test) > 0:
                xx = sns.lineplot(
                    data=mea_test,
                    x="ands",
                    y=measure,
                    ax=ax,
                    dashes=[(2, 2)],
                    style="Baseline",
                    palette=[
                        (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)
                    ],
                    legend=False,
                    linewidth=linewidth,
                    hue="Baseline",
                    **bar_params,
                )

        current_labels = [label.get_text() for label in leg.texts] + ["Baseline"]
        # get the line of the reference
        current_handlers = leg.legendHandles + [ax.lines[-4]]
        dict_legend = {
            label: handler for label, handler in zip(current_labels, current_handlers)
        }
        g.fig.subplots_adjust(wspace=0.1, hspace=0.1, top=1.0)

        g.add_legend(
            dict_legend,
            bbox_to_anchor=(0.18, 1.1, 1.0, 0.03),
            loc="upper left",
            adjust_subtitles=True,
            ncol=5,
            mode=None,
            borderaxespad=0.5,
            fontsize=28,
            borderpad=0.8,
        )
        for line in g._legend.get_lines():
            line.set_linewidth(linewidth + 0.5)
    # plt.tight_layout()
    g.set_titles(col_template="{col_name}", fontsize=40)
    g.savefig(
        f"report/{measure}.png",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(g._legend,),
    )
    g.set(xlabel="ANDS")
    return (g, dict_legend)


def friedman_test(*args, alpha=0.05):
    """
    Performs a Friedman ranking test.
    Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group.

    Returns
    -------
    F-value : float
        The computed F-value of the test.
    p-value : float
        The associated p-value from the F-distribution.
    rankings : array_like
        The ranking for each group.
    pivots : array_like
        The pivotal quantities for each group.

    References
    ----------
    M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
    D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """
    k = len(args)
    if k < 2:
        raise ValueError("Less than 2 levels")
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1:
        raise ValueError("Unequal number of samples")

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row)
        # get the ranks
        ranks = [row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2.0 for v in row]
        # print("row", row, "ranks", ranks)
        rankings.append(ranks)

    # average of rankings
    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (k + 1) / (6.0 * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
        (sum(r**2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4))
    )
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    print(f"p={p_value:.6F}")

    if p_value > alpha:
        print("Same distributions (fail to reject H0)")
    else:
        print("Different distributions (reject H0)")

    return iman_davenport, p_value, rankings_avg, rankings_cmp


c_bar_mapping = {"DPF": False, "RHF": False, "RHF+DPF": True}


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    col = data.lsh.unique()[0]
    d = pd.pivot_table(
        data, index=args[1], columns=args[0], values=args[2], aggfunc="first"
    )
    cbar = c_bar_mapping[col]
    sns.heatmap(d, cbar=cbar, **kwargs)

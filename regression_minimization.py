import pandas as pd
import numpy as np
import argparse
from config import *
import statsmodels.formula.api as sm
import statsmodels.api as sm_api
from statsmodels.stats.outliers_influence import variance_inflation_factor


np.random.seed(42)

parser = argparse.ArgumentParser(description="Run regression.")
parser.add_argument('--dataset', type=str, default='amazon-book')
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=900)
parser.add_argument('--characteristics', type=str, nargs='+', default=ALL_ACCEPTED_CHARACTERISTICS_MIN)
args = parser.parse_args()

if args.dataset =='yelp':
    results = pd.read_csv('data/minimized_dataset_all_performances_with_characteristics.tsv', sep='\t')
elif args.dataset =='amazon-book':
    results = pd.read_csv('data/final_enriched_results_book.tsv', sep='\t')
else:
    results = pd.read_csv('data/final_enriched_results_sw.tsv', sep='\t')
results = results[results['dataset_name'] == args.dataset]

characteristics = args.characteristics
models = ['LightGCN', 'UltraGCN', 'GFCF']
metrics = ['Recall']

all_rows = []
for metric in metrics:
    for idx, model in enumerate(models):
        model_data = results[(results['model'] == model) & (results['cutoff'] != 1)].copy()
        model_data[characteristics] = model_data[characteristics].apply(lambda x: (x - x.mean()))
        msk = np.random.rand(len(model_data)) < 0.9
        test_current = model_data[~msk]
        train = model_data[msk]

        X = train[characteristics]
        y = train[metric]

        # ===== Compute VIF for multicollinearity test =====
        X_with_const = sm_api.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(len(X.columns))]
        print(f"\nVIF per il modello {model} ({metric}):")
        print(vif_data)

        # Linear Regression OLS
        formula_str_ml = y.name + ' ~ ' + '+'.join(characteristics)
        model_ml = sm.ols(formula=formula_str_ml, data=train[characteristics+[metric]])
        fitted_ml = model_ml.fit(cov_type='HC1')

        y_test = test_current[metric].values
        y_pred_test = fitted_ml.predict(test_current[characteristics])
        mse_test = np.mean((y_test - y_pred_test) ** 2)
        rmse_test = np.sqrt(mse_test)
        mae_test = np.mean(np.abs((y_test - y_pred_test)))

        row = {
            'dataset': args.dataset,
            'metric': metric,
            'model': model,
            'score': fitted_ml.rsquared,
            'adjusted_score': fitted_ml.rsquared_adj,
            'mse_test': mse_test,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
        }
        # Coefficients and p-values
        row.update(fitted_ml.params.to_dict())
        row.update(fitted_ml.pvalues.rename(lambda x: 'p_' + x).to_dict())
        all_rows.append(row)

df = pd.DataFrame(all_rows)
print(df)

df = pd.DataFrame(all_rows)
print(df)

value_cols = [c for c in df.columns if c not in ["dataset", "metric", "model"]]


result = {}
for dataset, dfg in df.groupby("dataset"):
    pivot = dfg.set_index("model")[value_cols].T
    result[dataset] = pivot


amazon_book_df = result[args.dataset]





def significance_symbol(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def make_latex_rows(df, dataset):

    dfg = df[df["dataset"] == dataset]

    value_cols = [c for c in df.columns if c not in ["dataset", "metric", "model"] and not c.startswith("p_")]

    pivot = dfg.set_index("model")[value_cols].T

    rows = []
    for metric in value_cols:

        metric_clean = metric.replace("_", " ")
        row = [metric_clean]
        for model in pivot.columns:
            val = pivot.loc[metric, model]

            p_col = f"p_{metric}"
            if p_col in dfg.columns:
                p_val = dfg.set_index("model").loc[model, p_col]
                sig = significance_symbol(p_val)
            else:
                sig = ""
            row.append(f"{val:.4f}{sig}")
        rows.append(" & ".join(row) + " \\\\")

    return rows



latex_rows = make_latex_rows(df, args.dataset)
for r in latex_rows:
    print(r)

pass
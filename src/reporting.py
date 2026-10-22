"""Generate figures exclusively from the current held-out-well evaluation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def write_figures(metrics, model, feature_names, directory):
    directory.mkdir(exist_ok=True)
    keys=['random_forest','xgboost_smote']
    names=['Random Forest','XGBoost + SMOTE']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax, key, title in zip(axes,['accuracy','f1_weighted','f1_macro'],['Accuracy','F1 weighted','F1 macro']):
        values=[metrics[name][key]*100 for name in keys]
        bars=ax.bar(names,values,color=['#4A90D9','#F94C10'])
        ax.bar_label(bars,fmt='%.2f%%')
        ax.set_ylim(0,100)
        ax.set_title(title)
        ax.tick_params(axis='x',labelsize=8)
    fig.suptitle(f"FORCE 2020 — Holdout: {len(metrics['test_wells'])} unseen wells ({len(metrics['train_wells'])} training wells)")
    fig.tight_layout()
    fig.savefig(directory/'benchmark_comparison.png',dpi=180)
    plt.close(fig)
    importance=model.feature_importances_
    indices=np.argsort(importance)[-15:]
    fig,ax=plt.subplots(figsize=(9,6))
    ax.barh([feature_names[i] for i in indices],importance[indices],color='#F94C10')
    ax.set_title('XGBoost feature importance — training-well preprocessing')
    ax.set_xlabel('Relative importance')
    fig.tight_layout()
    fig.savefig(directory/'feature_importance.png',dpi=180)
    plt.close(fig)

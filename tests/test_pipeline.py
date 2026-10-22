import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import LogPreprocessor, LOG_FEATURES
from train import split_wells, balanced_sample, resample_training
import predict

def logs(n=20):
    return pd.DataFrame({'GR': np.linspace(10, 200, n), 'RHOB': [2.4]*n,
        'NPHI': [.2]*n, 'DTC': [90.]*n, 'RDEP': [5.]*n, 'PEF': [2.]*n,
        'WELL': [f'W{i%5}' for i in range(n)]})

class PipelineTests(unittest.TestCase):
    def test_wells_do_not_overlap(self):
        df=logs(); a,b=split_wells(df)
        self.assertFalse(set(df.iloc[a].WELL) & set(df.iloc[b].WELL))

    def test_preprocessing_is_train_fitted_and_bounded(self):
        train=logs(); p=LogPreprocessor().fit(train)
        before=p.gr_max_; test=logs(1); test['GR']=1e6
        self.assertEqual(p.transform(test).GR_NORM.iloc[0], 1)
        self.assertEqual(p.gr_max_, before)

    def test_imputation_precedes_derivation(self):
        p=LogPreprocessor().fit(logs()); df=logs(1); df['RHOB']=np.nan
        out=p.transform(df)
        self.assertAlmostEqual(out.AI.iloc[0], 2.4*304800/90)
        self.assertAlmostEqual(out.NPHI_RHOB_RATIO.iloc[0], .2/2.4)

    def test_invalid_inputs_rejected(self):
        p=LogPreprocessor().fit(logs())
        for col,value in [('DTC',0),('RHOB',-1),('RDEP',-2),('GR',np.inf),('GR','bad')]:
            df=logs(1); df[col]=value
            with self.subTest(col=col,value=value), self.assertRaises(ValueError): p.transform(df)
        with self.assertRaises(ValueError): p.transform(logs().drop(columns='GR'))
        with self.assertRaises(ValueError): p.transform(logs().iloc[:0])
        with self.assertRaises(ValueError): p.transform(pd.DataFrame([{c:np.nan for c in LOG_FEATURES}]))

    def test_small_sample_keeps_all_rows(self):
        X=logs(); y=np.array([0,1]*10)
        out,target=balanced_sample(X,y)
        self.assertEqual(len(out),20)
        np.testing.assert_array_equal(target,y)

    def test_subsample_preserves_singleton_class(self):
        X=logs(); y=np.array([0]*19+[1])
        sample, target=balanced_sample(X,y,limit=10)
        self.assertEqual(len(sample),10)
        self.assertEqual(set(target),{0,1})

    def test_smote_rare_classes(self):
        X=pd.DataFrame({'x':range(6),'z':range(6)})
        for y in [np.array([0,0,0,0,0,1]),np.array([0,0,0,0,1,1])]:
            out,target=resample_training(X,y)
            self.assertEqual(pd.Series(target).value_counts().nunique(),1)

    def test_bundle_roundtrip_and_batch_single_agree(self):
        frame=logs(); p=LogPreprocessor().fit(frame)
        le=LabelEncoder().fit(['Shale','Sandstone'])
        model=RandomForestClassifier(n_estimators=5,random_state=42).fit(p.transform(frame),np.array([0,1]*10))
        bundle={'schema_version':2,'model':model,'preprocessor':p,'label_encoder':le}
        with tempfile.TemporaryDirectory() as directory:
            joblib.dump(bundle,Path(directory)/'lithology_bundle.joblib')
            with patch.object(predict,'MODEL_DIR',Path(directory)):
                predict.load_model.cache_clear()
                query=logs(2); query['RHOB']=3.1
                with patch.object(joblib,'load',wraps=joblib.load) as loader:
                    batch=predict.predict_batch(query)
                    single=predict.predict_lithology(query.iloc[0].to_dict())
                    self.assertEqual(loader.call_count,1)
                self.assertEqual(batch[0],single)
                self.assertIn(batch[0]['lithology'],le.classes_)
                self.assertTrue(batch[0]['warnings'])
                self.assertAlmostEqual(sum(batch[0]['probabilities'].values()),1.)
                expected=model.predict_proba(p.transform(query))[0].max()
                self.assertEqual(batch[0]['confidence'],expected)
        predict.load_model.cache_clear()

    def test_training_cli_writes_usable_bundle_and_metrics(self):
        import contextlib
        import io
        import json
        import train
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            df=logs(100)
            df['FORCE_2020_LITHOFACIES_LITHOLOGY']=[30000,65000]*50
            data=root/'train.csv'
            df.to_csv(data,sep=';',index=False)
            with patch.object(train,'ROOT',root), patch.object(sys,'argv',['train.py','--data',str(data)]), contextlib.redirect_stdout(io.StringIO()):
                train.main()
            metrics=json.loads((root/'reports/metrics.json').read_text())
            self.assertFalse(set(metrics['train_wells']) & set(metrics['test_wells']))
            bundle=joblib.load(root/'models/lithology_bundle.joblib')
            self.assertEqual(bundle['schema_version'],2)
            self.assertEqual(len(bundle['model'].predict(bundle['preprocessor'].transform(df))),100)

    def test_legacy_model_fails_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(predict,'MODEL_DIR',Path(directory)):
            predict.load_model.cache_clear()
            with self.assertRaisesRegex(FileNotFoundError,'src/train.py'): predict.load_model()
        predict.load_model.cache_clear()

if __name__=='__main__': unittest.main()

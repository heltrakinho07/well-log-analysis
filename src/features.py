"""Shared, train-fitted preprocessing for training and inference."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LOG_FEATURES = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI',
                'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS', 'DCAL', 'DRHO',
                'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']
CORE_FEATURES = ['GR', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'PEF']
DERIVED_FEATURES = ['GR_NORM', 'AI', 'NPHI_RHOB_RATIO', 'RES_DIFF',
                    'LOG_RDEP', 'DTS_DTC_RATIO', 'DCAL_COMPUTED']

class LogPreprocessor(BaseEstimator, TransformerMixin):
    def _numeric(self, X):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError('Forneça uma tabela não vazia de well logs.')
        if X.columns.duplicated().any():
            raise ValueError('Existem nomes de colunas duplicados.')
        missing = set(CORE_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f'Logs obrigatórios ausentes: {sorted(missing)}')
        raw = X.reindex(columns=LOG_FEATURES)
        numeric = raw.apply(pd.to_numeric, errors='coerce')
        if (raw.notna() & numeric.isna()).any().any():
            raise ValueError('Os logs devem conter valores numéricos ou células vazias.')
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError('Os logs não podem conter valores infinitos.')
        if numeric[CORE_FEATURES].isna().all(axis=1).any():
            raise ValueError('Cada amostra deve conter pelo menos um log obrigatório válido.')
        if ((numeric['RHOB'] <= 0) | (numeric['DTC'] <= 0) | (numeric['RDEP'] < 0)).any():
            raise ValueError('RHOB e DTC devem ser positivos; RDEP não pode ser negativo.')
        return numeric

    def fit(self, X, y=None):
        raw = self._numeric(X)
        self.medians_ = raw.median().fillna(0.0)
        absent = raw[CORE_FEATURES].isna().all()
        if absent.any():
            raise ValueError(f'Logs sem dados no treino: {list(absent[absent].index)}')
        filled = raw.fillna(self.medians_)
        self.gr_min_ = float(filled.GR.quantile(.01))
        self.gr_max_ = float(filled.GR.quantile(.99))
        return self

    def transform(self, X):
        if not hasattr(self, 'medians_'):
            raise ValueError('O pré-processador precisa de ser ajustado no treino.')
        df = self._numeric(X).fillna(self.medians_).copy()
        df['GR_NORM'] = ((df.GR - self.gr_min_) / (self.gr_max_ - self.gr_min_ + 1e-8)).clip(0, 1)
        # Acoustic impedance = density * velocity; DTC is slowness in us/ft.
        df['AI'] = df.RHOB * (304800.0 / df.DTC)
        df['NPHI_RHOB_RATIO'] = df.NPHI / df.RHOB
        df['RES_DIFF'] = df.RDEP - df.RSHA
        df['LOG_RDEP'] = np.log1p(df.RDEP)
        df['DTS_DTC_RATIO'] = df.DTS / df.DTC
        df['DCAL_COMPUTED'] = df.CALI - df.BS
        if not np.isfinite(df.to_numpy()).all():
            raise ValueError('O pré-processamento produziu valores não finitos.')
        return df[LOG_FEATURES + DERIVED_FEATURES]

import numpy as np
import pandas as pd
import sklearn as SK
import sklearn.impute as SKI

from ._shared import ImputationBase

MODEL_NAME = "MICE"


class MICEImputation(ImputationBase):
    def MICE_imputation(self, input_data_pd):
        """
        Configure and impute data using the sklearn MICE impute method.

        Mirrors the Core_iHPC nowcasting MICE design while keeping the AQUISTIL
        model-runner wrapper below.
        """
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401

        empty_cols = [col for col in input_data_pd.columns if input_data_pd[col].isna().all()]
        working_data_pd = input_data_pd.drop(columns=empty_cols) if empty_cols else input_data_pd

        if working_data_pd.shape[1] == 0:
            self.logger.warning(
                "MICE received only all-missing columns; filling with zeros".ljust(self.justif - 2, '.') + 'FALLBACK'
            )
            return input_data_pd.fillna(0)

        mice_imputer = SKI.IterativeImputer(
            estimator=SK.linear_model.BayesianRidge(),
            missing_values=np.nan,
            sample_posterior=False,
            max_iter=10,
            tol=0.001,
            n_nearest_features=None,
            initial_strategy="mean",
            imputation_order="ascending",
        )

        imputed_values = mice_imputer.fit_transform(working_data_pd)
        imputed_data_pd = pd.DataFrame(
            imputed_values,
            columns=working_data_pd.columns,
            index=working_data_pd.index,
        )

        if empty_cols:
            self.logger.warning(
                "MICE skipped all-missing columns {cols}; filling with zeros".format(cols=empty_cols).ljust(
                    self.justif - 2, '.') + 'FALLBACK'
            )
            for col in empty_cols:
                imputed_data_pd[col] = 0.0
            imputed_data_pd = imputed_data_pd[input_data_pd.columns]

        return imputed_data_pd

    def impute_frame(self, input_data_pd):
        return self.MICE_imputation(input_data_pd)

    def impute(self, input_data_pd, save_data=False):
        var_to_predict = getattr(self.Configuration, 'var_to_predict', None)
        station_dict = self.extract_station_data(input_data_pd, var_to_predict=var_to_predict)
        station_imputed_dict = {}

        for station, station_df in station_dict.items():
            self.logger.info(''.ljust(self.justif, '-'))
            self.logger.info("Imputing station: {s}".format(s=station).center(self.justif, '|'))
            self.logger.info(''.ljust(self.justif, '-'))
            imputed_station_df = self.MICE_imputation(station_df)
            station_imputed_dict[station] = imputed_station_df
            self.logger.info("MICE Imputation for {s}".format(s=station).ljust(self.justif - 2, '.') + 'OK')
            if save_data:
                self.save_imputed_data(imputed_station_df, station_name=station)

        imputed_data_pd = self.combine_station_data(station_imputed_dict)
        self.logger.info("Combined {n} stations".format(
            n=len(station_imputed_dict)).ljust(self.justif - 2, '.') + 'OK')

        if save_data:
            self.save_imputed_data(imputed_data_pd, station_name=None)

        return imputed_data_pd, station_imputed_dict


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    df = data.copy()
    cols = [c for c in ([target_column] + list(input_columns or [])) if c in df.columns]
    if not cols:
        return df

    numeric = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    imputer = MICEImputation()
    imputed = imputer.MICE_imputation(numeric).reindex(index=df.index, columns=cols)

    for col in cols:
        df[col] = imputed[col]
    return df

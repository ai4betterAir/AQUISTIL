import numpy as np
import pandas as pd
import sklearn as SK
import sklearn.impute as SKI
import time
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from ._shared import ImputationBase

MODEL_NAME = "MICE_PosteriorRefined"


class MICEPosteriorRefinedImputation(ImputationBase):
    def create_lag_lead_features(self, df, variables, lags, leads):
        df_extended = df.copy()

        for var in variables:
            if var not in df.columns:
                self.logger.warning(
                    "Variable {vv} not found in dataframe".format(vv=var).ljust(self.justif - 2, '.') + 'SKIP'
                )
                continue

            for lag in lags:
                df_extended["{var}_lag{lag}".format(var=var, lag=lag)] = df[var].shift(lag)

            for lead in leads:
                df_extended["{var}_lead{lead}".format(var=var, lead=lead)] = df[var].shift(-lead)

        return df_extended

    def impute_frame(self, input_data_pd):
        n_imputations = getattr(self.Configuration, 'temporal_n_imputations', 5)
        max_iter = getattr(self.Configuration, 'temporal_max_iter', 10)
        temporal_lags = getattr(self.Configuration, 'temporal_lags', [])
        temporal_leads = getattr(self.Configuration, 'temporal_leads', [])
        random_state = getattr(self.Configuration, 'temporal_random_state', 42)
        refinement_iterations = getattr(self.Configuration, 'temporal_refinement_iterations', 2)
        n_nearest_features = getattr(self.Configuration, 'temporal_n_nearest_features', 20)

        self.logger.info(''.ljust(self.justif, '-'))
        self.logger.info('MICE_PosteriorRefined Configuration:'.ljust(self.justif, '|'))
        self.logger.info("  n_imputations = {n}".format(n=n_imputations).ljust(self.justif, '|'))
        self.logger.info("  max_iter = {m}".format(m=max_iter).ljust(self.justif, '|'))
        self.logger.info("  lags = {l}".format(l=temporal_lags).ljust(self.justif, '|'))
        self.logger.info("  leads = {l}".format(l=temporal_leads).ljust(self.justif, '|'))
        self.logger.info("  n_nearest_features = {n}".format(n=n_nearest_features).ljust(self.justif, '|'))
        self.logger.info(''.ljust(self.justif, '-'))

        original_columns = input_data_pd.columns.tolist()
        original_missing = input_data_pd.isnull().sum()
        empty_columns = [col for col in original_columns if original_missing[col] == len(input_data_pd)]

        working_input_pd = input_data_pd.drop(columns=empty_columns) if empty_columns else input_data_pd
        if working_input_pd.shape[1] == 0:
            self.logger.warning(
                "MICE_PosteriorRefined received only all-missing columns; filling with zeros".ljust(
                    self.justif - 2, '.') + 'FALLBACK'
            )
            return input_data_pd.fillna(0)

        self.logger.info("Original missing values:".ljust(self.justif, '|'))
        for col in original_columns:
            if original_missing[col] > 0:
                pct = 100.0 * original_missing[col] / len(input_data_pd)
                self.logger.info("  {col}: {n} ({pct:.2f}%)".format(
                    col=col, n=original_missing[col], pct=pct).ljust(self.justif, '|'))

        imputed_datasets = []

        for m in range(n_imputations):
            self.logger.info("Processing imputation set {curr}/{total}".format(
                curr=m + 1, total=n_imputations).ljust(self.justif - 2, '.') + 'RUN')

            df_m = working_input_pd.copy()
            current_random_state = random_state + m

            if temporal_lags or temporal_leads:
                df_m = self.create_lag_lead_features(df_m, original_columns, temporal_lags, temporal_leads)
            all_missing_extended = [col for col in df_m.columns if df_m[col].isna().all()]
            if all_missing_extended:
                self.logger.warning(
                    "MICE_PosteriorRefined skipped all-missing generated columns {cols}".format(
                        cols=all_missing_extended
                    ).ljust(self.justif - 2, '.') + 'FALLBACK'
                )
                df_m = df_m.drop(columns=all_missing_extended)

            self.logger.info("  Extended to {n} columns with temporal features".format(
                n=df_m.shape[1]).ljust(self.justif - 2, '.') + 'OK')

            nearest_features = n_nearest_features
            if nearest_features is not None:
                nearest_features = min(int(nearest_features), max(df_m.shape[1] - 1, 1))

            mice_imputer = SKI.IterativeImputer(
                estimator=SK.linear_model.BayesianRidge(),
                sample_posterior=True,
                missing_values=np.nan,
                max_iter=max_iter,
                tol=0.001,
                n_nearest_features=nearest_features,
                initial_strategy="mean",
                imputation_order="ascending",
                random_state=current_random_state,
            )

            start_time = time.monotonic()
            df_m_values = mice_imputer.fit_transform(df_m)
            df_m = pd.DataFrame(df_m_values, columns=df_m.columns, index=df_m.index)
            self.logger.info(
                "  Initial imputation ({seconds:.1f}s)".format(
                    seconds=time.monotonic() - start_time
                ).ljust(self.justif - 2, '.') + 'OK'
            )

            for refinement in range(refinement_iterations):
                mice_refiner = SKI.IterativeImputer(
                    estimator=SK.linear_model.BayesianRidge(),
                    sample_posterior=True,
                    missing_values=np.nan,
                    max_iter=max_iter,
                    tol=0.001,
                    n_nearest_features=nearest_features,
                    initial_strategy="mean",
                    imputation_order="ascending",
                    random_state=current_random_state + refinement + 1,
                )

                start_time = time.monotonic()
                df_m_values = mice_refiner.fit_transform(df_m)
                df_m = pd.DataFrame(df_m_values, columns=df_m.columns, index=df_m.index)
                self.logger.info(
                    "  Refinement step {step}/{total} ({seconds:.1f}s)".format(
                        step=refinement + 1,
                        total=refinement_iterations,
                        seconds=time.monotonic() - start_time,
                    ).ljust(self.justif - 2, '.') + 'OK'
                )

            self.logger.info("  Refinement ({n} iterations)".format(
                n=refinement_iterations).ljust(self.justif - 2, '.') + 'OK')

            df_m_final = df_m.reindex(columns=original_columns).copy()
            if empty_columns:
                for col in empty_columns:
                    df_m_final[col] = 0.0
                df_m_final = df_m_final[original_columns]
            imputed_datasets.append(df_m_final)

        self.logger.info("Pooling {n} imputation sets".format(
            n=n_imputations).ljust(self.justif - 2, '.') + 'RUN')

        pooled_data = sum(imputed_datasets) / n_imputations
        imputed_data_pd = pd.DataFrame(pooled_data, columns=original_columns, index=input_data_pd.index)

        remaining_missing = imputed_data_pd.isnull().sum().sum()
        if remaining_missing > 0:
            self.logger.warning("Remaining missing values after imputation: {n}".format(
                n=remaining_missing).ljust(self.justif - 2, '.') + 'WARN')
        else:
            self.logger.info("All missing values imputed successfully".ljust(self.justif - 2, '.') + 'OK')

        return imputed_data_pd

    def impute(self, input_data_pd, save_data=False):
        var_to_predict = getattr(self.Configuration, 'var_to_predict', None)
        station_dict = self.extract_station_data(input_data_pd, var_to_predict=var_to_predict)
        station_imputed_dict = {}

        for station, station_df in station_dict.items():
            self.logger.info(''.ljust(self.justif, '-'))
            self.logger.info("Imputing station: {s}".format(s=station).center(self.justif, '|'))
            self.logger.info(''.ljust(self.justif, '-'))
            imputed_station_df = self.impute_frame(station_df)
            station_imputed_dict[station] = imputed_station_df
            self.logger.info("MICE_PosteriorRefined Imputation for {s}".format(s=station).ljust(self.justif - 2, '.') + 'OK')
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
    config = type(
        "Configuration",
        (),
        {
            "temporal_n_imputations": kwargs.get("temporal_n_imputations", 1),
            "temporal_max_iter": kwargs.get("temporal_max_iter", 10),
            "temporal_lags": kwargs.get("temporal_lags", []),
            "temporal_leads": kwargs.get("temporal_leads", []),
            "temporal_random_state": kwargs.get("random_state", 42),
            "temporal_refinement_iterations": kwargs.get("temporal_refinement_iterations", 1),
            "temporal_n_nearest_features": kwargs.get("temporal_n_nearest_features", 20),
        },
    )()
    imputer = MICEPosteriorRefinedImputation(Configuration=config)
    imputed = imputer.impute_frame(numeric)
    imputed = imputed.reindex(index=df.index, columns=cols)
    for col in cols:
        df[col] = imputed[col]
    return df

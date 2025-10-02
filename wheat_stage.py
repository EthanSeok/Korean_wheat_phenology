import pandas as pd
from datetime import datetime
from thermal_time import APSIMWheatPhenology
import os

def calculate_stage(df, stage_div, previous_stage_col, current_stage_tt, current_stage_col, temperature_col):
    stage = df.dropna(subset=[previous_stage_col])
    first_index = stage[previous_stage_col].first_valid_index()

    if first_index is not None:
        stage = df.loc[first_index + 1:].copy()
        stage[f'{current_stage_col}_tt'] = stage[temperature_col].cumsum()

        tt_threshold = stage_div[current_stage_tt]
        stage_exceedance = stage[stage[f'{current_stage_col}_tt'] >= tt_threshold]

        if not stage_exceedance.empty:
            first_exceedance_index = stage_exceedance.index[0]
            previous_day_index = first_exceedance_index - 1 if first_exceedance_index > stage.index[0] else first_exceedance_index

            prev_day_tt = stage.loc[previous_day_index, f'{current_stage_col}_tt']
            next_day_tt = stage_exceedance.iloc[0][f'{current_stage_col}_tt']

            if abs(prev_day_tt - tt_threshold) <= abs(next_day_tt - tt_threshold):
                stage_date = pd.to_datetime(stage.loc[previous_day_index, 'Date'])
            else:
                stage_date = pd.to_datetime(stage_exceedance.iloc[0]['Date'])

            stage_date_doy = stage_date.dayofyear
            df.loc[df['Date'] >= stage_date, current_stage_col] = stage_date_doy
        else:
            df[current_stage_col] = None
    else:
        df[current_stage_col] = None

    return df

def wheat_stage_process(df, stage_div):
    df = calculate_stage(df, stage_div, 'Emergence_date', 'tt_emergence', 'End_of_juvenile_date', 'delta_TT')
    df = calculate_stage(df, stage_div, 'End_of_juvenile_date', 'tt_end_of_juvenile', 'floral_initiation_date', 'delta_TT')
    df = calculate_stage(df, stage_div, 'floral_initiation_date', 'tt_floral_initiation', 'flowering_date', 'Crown temperature (T_c)')
    df = calculate_stage(df, stage_div, 'flowering_date', 'tt_flowering', 'heading_date', 'Crown temperature (T_c)')
    df = calculate_stage(df, stage_div, 'heading_date', 'tt_start_grain_fill', 'end_grain_fill_date', 'Crown temperature (T_c)')
    df = calculate_stage(df, stage_div, 'end_grain_fill_date', 'tt_end_grain_fill', 'maturity_date', 'Crown temperature (T_c)')

    floral_initiation_date = df['floral_initiation_date'].dropna().unique()
    if len(floral_initiation_date) > 0:
        floral_initiation_date = \
            pd.to_datetime(df[df['floral_initiation_date'] == floral_initiation_date[0]]['Date']).values[0]

        df['TT_prime'] = df.apply(
            lambda row: row['delta_TT'] if row['Date'] < floral_initiation_date else row['Crown temperature (T_c)'],
            axis=1)
        df['TT_prime'] = df['TT_prime'].cumsum().round(3)

    maturity_date = df['maturity_date'].dropna().unique()
    if len(maturity_date) > 0:
        maturity_date = pd.to_datetime(df[df['maturity_date'] == maturity_date[0]]['Date']).values[0]
        next_day = maturity_date + pd.Timedelta(days=1)
        df = df[df['Date'] <= next_day]

    return df

def main():
    latitude = 35.7281
    file_path = './input/input_weather.csv'
    daily_data = pd.read_csv(file_path)

    sowing_date = datetime(1976, 11, 5)
    apsim_wheat = APSIMWheatPhenology(R_p=1.5, R_v=1.5, sowing_date=sowing_date)
    results_df = apsim_wheat.accumulate_daily_values(daily_data, latitude)

    stage_div = {
        'tt_emergence': 1,
        'tt_end_of_juvenile': 400.0,
        'tt_floral_initiation': 555.0,
        'tt_flowering': 120.0,
        'tt_start_grain_fill': 545,
        'tt_end_grain_fill': 35,
    }

    result = wheat_stage_process(results_df, stage_div)
    output_path = './output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_filename = 'result.csv'
    result.to_csv(os.path.join(output_path, output_filename), index=False)

if __name__ == '__main__':
    main()

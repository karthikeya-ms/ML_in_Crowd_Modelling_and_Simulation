import pandas as pd


def delay_series(series, delta_n, m):
    """
    Makes a time delay embedding on timeseries data

    :param series: the timeseries data
    :param delta_n: how many time steps the time delay will skip
    :param m: the dimension of the output
    :returns: a dataframe of the time delayed series
    """

    n = len(series)
    delayed_list = []
    header_list = []
    max_delay = (m - 1) * delta_n
    for i in range(m):
        current_delay = i * delta_n
        delay_difference = max_delay - current_delay
        delayed_list.append(
            series[current_delay : n - delay_difference].reset_index(drop=True)
        )
        header_list.append(f"x(t - {current_delay})")

    df = pd.concat(delayed_list, axis=1)
    df.columns = header_list
    return df

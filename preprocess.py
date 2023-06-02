import pandas as pd
import numpy as np


def roll_from_csv(filename):
    df = pd.read_csv(filename)
    df.drop("timestamp", axis=1)

    arr = np.array([window.to_numpy() for window in df.rolling(20, min_periods=20)][19:])

    return arr


def filter_mad(arr):
    vol = np.sum(np.abs(arr), axis=(1,2))
    median = np.median(vol)
    mad = np.median(np.abs(vol - median))

    poi = vol > mad + median

    return arr[poi, :, :]


if __name__ == "__main__":
    ht = roll_from_csv("data/hardtap.csv")
    tot_frames = ht.shape[0]
    ht = filter_mad(ht)
    intresting_frames = ht.shape[0]
    print("TPS:", tot_frames / intresting_frames)

    st = roll_from_csv("data/softtap.csv")
    tot_frames = st.shape[0]
    st = filter_mad(st)
    intresting_frames = st.shape[0]
    print("TPS:", tot_frames / intresting_frames)

    nt = roll_from_csv("data/notap.csv")
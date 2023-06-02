import pandas as pd
import numpy as np


def roll_from_csv(filename):
    df = pd.read_csv(filename)
    df = df.drop("timestamp", axis=1)

    arr = np.array([window.to_numpy() for window in df.rolling(20, min_periods=20)][19:])

    return arr


def filter_mad(arr):
    vol = np.sum(np.abs(arr), axis=(1,2))
    median = np.median(vol)
    mad = np.median(np.abs(vol - median))

    poi = vol > 3*mad + median
    print("left after filtering:", np.sum(poi) / vol.shape[0])

    return arr[poi, :, :]


def export_dataset(no_taps, soft_taps, hard_taps, balance=True):
    if balance:
        no_no_taps = max(soft_taps.shape[0], hard_taps.shape[0])
    else:
        no_no_taps = no_taps.shape[0]

    x = np.concatenate((
        no_taps.reshape((no_taps.shape[0], -1))[:no_no_taps],
        soft_taps.reshape((soft_taps.shape[0], -1)),
        hard_taps.reshape((hard_taps.shape[0], -1))
    ))
    y = np.concatenate((
        np.repeat(0, no_taps.shape[0])[:no_no_taps],
        np.repeat(1, soft_taps.shape[0]),
        np.repeat(2, hard_taps.shape[0])
    ))
    y = np.eye(3)[y]

    print("X shape", x.shape)
    print("Y shape", y.shape)

    assert(x.shape[0] == y.shape[0])

    shuffle = np.arange(x.shape[0])
    np.random.shuffle(shuffle)
    x = x[shuffle]
    y = y[shuffle]

    np.save("data/input.npy", x)
    np.save("data/output.npy", y)


def main():
    ht = roll_from_csv("data/hardtap.csv")
    tot_frames = ht.shape[0]
    ht = filter_mad(ht)
    intresting_frames = ht.shape[0]
    print("hard tap TPS:", intresting_frames / tot_frames )

    st = roll_from_csv("data/softtap.csv")
    tot_frames = st.shape[0]
    st = filter_mad(st)
    intresting_frames = st.shape[0]
    print("soft tap TPS:", intresting_frames / tot_frames )

    nt = roll_from_csv("data/notap.csv")

    export_dataset(nt, st, ht)


if __name__ == "__main__":
    main()
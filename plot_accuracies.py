import os
import time
import pandas as pd
import asciichartpy as ac


CSV_PATH = os.path.join("./logs", "style_probe.csv")


def ensure_csv():
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}. Run style_probe first to generate it.")


def load_df():
    df = pd.read_csv(CSV_PATH)
    # basic sanity fill
    for c in ["acc_john", "acc_ram", "acc_lily", "acc_overall"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def plot_series(title: str, y, height=12):
    if not y:
        print(f"{title}: no data")
        return
    print("\n" + title)
    print(ac.plot(y, {"height": height}))


def main():
    ensure_csv()
    df = load_df()

    # Overall accuracy vs steps
    y_overall = (df["acc_overall"].tolist() if "acc_overall" in df.columns else [])
    plot_series("Overall accuracy (0..1)", y_overall, height=12)
    print(f"Last step: {int(df['step'].iloc[-1]) if 'step' in df.columns else 'n/a'}  acc_overall: {y_overall[-1] if y_overall else 'n/a'}")

    # Pause 10s
    time.sleep(10)

    # Per-user accuracies
    y_j = (df["acc_john"].tolist() if "acc_john" in df.columns else [])
    y_r = (df["acc_ram"].tolist() if "acc_ram" in df.columns else [])
    y_l = (df["acc_lily"].tolist() if "acc_lily" in df.columns else [])
    plot_series("John accuracy (0..1)", y_j, height=8)
    plot_series("Ram accuracy (0..1)", y_r, height=8)
    plot_series("Lily accuracy (0..1)", y_l, height=8)

    print("\nTip: re-run this script to refresh after more training steps.")


if __name__ == "__main__":
    main()



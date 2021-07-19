import numpy as np
import pandas as pd
import requests

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)

df = pd.DataFrame(
    columns=["Player", "Year", "PTS", "stdPTS", "FG%", "avgFG%", "stdFG%", "3P%", "avg3P%", "std3P%", "FT%", "avgFT%",
             "stdFT%", "avgGmSc", "stdGmSc"])
count = 0
for year in range(1979, 2022):
    nb_games = 82
    if year == 1998:
        nb_games = 50
    if year == 2011:
        nb_games = 66
    if year == 2019:
        nb_games = 69
    if year == 2020:
        nb_games = 72
    print("{}".format(year), end=" -> ")
    df_names: pd.DataFrame = pd.read_html(
        "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year), encoding="utf-8")[0]
    df_names.drop(index=df_names[df_names["PTS"] == "PTS"].index, inplace=True)
    df_names = df_names.astype({"Player": str, "PTS": float})
    df_names.drop_duplicates(subset="Player", keep="first", inplace=True)
    df_names.sort_values(by="PTS", ascending=False, inplace=True)
    df_names.drop(index=df_names[df_names["G"].values.astype(int) < int(0.8 * nb_games)].index, inplace=True)
    df_names.replace(regex="\*$", value="", inplace=True)
    for name in df_names.loc[:, "Player"].iloc[:10]:
        name = name.replace("č", "c")
        name = name.replace("ć", "c")
        dict_param = {"a": name.replace("'", "").split(" ")[-1][0].lower(),
                      "b": name.replace("'", "").split(" ")[-1][:5].lower() if len(
                          name.replace("'", "").split(" ")[-1]) > 5 else name.replace("'", "").split(" ")[-1].lower(),
                      "c": name.replace("'", "").split(" ")[0][:2].lower(), "d": year, "e": 1}
        if name in ["Clifford Robinson", "Larry Johnson", "Antoine Walker", "Ray Allen", "Joe Johnson", "Anthony Davis",
                    "Isaiah Thomas"]:
            dict_param["e"] = 2
        if name in ["Josh Smith"]:
            dict_param["e"] = 3
        url = "https://www.basketball-reference.com/players/{a}/{b}{c}0{e}/gamelog/{d}".format(**dict_param)
        r = requests.get(url)
        df_stats: pd.DataFrame = pd.read_html(r.text, encoding="utf-8")[-1]
        df_stats.drop(index=df_stats[df_stats.loc[:, "G"] == "G"].index, inplace=True)
        df_stats["3P%"] = np.where(df_stats["3PA"] == 0, 0., df_stats["3P%"])
        df_stats.drop(
            index=df_stats[df_stats["PTS"].isin(
                ["G", "Did Not Dress", "Not With Team", "Did Not Play", "Player Suspended", "Inactive"])].index,
            inplace=True)
        df_stats = df_stats.loc[:, ["FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "FT%", "PTS", "GmSc"]]
        df_stats = df_stats.astype(
            {"FG": int, "FGA": int, "FG%": float, "3P": int, "3PA": int, "3P%": float, "FT": int, "FTA": int,
             "FT%": float, "PTS": int, "GmSc": float})
        df_describe = df_stats.describe()
        df_describe["FGA"] = np.where(df_describe["FGA"] == 0, np.inf, df_describe["FGA"])
        df_describe["3PA"] = np.where(df_describe["3PA"] == 0, np.inf, df_describe["3PA"])
        df_describe["FTA"] = np.where(df_describe["FGA"] == 0, np.inf, df_describe["FGA"])
        l = [name, "{}-{}".format(year - 1, str(year)[2:]), df_describe.loc["mean", "PTS"],
             df_describe.loc["std", "PTS"],
             df_describe.loc["mean", "FG"] / df_describe.loc["mean", "FGA"] * 100, df_describe.loc["mean", "FG%"] * 100,
             df_describe.loc["std", "FG%"] * 100,
             df_describe.loc["mean", "3P"] / df_describe.loc["mean", "3PA"] * 100, df_describe.loc["mean", "3P%"] * 100,
             df_describe.loc["std", "3P%"] * 100,
             df_describe.loc["mean", "FT"] / df_describe.loc["mean", "FTA"] * 100, df_describe.loc["mean", "FT%"] * 100,
             df_describe.loc["std", "FT%"] * 100,
             df_describe.loc["mean", "GmSc"], df_describe.loc["std", "GmSc"]]
        df.loc[count] = l
        df.replace(np.nan, 0., inplace=True)
        count += 1
    if year % 10 == 0:
        print("")

print("")
print(df.head())
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# !pwd
import sys
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
sys.executable

# # スプラコンペ 勝率検証EDA
# スプラトゥーン2を普段やっているので、ドメイン知識を使ってブキ/ルールごとの勝率の違いや強いと言われているスペシャルなどについて、いくつか検証をしてみました。参考になればと思います。  
# 特徴量として使用して精度が上がるかどうかの検証はしていません。  　　
# また今回のデータは5.0.1環境でのものであり、今回のコンペティションで与えられたデータに限るもので、現環境での分析とは異なるものであることにご留意ください。  
# - https://prob.space/competitions/game_winner/discussions/uratatsu-Post2240d51da94b313ed72c

# +
from collections import Counter
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# +
ORIG_DIR = "../data/orig"

train = pd.read_csv(f'{ORIG_DIR}/train_data.csv')
test = pd.read_csv(f'{ORIG_DIR}/test_data.csv')
#https://prob.space/competitions/game_winner/discussions/e-toppo-Post0082a60376ef134af3a4
buki_data = pd.read_csv(f'{ORIG_DIR}/statink-weapon2.csv')

# カテゴリ名称を変更比較用
display(buki_data.loc[buki_data['category2'] == 'maneuver'].head(3))

# カテゴリ名称とブキ名称が被るためカテゴリ名称を変更する
buki_data.loc[buki_data['category2'] == 'maneuver', 'category2'] = 'maneuver_cat' 
display(buki_data.loc[buki_data['category2'] == 'maneuver_cat'].head(3))

# +
#https://prob.space/competitions/game_winner
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()  # マルチラベル形式でonehot化（複数列で1つく）
mlb.fit([set(train['A1-weapon'].unique())])
MultiLabelBinarizer(classes=None, sparse_output=False)

def trans_weapon(df, columns=['A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon']):
    """指定列をonehot化"""
    weapon = df.fillna('none') 
    weapon_binarized = mlb.transform(weapon[columns].values)
    return pd.DataFrame(weapon_binarized, columns=mlb.classes_)

def make_input_output(df, with_y=False):
    """各武器列をonehot化して結合"""
    a_weapon = trans_weapon(df, ['A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon'])
    b_weapon = trans_weapon(df, ['B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon'])
    a_weapon = a_weapon.add_suffix('_A')
    b_weapon = b_weapon.add_suffix('_B')
    X = pd.concat([a_weapon, b_weapon], axis=1)
    if with_y:
        y = df['y']
        return X, y
    return X

X = make_input_output(train, with_y=False)
train = pd.concat([train, X], axis=1)
train


# -

# # 【検証1】 強いブキ、弱いブキはやっぱりある？？
# まずはブキの勝率から。

# +
def win_rate(buki, df):
    """ブキの勝率"""
    # それぞれのチームで対象ブキが出現する試合数のカウント
    count = len(df[df[buki + "_A"] == 1]) + len(
        df[df[buki + "_B"] == 1]
    )  
    # それぞれのチームで対象ブキが出現する試合のうち各チームが勝利する試合数のカウント
    win = len(df[(df[buki + "_A"] == 1) & (df.y == 1)]) + len(
        (df[(df[buki + "_B"] == 1) & (df.y == 0)])
    )  
    rate = win / count
    return rate, count

if __name__ == '__main__':
    win_rate_df = []
    for buki in buki_data.key.unique():
        #print(buki, win_rate(buki))
        rate, count = win_rate(buki, train)
        win_rate_df.append([buki, rate, count])
    #display(win_rate_df)
    win_rate_df = pd.DataFrame(win_rate_df, columns=['buki', 'win_rate', 'count'])
    win_rate_df = win_rate_df.sort_values(by= 'win_rate', ascending=False)
    buki_ja_dict = buki_data[['key','[ja-JP]']].set_index('key').to_dict()['[ja-JP]']
    win_rate_df["buki_orig"] = win_rate_df.buki
    win_rate_df.buki = win_rate_df.buki.map(buki_ja_dict)
    win_rate_df = win_rate_df.reset_index(drop=True)

win_rate_df.head(20)    
# -

win_rate_df[(win_rate_df['buki'] == 'パラシェルター') | (win_rate_df['buki'] == 'ホクサイ')]

# 全体ではL3リールガンの勝率がトップでした。頭一つ抜けている感じがします。  
# 2位3位にヒーロー種が来ているのが面白いです。  
# 同じ性能のパラシェルターはwin_rate 0.449、ホクサイはwin_rate0.483となっています。  
# ヒーロー種はヒーローモードをある程度やりこまないと手に入らないため、それをあえて使う人はやりこんでいる人だとみなせるかもしれません。  
#
# ※ヒーロー種とはヒーローモードという1人プレイモードで手に入るブキのことを言い、各ブキ種にひとつづつあり、それぞれの無印と見た目が違うだけで、性能やサブ、スペシャルは全く同一になっています。  

win_rate_df.tail(20)


# 勝率最下位はリッター4Kカスタムでした。  
# サブ/スペシャルが長射程ブキと噛み合っていないからでしょうか。あまり使い慣れた人がいない印象はあります。  
# 全体的にチャージャー種やバレル、ブラスターなど扱いが難しいブキが下位ランキングに入っている印象です。  
# 意外だったのはジェットスイーパーカスタムがワースト10に入っていることでした。上手い人たちの対抗戦や大会でもよく見ますし、ホコやヤグラにおいてハイパープレッサーで無双してる印象がありますが、今回のデータの範囲では勝率は高くないようです。  

# +
def win_rate_mode(buki, train):
    """modeごとでの勝率"""
    win_rate_mode_df = pd.DataFrame()
    for mode in train["mode"].unique():
        win_rate_df = []
        train_mode = train[train["mode"] == mode]
        for buki in buki_data.key.unique():
            # print(buki, win_rate(buki))
            rate, count = win_rate(buki, train_mode)
            win_rate_df.append([buki, rate, count])
        win_rate_df = pd.DataFrame(
            win_rate_df, columns=[mode + "_buki", mode + "_win_rate", mode + "_count"]
        )
        win_rate_df = win_rate_df.sort_values(by=mode + "_win_rate", ascending=False)
        buki_ja_dict = (
            buki_data[["key", "[ja-JP]"]].set_index("key").to_dict()["[ja-JP]"]
        )
        win_rate_df[mode + "_buki"] = win_rate_df[mode + "_buki"].map(buki_ja_dict)
        win_rate_df = win_rate_df.reset_index(drop=True)
        return pd.concat([win_rate_mode_df, win_rate_df], axis=1)
    
win_rate_mode(buki, train).head(10)
# -

# 全ルールでトップだったL3リールガンの強さはナワバリバトルのおかげのようです。  
# 勝率6割はヤバいですね。一方のチームにしか含まれない場合などを調べるともっと高いかもしれません。  
# 他のルールは上位に来ているもののサンプル数が少なく、ブキ以外の要因(特定のプレイヤースキルなど)に左右されている可能性があるように思えます。

# ## 【検証3】強いスペシャルはある？？
# インクを一定量塗ると、スペシャルゲージがたまり、スペシャルと呼ばれる特別な攻撃を放つことができます。  
# 単体で相手のイカをキルできるものや、メインブキの攻撃を補助するもの、味方の防御力を高めるものなど様々なものがあります。  
# それぞれルールごとに強いものや弱いものがあると言われています。  

cols = ['A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', 'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon']
for col in cols:
    train = train.merge(buki_data[['key','subweapon', 'special']], left_on=col, right_on='key', how='left').drop(columns='key').rename(columns={'subweapon':'subweapon' + '_' + col, 'special':'special' + '_' + col})

# +
from collections import Counter

def special_count(df):
    df_A = df[['special_A1-weapon', 'special_A2-weapon', 'special_A3-weapon', 'special_A4-weapon']].values
    df_B = df[['special_B1-weapon', 'special_B2-weapon', 'special_B3-weapon', 'special_B4-weapon']].values
    A_special = pd.DataFrame([Counter(x) for x in df_A]).fillna(0).astype('int8')
    B_special = pd.DataFrame([Counter(x) for x in df_B]).fillna(0).astype('int8')
    A_special = A_special.add_suffix('_A')
    B_special = B_special.add_suffix('_B')
    X = pd.concat([A_special, B_special], axis=1)
    return X

X = special_count(train)
train = pd.concat([train, X], axis=1)
# -

train


# +
def win_rate_special(special, df):
    count = (len(df[df[special + '_A'] != 0]) + len(df[df[special + '_B'] != 0])) # それぞれのチームで対象スペシャルが出現する試合数のカウント
    win = (len(df[(df[special + '_A'] != 0) & (df.y == 1)]) +len((df[(df[special + '_B'] != 0) & (df.y == 0)]))) # それぞれのチームで対象スペシャルが出現する試合のうち各チームが勝利する試合数のカウント
    rate = win / count
    return rate, count

win_rate_special_df = []
for special in buki_data.special.unique():
    rate, count = win_rate_special(special, train)
    win_rate_special_df.append([special, rate, count])
win_rate_special_df = pd.DataFrame(win_rate_special_df, columns=['special', 'win_rate', 'count'])
win_rate_special_df = win_rate_special_df.sort_values(by= 'win_rate', ascending=False)
win_rate_special_df = win_rate_special_df.reset_index(drop=True)

display(win_rate_special_df)
# -

# クイックボムピッチャーが最も勝率の高いスペシャルとなりました。  
# このスペシャルは非常に珍しく、バケットスロッシャーソーダと14式竹筒銃・乙の2ブキのみしかこのスペシャルを使えるブキはありません。  
# 非常にキル力の高いスペシャルと言われています。  
# 最下位はハイパープレッサーとなりました。ハイパープレッサーはスペシャルの性質上、後衛ブキが持つことが多く、チャージャーやジェットスイーパーカスタムの勝率の低さを反映していると思われます。  

# # 【検証4】ヤグラのナイス玉強すぎない？？
# それぞれのスペシャルの性質上、ルールごとに有利不利があると言われています。  
# 次からはルールごとのスペシャルの勝率を見てみます。  
# まずは普段から感じているヤグラでのナイス玉の強さ。  
# ガチヤグラではヤグラにイカが乗ってカウントを進めなければいけないため、ヤグラを中心にダメージフィールドを展開できるナイス玉は強制的に敵のカウントを止めたり、味方のヤグラを進めたりできるため、非常に強力だと言われていますが実際どうでしょうか。  
# ルール別/スペシャル別の勝率を見てみます。  

# +
win_rate_special_mode_df = pd.DataFrame()
for mode in train['mode'].unique():
    win_rate_special_df = []
    train_mode = train[train['mode'] == mode]
    for special in buki_data.special.unique():
        rate, count = win_rate_special(special, train_mode)
        win_rate_special_df.append([special, rate, count])
    win_rate_special_df = pd.DataFrame(win_rate_special_df, columns=[mode + '_special', mode + '_win_rate', mode + '_count'])
    win_rate_special_df = win_rate_special_df.sort_values(by= mode + '_win_rate', ascending=False)
    win_rate_special_df = win_rate_special_df.reset_index(drop=True)
    win_rate_special_mode_df = pd.concat([win_rate_special_mode_df, win_rate_special_df], axis=1)
    
display(win_rate_special_mode_df)
# -

nawabari_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['nawabari_special'] == 'nicedama']['nawabari_win_rate'].values[0]
area_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['area_special'] == 'nicedama']['area_win_rate'].values[0]
yagura_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['yagura_special'] == 'nicedama']['yagura_win_rate'].values[0]
hoko_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['hoko_special'] == 'nicedama']['hoko_win_rate'].values[0]
asari_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['asari_special'] == 'nicedama']['asari_win_rate'].values[0]
plt.bar(['nawabari', 'area', 'yagura', 'hoko', 'asari'],[nawabari_win_rate, area_win_rate, yagura_win_rate, hoko_win_rate, asari_win_rate])
plt.ylim(0.4, 0.6) ,plt.title('nicedama win rate');

# ヤグラでの勝率1位はやはりナイス玉でした。  
# ただ、ヤグラ、アサリは他のルールに比べてスペシャルによる勝率の違いは少ないようです。  
# ナイス玉はナワバリやエリアでも強いことがわかります。

# # 【検証5】ウルトラハンコはホコに強く、ヤグラに弱い？？
# ウルトラハンコは発動すると巨大なハンコを地面に打ち付けながら進むスペシャルのため、ヤグラへの干渉が難しく弱いと言われています。  
# ホコではホコバリアを一瞬で割れたり、ホコが進む道を作ったりとウルトラハンコがあると非常に助かる場面が多い印象です。

nawabari_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['nawabari_special'] == 'ultrahanko']['nawabari_win_rate'].values[0]
area_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['area_special'] == 'ultrahanko']['area_win_rate'].values[0]
yagura_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['yagura_special'] == 'ultrahanko']['yagura_win_rate'].values[0]
hoko_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['hoko_special'] == 'ultrahanko']['hoko_win_rate'].values[0]
asari_win_rate = win_rate_special_mode_df[win_rate_special_mode_df['asari_special'] == 'ultrahanko']['asari_win_rate'].values[0]
plt.bar(['nawabari', 'area', 'yagura', 'hoko', 'asari'],[nawabari_win_rate, area_win_rate, yagura_win_rate, hoko_win_rate, asari_win_rate])
plt.ylim(0.4, 0.6)
plt.title('ultrahanko win rate')


# 本当にヤグラに弱かったですね。ちょっとびっくりしました。  
# ホコは2位でやはり強いようですが、ナワバリやエリアでも強さを見せます。

# ## 【検証6】 インクアーマー二枚編成は強い？？
# インクアーマーとは発動すると自分と味方全員にインクのアーマーをまとわせます。  
# アーマーは30のダメージを受けると消滅します。  
# ガチマッチに潜ってアーマー2枚編成だとうまくアーマーが回って強いような気がしているので、検証してみます。  
# 3枚以上だと、発動の被りなどがあってイマイチな気がしていますが、どうでしょうか。  
# インクアーマーがある場合の平均勝率は0.506です。

# +
def win_rate_special_count(special, special_count, df):
    count = (len(df[df[special + '_A'] == special_count]) + len(df[df[special + '_B'] == special_count])) # それぞれのチームで対象スペシャルが出現する試合数のカウント
    win = (len(df[(df[special + '_A'] == special_count) & (df.y == 1)]) +len((df[(df[special + '_B'] == special_count) & (df.y == 0)]))) # それぞれのチームで対象スペシャルが出現する試合のうち各チームが勝利する試合数のカウント
    rate = win / count
    return rate, count

win_rate_special_df = []
special = 'armor'
for armor_count in range(1, 5):
    rate, count = win_rate_special_count(special, armor_count, train)
    win_rate_special_df.append([special, armor_count, rate, count])
win_rate_special_df = pd.DataFrame(win_rate_special_df, columns=['special','armor_count', 'win_rate', 'count'])
win_rate_special_df = win_rate_special_df.sort_values(by= 'win_rate', ascending=False)
win_rate_special_df = win_rate_special_df.reset_index(drop=True)

display(win_rate_special_df)
# -

# 仮説通りインクアーマー二枚が最も勝率が高いことがわかりました。

# # 【検証7】 長射程有利のステージがあるって本当？？
# 見晴らしが良かったり、高台が存在するステージ(アロワナモール、デボン海洋博物館、ホテルニューオートロ、モンガラキャンプ場など)では長射程が有利だと言われていますが、本当でしょうか。  
# 今回はチャージャー種、スピナー種を長射程とします。(ジェットスイーパーも長いけど若干役割が違いそうので今回は除外)

cols = ['A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', 'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon']
for col in cols:
    train = train.merge(buki_data[['key','category2']], left_on=col, right_on='key', how='left').drop(columns='key').rename(columns={'category2':'category2' + '_' + col})


def category2_count(df):
    df_A = df[['category2_A1-weapon', 'category2_A2-weapon', 'category2_A3-weapon', 'category2_A4-weapon']].values
    df_B = df[['category2_B1-weapon', 'category2_B2-weapon', 'category2_B3-weapon', 'category2_B4-weapon']].values
    A_category2 = pd.DataFrame([Counter(x) for x in df_A]).fillna(0).astype('int8')
    B_category2 = pd.DataFrame([Counter(x) for x in df_B]).fillna(0).astype('int8')
    A_category2 = A_category2.add_suffix('_A')
    B_category2 = B_category2.add_suffix('_B')
    X = pd.concat([A_category2, B_category2], axis=1)
    return X
X = category2_count(train)
train = pd.concat([train, X], axis=1)


# +
def win_rate_category2(category2, df):
    count = (len(df[df[category2 + '_A'] != 0]) + len(df[df[category2 + '_B'] != 0])) # それぞれのチームで対象カテゴリーが出現する試合数のカウント
    win = (len(df[(df[category2 + '_A'] != 0) & (df.y == 1)]) +len((df[(df[category2 + '_B'] != 0) & (df.y == 0)]))) # それぞれのチームで対象カテゴリーが出現する試合のうち各チームが勝利する試合数のカウント
    rate = win / count
    return rate, count

win_rate_category2_df = []
for category2 in buki_data.category2.unique():
    rate, count = win_rate_category2(category2, train)
    win_rate_category2_df.append([category2, rate, count])
win_rate_category2_df = pd.DataFrame(win_rate_category2_df, columns=['category2', 'win_rate', 'count'])
win_rate_category2_df = win_rate_category2_df.sort_values(by= 'win_rate', ascending=False)
win_rate_category2_df = win_rate_category2_df.reset_index(drop=True)

display(win_rate_category2_df)
# -

# チャージャー種の平均勝率は0.475、スピナー種の勝率は0.505

win_rate_charger_df = []
category2 = 'charger'
print('Charger win rate by stage\n')
for stage in train.stage.unique():
    df = train[train['stage'] == stage] 
    rate, count = win_rate_category2(category2, df)
    win_rate_charger_df.append([stage, rate, count])
win_rate_charger_df = pd.DataFrame(win_rate_charger_df, columns=['category2', 'win_rate', 'count'])
win_rate_charger_df = win_rate_charger_df.sort_values(by= 'win_rate', ascending=False)
win_rate_charger_df = win_rate_charger_df.reset_index(drop=True)
win_rate_charger_df

win_rate_charger_df = []
category2 = 'splatling'
print('Splatling win rate by stage\n')
for stage in train.stage.unique():
    df = train[train['stage'] == stage] 
    rate, count = win_rate_category2(category2, df)
    win_rate_charger_df.append([stage, rate, count])
win_rate_charger_df = pd.DataFrame(win_rate_charger_df, columns=['category2', 'win_rate', 'count'])
win_rate_charger_df = win_rate_charger_df.sort_values(by= 'win_rate', ascending=False)
win_rate_charger_df = win_rate_charger_df.reset_index(drop=True)
win_rate_charger_df


# チャージャー種の勝率が高いのはデボン海洋博物館、タチウオパーキング、コンブトラック、モンガラキャンプ場、ホテルニューオートロ、  
# スピナー種の勝率が高いのはハコフグ倉庫、スメーシーワールド、ホテルニューオートロ、モズク農園、アジフライスタジアムとなりました。  
# 筆者の経験則とは少しズレますが、ある程度納得のいく結果のようにも見えます。  
#
# チャージャーでは平均と比較して勝率の高いタチウオパーキングが、スピナーでは最も勝率が低くなっているのが面白いです。  
# 長射程といってもステージごとの勝ちやすさの違いがあるようです。筆者はどちらも使ったことがないのでよくわかりません。  
#
# スピナー種はチャージャー種に比べて、ステージごとの勝率の差が大きいのも面白い発見だと思います。  
# スピナー種 : 0.475～0.528  
# チャージャー種: 0.453～0.481  
#   
# 他のブキもステージごとの勝率を見てみるといろいろ面白いと思います。

# # 【検証8】 チャージャー二枚編成は弱い？？
# 筆者は自分のチームにチャージャー二枚がくると負けることが多いような気がしています。実際のところどうなのか検証してみます。

def win_rate_category2_count(category2, category2_count, df):
    count = (len(df[df[category2 + '_A'] == category2_count]) + len(df[df[category2 + '_B'] == category2_count])) # それぞれのチームで対象カテゴリーが出現する試合数のカウント
    win = (len(df[(df[category2 + '_A'] == category2_count) & (df.y == 1)]) +len((df[(df[category2 + '_B'] == category2_count) & (df.y == 0)]))) # それぞれのチームで対象カテゴリーが出現する試合のうち各チームが勝利する試合数のカウント
    rate = win / count
    return rate, count


# +
win_rate_category2_df = []
category2 = 'charger'
for category2_count in range(1, 5):
    rate, count = win_rate_category2_count(category2, category2_count, train)
    win_rate_category2_df.append([category2, category2_count, rate, count])
win_rate_category2_df = pd.DataFrame(win_rate_category2_df, columns=['category2','category2_count', 'win_rate', 'count'])
win_rate_category2_df = win_rate_category2_df.sort_values(by= 'win_rate', ascending=False)
win_rate_category2_df = win_rate_category2_df.reset_index(drop=True)

display(win_rate_category2_df)
# -

# 仮説通りチャージャー二枚編成は勝率が低いようです。
#
# # 最後に
# ルールやステージにより勝ちやすいブキやスペシャルは確かに存在しそうです。  
# 今回やらなかったこととして、ブキの組み合わせやウデマエ(C-~X)別でのブキの勝率の違いなども参考になるかもしれません。  



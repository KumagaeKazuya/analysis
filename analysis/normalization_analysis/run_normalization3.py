#!/usr/bin/env python3
"""
run_normalization3.py

人物ごと・列ごとIQRスケール計算、列ごと集約、正規化関数適用、Zスコア化

使い方例:
	python run_normalization3.py input.csv function_parameters_exp.json
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd



def compute_person_column_iqr(df):
	"""
	人物ごと・列ごとにIQRを計算
	Returns: DataFrame [person_id, column_position, iqr, median]
	"""
	results = []
	for (pid, col), group in df.groupby(["person_id", "column_position"]):
		if len(group) < 4:
			continue
		q1 = group["shoulder_width"].quantile(0.25)
		q3 = group["shoulder_width"].quantile(0.75)
		iqr = q3 - q1
		median = group["shoulder_width"].median()
		results.append({
			"person_id": pid,
			"column_position": col,
			"iqr": iqr,
			"median": median
		})
	return pd.DataFrame(results)

def aggregate_iqr_by_column(iqr_df):
	"""
	列ごとにIQRの中央値を集約
	Returns: DataFrame [column_position, iqr_median]
	"""
	return iqr_df.groupby("column_position")["iqr"].median().reset_index().rename(columns={"iqr": "iqr_median"})

def aggregate_median_by_column(iqr_df):
	"""
	列ごとに人物中央値を集約
	Returns: DataFrame [column_position, median_of_medians]
	"""
	return iqr_df.groupby("column_position")["median"].median().reset_index().rename(columns={"median": "median_of_medians"})

def apply_feature_space_zscore(df):
	"""
	特徴空間でのZスコア正規化（全体平均・全体標準偏差で一括）
	"""
	df = df.copy()
	df = df[df["shoulder_width"].notnull()]
	mean = df["shoulder_width"].mean()
	std = df["shoulder_width"].std(ddof=0)
	df["zscore"] = (df["shoulder_width"] - mean) / std
	return df


def main():
	parser = argparse.ArgumentParser(description="人物ごと・列ごとIQRスケール計算と特徴空間Zスコア正規化")
	parser.add_argument("csv_path", help="入力CSVファイル（shoulder_width, person_id, column_position列が必要）")
	parser.add_argument("--output", default="normalized_output.csv", help="出力CSVファイル名")
	args = parser.parse_args()

	df = pd.read_csv(args.csv_path, encoding="utf-8-sig")
	if not set(["shoulder_width", "person_id", "column_position"]).issubset(df.columns):
		print("❌ 必要なカラムがありません")
		sys.exit(1)

	print("① 人物ごと・列ごとIQR計算...")
	iqr_df = compute_person_column_iqr(df)
	print(iqr_df.head())

	print("② 列ごとにIQR中央値を集約...")
	iqr_agg = aggregate_iqr_by_column(iqr_df)
	print(iqr_agg)

	print("② 列ごとに人物中央値を集約...")
	median_agg = aggregate_median_by_column(iqr_df)
	print(median_agg)

	print("③ 特徴空間でのZスコア正規化...")
	df_z = apply_feature_space_zscore(df)
	print(df_z[["person_id", "column_position", "shoulder_width", "zscore"]].head())

	print(f"💾 Zスコア化データを {args.output} に保存")
	df_z.to_csv(args.output, index=False, encoding="utf-8-sig")

	print("--- 完了 ---")

if __name__ == "__main__":
	main()

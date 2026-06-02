import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def check_regression_accuracy(
	csv_file="output.csv",
	actual_col="actual",
	predicted_col="predicted",
):
	"""Compute regression accuracy metrics by comparing two numeric columns."""
	try:
		df = pd.read_csv(csv_file)

		if actual_col not in df.columns:
			raise KeyError(actual_col)
		if predicted_col not in df.columns:
			raise KeyError(predicted_col)

		actual = pd.to_numeric(df[actual_col], errors="coerce")
		predicted = pd.to_numeric(df[predicted_col], errors="coerce")

		# Keep only rows where both columns contain valid numeric values.
		valid_mask = actual.notna() & predicted.notna()
		actual = actual[valid_mask]
		predicted = predicted[valid_mask]

		total_rows = len(df)
		used_rows = len(actual)
		if used_rows == 0:
			raise ValueError("No valid numeric row pairs found to compare.")

		rmse = np.sqrt(mean_squared_error(actual, predicted))
		mae = mean_absolute_error(actual, predicted)
		r2 = r2_score(actual, predicted)

		non_zero_mask = actual != 0
		if non_zero_mask.any():
			mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
			accuracy_percent = float(np.clip(100.0 - mape, 0.0, 100.0))
		else:
			mape = np.nan
			accuracy_percent = np.nan

		print(f"Rows used: {used_rows}/{total_rows}")
		print(f"RMSE: {rmse:.4f}")
		print(f"MAE: {mae:.4f}")
		print(f"R2 Score: {r2:.4f}")
		if np.isnan(mape):
			print("MAPE: NaN (actual values are all zero)")
			print("Accuracy: NaN%")
		else:
			print(f"MAPE: {mape:.2f}%")
			print(f"Accuracy: {accuracy_percent:.2f}%")

		return {
			"rows_used": used_rows,
			"rows_total": total_rows,
			"rmse": float(rmse),
			"mae": float(mae),
			"r2": float(r2),
			"mape_percent": float(mape) if not np.isnan(mape) else np.nan,
			"accuracy_percent": accuracy_percent,
		}

	except FileNotFoundError:
		print(f"Error: File '{csv_file}' not found.")
		return None
	except KeyError as e:
		print(f"Error: Column '{e.args[0]}' not found in '{csv_file}'.")
		return None
	except Exception as e:
		print(f"Error: {e}")
		return None


if __name__ == "__main__":
	# Update column names as needed for your file.
	check_regression_accuracy("output.csv", "median_house_value", "median_house_value_predicted")

# 🏥 Medical Insurance Charges Prediction

Predicts yearly medical insurance `charges` from age, sex, BMI, number of children, smoking status and region, using regression models. Includes a Streamlit web app for live predictions.

**Live demo:** https://insurancechargespredictor33.streamlit.app/

## Dataset

`insurance.csv`: about 1,338 records with these columns:

| Column | Description |
|---|---|
| `age` | Age of the policyholder (18–64) |
| `sex` | male / female |
| `bmi` | Body mass index |
| `children` | Number of dependants (0–5) |
| `smoker` | yes / no |
| `region` | southwest / southeast / northwest / northeast |
| `charges` | Yearly medical charges in USD (target) |

## Key observations

**1. Smoking dominates everything else.**
Smoker status has by far the strongest correlation with charges (Pearson r ≈ **0.79**), compared with age (≈ 0.29) and BMI (≈ 0.20). Only about 20% of people in the data are smokers, yet they drive most of the high-cost cases.

**2. Age and BMI matter, but less.**
Charges rise steadily with age. Higher BMI adds to costs, and the obese BMI category (BMI ≥ 30) shows a clear positive association (r ≈ 0.20). Normal and overweight categories show weak negative correlations.

**3. Sex, children and region have weak effects.**
Sex (r ≈ −0.06), number of children (≈ 0.06) and region (|r| < 0.08) barely move the charges. Only `region_southeast` and `is_female` passed the Chi-square test (p < 0.05); `region_northwest`, `region_southwest` and the normal/overweight BMI categories did not.

**4. Charges are right-skewed.**
The mean charge is about $13,300 but the median is about $9,400, and the maximum is about $63,800. A minority of expensive cases pull the average up, which is why the model has trouble with extreme values.

**5. Complex models beat Linear Regression.**
Charges depend on interactions, such as smoker × BMI. A smoker with high BMI costs far more than the effects of each added up. Tree-based models capture this and linear regression does not.

**6. The data is well balanced and clean.**
Roughly equal numbers of men and women, no missing values, and only a small number of duplicate rows.

## Model results

Results from the original notebook (80/20 split):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.786 | ~4,150 | ~5,786 |
| Random Forest | 0.861 | ~2,689 | ~4,663 |
| **Gradient Boosting** | **0.883** | **~2,464** | **~4,278** |

`train.py` prints fresh R² and MAE for its own run. Those numbers may differ slightly from the table above.

## Example inputs

These are real rows from the dataset. Predictions should be close to the actual charges but not identical, since the model learns general patterns.

| Age | Sex | BMI | Children | Smoker | Region | Actual charges |
|---|---|---|---|---|---|---|
| 19 | female | 27.9 | 0 | yes | southwest | $16,885 |
| 18 | male | 33.77 | 1 | no | southeast | $1,726 |
| 28 | male | 33.0 | 3 | no | southeast | $4,449 |
| 33 | male | 22.7 | 0 | no | northwest | $21,984 |
| 32 | male | 28.9 | 0 | no | northwest | $3,867 |

**Sanity checks to try in the app**

- Same person, Smoker = no vs. yes: the smoker's estimate should be several times higher.
- Same person at age 20 vs. 60: the older estimate should be clearly higher.
- A young, non-smoking, normal-BMI person should land at the low end (a few thousand dollars).

The 33-year-old non-smoker with $21,984 is an outlier. Expect the model to predict much lower for that row.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Place `insurance.csv` in the same folder as the app.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub, including `insurance.csv`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repo, set the main file to `app_streamlit.py`, and deploy.

## Limitations

- Small dataset (about 1.3k rows), so predictions are estimates and not quotes.
- Trained on one dataset; real insurance pricing involves many more factors.
- Higher charges (above about $40k) are predicted less accurately.
- For educational use only.

## Tech stack

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, SciPy, Streamlit.

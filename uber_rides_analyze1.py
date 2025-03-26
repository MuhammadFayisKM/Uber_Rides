import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import OneHotEncoder

mydata = pd.read_csv("UberDataset.csv")


# print(mydata.head())
# print()
print(mydata.info())
print()
# print(mydata.describe())
# print()
# print(mydata.shape)

mydata["PURPOSE"].fillna("NOT", inplace=True)
# print(mydata.info())
# print()

# print("Original data: ")
# print(mydata.head(10))
# print()

mydata["START_DATE"] = pd.to_datetime(mydata["START_DATE"], errors="coerce")
mydata["END_DATE"] = pd.to_datetime(mydata["END_DATE"], errors="coerce")

# print("Updated data: ")
# print(mydata.head(10))
# print()

mydata["DATE"] = pd.DatetimeIndex(mydata["START_DATE"]).date
mydata["TIME"] = pd.DatetimeIndex(mydata["START_DATE"]).hour

# changing into categories of day and night
mydata["DAY-NIGHT"] = pd.cut(
    x=mydata["TIME"],
    bins=[0, 10, 15, 19, 24],
    labels=["Morning", "Afternoon", "Evening", "Night"],
)
# print(mydata.head(10))
# print()

mydata.dropna(inplace=True)
mydata.drop_duplicates(inplace=True)
# print(mydata.head())

obj = mydata.dtypes == "object"
object_cols = list(obj[obj].index)

unique_values = {}
for col in object_cols:
    unique_values[col] = mydata[col].unique().size
# print(unique_values)

# for the analyze of the data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=mydata["CATEGORY"])
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
sns.countplot(x=mydata["PURPOSE"])
plt.xticks(rotation=90)
plt.savefig("purpose-category-distribution.png")
plt.show()

sns.countplot(x=mydata["DAY-NIGHT"])
plt.xticks(rotation=90)
plt.savefig("day-night-distribution.png")
plt.show()


plt.figure(figsize=(15, 5))
sns.countplot(data=mydata, x="PURPOSE", hue="CATEGORY")
plt.xticks(rotation=90)
plt.savefig("purpose-category-distribution.png")
plt.show()

object_cols = ["CATEGORY", "PURPOSE"]
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

OH_cols = pd.DataFrame(OH_encoder.fit_transform(mydata[object_cols]))

OH_cols.index = mydata.index
OH_cols.columns = OH_encoder.get_feature_names_out()
dataFinal = mydata.drop(object_cols, axis=1)
mydata = pd.concat([dataFinal, OH_cols], axis=1)


numeric_data = mydata.select_dtypes(include=["number"])

sns.heatmap(
    numeric_data.corr(),
    cmap="BrBG",
    fmt=".2f",
    linewidths=2,
    annot=True,
)
plt.savefig("correlation_heatmap.png")
plt.show()

mydata["MONTH"] = pd.DatetimeIndex(mydata["START_DATE"]).month
month_label = {
    1.0: "Jan",
    2.0: "Feb",
    3.0: "Mar",
    4.0: "Apr",
    5.0: "May",
    6.0: "Jun",
    7.0: "Jul",
    8.0: "Aug",
    9.0: "Sep",
    10.0: "Oct",
    11.0: "Nov",
    12.0: "Dec",
}

mydata["MONTH"] = mydata.MONTH.map(month_label)

month_val = mydata.MONTH.value_counts(sort=False)

# total month ride count vs month ride max count

data_f = pd.DataFrame(
    {
        "MONTHS": month_val.values,
        "VALUE COUNT": mydata.groupby("MONTH", sort=False)["MILES"].max(),
    }
)

pval = sns.lineplot(data=data_f)
pval.set(xlabel="MONTHS", ylabel="VALUE COUNT")
plt.savefig("monthly-distribution.png")
plt.show()

# day by day data analyze
mydata["DAY"] = mydata.START_DATE.dt.weekday

day_label = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}

mydata["DAY"] = mydata["DAY"].map(day_label)

day_label = mydata.DAY.value_counts()
sns.barplot(x=day_label.index, y=day_label.values)
plt.xlabel("DAY")
plt.ylabel("COUNT")
plt.savefig("day-count-distribution.png")
plt.show()


sns.distplot(mydata[mydata["MILES"] < 40]["MILES"])
plt.savefig("miles_distribution.png")
plt.show()

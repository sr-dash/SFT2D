import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



import pandas as pd
import os

# Directory containing the downloaded sunspot files
input_dir = "./RGO_Sunspots"

# Define column names based on the format file
columns = [
    "Year", "Month", "Day", "Time_Thousandths", "GSG/NOAA", "Spot_Area", "Latitude", "Longitude"
]

# Initialize an empty DataFrame
sunspot_df = pd.DataFrame(columns=columns)

# Iterate through all files in the directory
for filename in sorted(os.listdir(input_dir)):
# for filename in ['/Users/soumya/Uni-Hawaii/Work/GIT-Projects/SFT-2D/SFT2D/RGO_Sunspots/g1874.txt']:
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)

        # Read the file, skipping header lines (first 2 lines)
        temp_df = pd.read_fwf(file_path, colspecs=[(0,4), (4,6), (6,8), (8,12), (15,20), (40,45), (63,69), (57,63)],
                              header=None, names=columns)
        temp_df.dropna(inplace=True)
        # Create a date column from year, month, day, and time in thousandths of a day
        temp_df["Date"] = pd.to_datetime(temp_df["Year"].astype(str) + '-' + temp_df["Month"].astype(str) + '-' + temp_df["Day"].astype(str))
        temp_df["Date"] += pd.to_timedelta(temp_df["Time_Thousandths"] * 86400, unit='s')

        # Append data to the main DataFrame
        sunspot_df = pd.concat([sunspot_df, temp_df], ignore_index=True)

# Drop the year, month, day, and time columns after forming the Date
sunspot_df.drop(columns=["Year", "Month", "Day", "Time_Thousandths"], inplace=True)

# Multiply 1.4 into Spot_Area column after 1976
sunspot_df.loc[sunspot_df["Date"].dt.year >= 1976, "Spot_Area"] *= 1.4
# Reset the index
sunspot_df.reset_index(drop=True, inplace=True)

# Sort the dataframe by repeating GSG/NOAA number for which Sunspot_Area is maximum
sunspot_df_sorted = sunspot_df.sort_values(by=["Spot_Area"], ascending=True)
sunspot_df_sorted.drop_duplicates(subset="GSG/NOAA", keep="first", inplace=True)
# sunspot_df_sorted_final = sunspot_df_sorted.reset_index(drop=True, inplace=True)
sunspot_df_sorted.sort_values(by=["Date"], inplace=True)
sunspot_df_sorted.reset_index(drop=True, inplace=True)

# Count the number of sunspots per month and apply a rolling mean of 13 months. Use the sunspot_df_sorted dataframe..
sunspot_monthly = sunspot_df.set_index("Date").resample("ME").size().reset_index(name="Sunspot_Count")
sunspot_monthly["Rolling_Mean"] = sunspot_monthly["Sunspot_Count"].rolling(window=13,center=True).mean()


# plt.figure(figsize=[10,5])
# plt.plot(sunspot_monthly["Date"], sunspot_monthly["Sunspot_Count"],c='blue')
# # plt.plot(sunspot_per_year["Date"], sunspot_per_year["Sunspot_Count"])
# plt.xlabel("Year")
# plt.ylabel("Number of Sunspots")
# plt.title("13 Month Smoothed Sunspot Count")
# plt.savefig('./RGO_Sunspots_timeseries.png',dpi=200,bbox_inches='tight',pad_inches=0.1)

# plt.show()

# Function to convert datetime to fractional year
def datetime_to_fractional_year(date):
    year_start = pd.Timestamp(f"{date.year}-01-01")
    year_end = pd.Timestamp(f"{date.year + 1}-01-01")
    return date.year + (date - year_start).total_seconds() / (year_end - year_start).total_seconds()

sunspot_df_sorted["Fractional_Year"] = sunspot_df_sorted["Date"].apply(datetime_to_fractional_year)

# plt.figure(figsize=[10,5])
# plt.scatter(sunspot_df_sorted['Fractional_Year'],sunspot_df_sorted['Latitude'],s=1,c='k')
# # Format the x axis as year with 50 years interval where the dataframe date has a daily cadence.
# plt.xticks(np.arange(1874,2026,20))
# plt.xlabel("Year")
# plt.ylabel("Latitude")
# # plt.title("Sunspot Latitude and Area")
# plt.text(0.65,0.93,'Total number of spots: {}'.format(sunspot_df_sorted.shape[0]),transform=plt.gca().transAxes,fontsize=12)
# plt.savefig('./RGO_Sunspots_Butterfly.png',dpi=200,bbox_inches='tight',pad_inches=0.1)
# plt.show()

# Reorganize sunspot_df_sorted dataframe to have Date as first column and Fractional_Year as second column.
sunspot_df_sorted = sunspot_df_sorted[['Date','Fractional_Year','GSG/NOAA','Spot_Area','Latitude','Longitude']]


# Save the combined DataFrame to a CSV file
# sunspot_df_sorted.to_csv("sunspot_data_rgo.csv", index=False, float_format='%.3f', date_format='%Y-%m-%dT%H:%M:%S',sep='\t')

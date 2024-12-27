import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the NDVI GeoTIFF file
file_path = 'NDVI_Composite_Pune.tif'  # Replace with the actual file path
ndvi_dataset = rasterio.open(file_path)

# Step 2: Read NDVI data
ndvi = ndvi_dataset.read(1)  # Read the first band of the GeoTIFF file

# Step 3: Handle invalid data (e.g., no-data values are typically represented as -9999)
ndvi = np.where(ndvi == -9999, np.nan, ndvi)

# Step 4: Visualize NDVI
plt.figure(figsize=(10, 6))
plt.imshow(ndvi, cmap='YlGn', vmin=0, vmax=1)
plt.colorbar(label='NDVI')
plt.title('NDVI for Pune')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Step 5: Calculate NDVI Statistics
mean_ndvi = np.nanmean(ndvi)
median_ndvi = np.nanmedian(ndvi)
std_ndvi = np.nanstd(ndvi)

print(f"NDVI Statistics:")
print(f"  Mean NDVI: {mean_ndvi:.4f}")
print(f"  Median NDVI: {median_ndvi:.4f}")
print(f"  Standard Deviation NDVI: {std_ndvi:.4f}")

# Step 6: Export NDVI Data to CSV for Model Integration
# Flatten the NDVI array and create a DataFrame
ndvi_flat = ndvi.flatten()  # Flatten the 2D NDVI array to 1D
df = pd.DataFrame({'NDVI': ndvi_flat})  # Create a DataFrame
df = df.dropna()  # Remove NaN values

# Save NDVI values to a CSV file
csv_path = 'NDVI_Pune.csv'
df.to_csv(csv_path, index=False)
print(f"NDVI data saved to {csv_path}")

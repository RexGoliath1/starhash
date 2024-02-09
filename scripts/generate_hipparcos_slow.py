from astroquery.vizier import Vizier
import pandas as pd

# Define the Vizier instance with a specific row_limit
v = Vizier(catalog="I/239/hip_main", row_limit=50)
v.ROW_LIMIT = 50  # Adjust as needed

# Initialize an empty DataFrame to store all results
df = pd.DataFrame()

# Initialize the HIP number to start with
current_hip = 1

# Loop until no more data is found
while True:
    # Define the query constraint, here we select a range of HIP numbers
    query_str = f"HIP >= {current_hip} && HIP < {current_hip + 50}"
    
    # Query the Vizier catalog with the defined constraint
    catalog_list = v.query_constraints(query_str, verbose=True)
    
    # Check if any results were returned
    if catalog_list is not None and len(catalog_list) > 0:
        # Convert to pandas DataFrame and concatenate
        table = catalog_list[0].to_pandas()
        df = pd.concat([df, table], ignore_index=True)
        
        # Update the current HIP number to the last one retrieved plus one
        current_hip = df['HIP'].max() + 1
    else:
        # If no results are returned, we've reached the end of the catalog
        break

# Now df has all the concatenated results from the queries
print(df)

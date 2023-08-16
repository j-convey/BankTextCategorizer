import pandas as pd
import random

df = pd.read_csv('data/Decor Stores.csv')
citiesdf = pd.read_csv('data/uscities.csv')
result_data = []
used_cities = set()
counter = 0

while counter < 2000:
    added_this_loop = False  # A flag to check if we added any new city-state combination in the current loop

    for index, row in df.iterrows():
        # Stop appending after 2000 entries
        if counter >= 2000:
            break

        # Randomly select a city and state
        rand_city = citiesdf.sample().iloc[0]
        city_state = rand_city['city'] + " " + rand_city['state_id']

        # Check if we've used this combination before
        if city_state not in used_cities:
            new_row = row.copy()
            new_row['Description'] = row['Description'] + " " + city_state
            result_data.append(new_row.to_dict())
            used_cities.add(city_state)
            counter += 1
            added_this_loop = True

    if not added_this_loop:
        break


# Convert the list of rows into a DataFrame
result_df = pd.DataFrame(result_data)

# Save the updated dataframe
result_df.to_csv('Categorize Bank Descriptions/Training Data/Decor Stores 2k.csv', index=False)

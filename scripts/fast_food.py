import pandas as pd

def process_csv(input_filename, output_filename):
    df = pd.read_csv(input_filename)

    # Create the description column
    df['description'] = df['name'] + ' ' + df['address']
    df['description'] += ' ' + df['name'] + ' ' + df['categories']
    df['description'] += ' ' + df['name'] + ' ' + df['city'] + ' ' + df['country'] + ' ' + df['province'] + ' ' + df['country']

    # Drop other columns
    df = df[['description']]

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_filename, index=False)


input_file = 'data/Datafiniti_Fast_Food_Restaurants.csv/Datafiniti_Fast_Food_Restaurants.csv'
output_file = 'output.csv'
process_csv(input_file, output_file)

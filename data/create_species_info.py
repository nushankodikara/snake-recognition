import pandas as pd

def create_species_info():
    # Read the CSV files
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Combine relevant columns from both datasets
    columns_to_keep = ['class_id', 'binomial', 'genus', 'family', 
                      'snake_sub_family', 'poisonous']
    
    # Get unique species info from both datasets
    train_species = train_df[columns_to_keep].drop_duplicates()
    test_species = test_df[columns_to_keep].drop_duplicates()
    
    # Combine and remove any duplicates
    all_species = pd.concat([train_species, test_species]).drop_duplicates()
    
    # Sort by class_id for better readability
    all_species = all_species.sort_values('class_id')
    
    # Add some additional statistics
    train_counts = train_df['class_id'].value_counts()
    test_counts = test_df['class_id'].value_counts()
    
    # Add count columns
    all_species['train_samples'] = all_species['class_id'].map(train_counts).fillna(0).astype(int)
    all_species['test_samples'] = all_species['class_id'].map(test_counts).fillna(0).astype(int)
    all_species['total_samples'] = all_species['train_samples'] + all_species['test_samples']
    
    # Get most common countries and continents for each species
    def get_most_common(df, class_id, column):
        species_data = df[df['class_id'] == class_id][column]
        if len(species_data) == 0:
            return 'unknown'
        # Exclude 'unknown' unless it's the only value
        valid_data = species_data[species_data != 'unknown']
        if len(valid_data) == 0:
            return 'unknown'
        return valid_data.mode().iloc[0]
    
    # Add geographical information
    all_species['primary_country'] = all_species['class_id'].apply(
        lambda x: get_most_common(pd.concat([train_df, test_df]), x, 'country')
    )
    all_species['primary_continent'] = all_species['class_id'].apply(
        lambda x: get_most_common(pd.concat([train_df, test_df]), x, 'continent')
    )
    
    # Reorder columns for better organization
    final_columns = [
        'class_id', 'binomial', 'genus', 'family', 'snake_sub_family',
        'poisonous', 'primary_continent', 'primary_country',
        'train_samples', 'test_samples', 'total_samples'
    ]
    
    all_species = all_species[final_columns]
    
    # Save to CSV
    output_file = 'snake_species_info.csv'
    all_species.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(all_species)} unique snake species")
    
    # Print some basic statistics
    print("\nDataset Statistics:")
    print(f"Total number of species: {len(all_species)}")
    print(f"Venomous species: {len(all_species[all_species['poisonous'] == 1])}")
    print(f"Non-venomous species: {len(all_species[all_species['poisonous'] == 0])}")
    print(f"\nMost represented families:")
    print(all_species['family'].value_counts().head())

if __name__ == "__main__":
    create_species_info()

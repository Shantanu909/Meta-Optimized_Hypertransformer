import os
import csv

folder_path = 'OMicrons/H1_PEM-VAULT_MAG_1030X195Y_COIL_Y_DQ/csvs'  # Folder containing all your CSVs
channel_value = "H1_PEM-VAULT_MAG_1030X195Y_COIL_Y_DQ"  # The value to assign to the Channel Name column
label_value = 1  # The value to assign to the Label column

# Iterate through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        csv_file_path = os.path.join(folder_path, filename)

        # Open and read the CSV file
        with open(csv_file_path, 'r', newline='') as file:
            reader = list(csv.reader(file))
            header, rows = reader[0], reader[1:]

        # Add the 'Channel Name' column if it doesn't exist
        if 'Channel Name' not in header:
            header.append('Channel Name')
            for row in rows:
                row.append(str(channel_value))
        else:
            # If 'Channel Name' exists, make sure to add the value if it's missing
            channel_index = header.index('Channel Name')
            for row in rows:
                if len(row) <= channel_index:
                    row.append(str(channel_value))

        # Add the 'Label' column if it doesn't exist
        if 'Label' not in header:
            header.append('Label')
            for row in rows:
                row.append(str(label_value))
        else:
            # If 'Label' exists, make sure to add the value if it's missing
            label_index = header.index('Label')
            for row in rows:
                if len(row) <= label_index:
                    row.append(str(label_value))

        # Write the updated data back to the CSV
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"Processed: {filename}")

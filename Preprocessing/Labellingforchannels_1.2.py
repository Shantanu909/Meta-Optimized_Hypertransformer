import csv


csv_file_path = 'csvs_main\merged.csv'
label_value = 0  

with open(csv_file_path, 'r', newline='') as file:
    reader = list(csv.reader(file))
    header, rows = reader[0], reader[1:]

if 'label' not in header:
    header.append('label')

for row in rows:
    row.append(str(label_value))

with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Label column added with value {label_value} to each row.")

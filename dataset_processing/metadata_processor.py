import csv


class MetadataProcessor:
    def __init__(self, metadata_file_path):
        self.metadata_file_path = metadata_file_path

    def get_column(self, column_num):
        column = []
        with open(self.metadata_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                column.append(row[column_num])
        return column

    def get_row(self, row_num):
        with open(self.metadata_file_path, 'r', newline='') as file:
            rows = list(csv.reader(file))
            return rows[row_num]

    def get_dictionary(self, id_field_num):
        dictionary = {}
        with open(self.metadata_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                key = ''
                value = []
                for i in range(len(row)):
                    if i == id_field_num:
                        key = row[i]
                    else:
                        value.append(row[i])
                dictionary[key] = value
        return dictionary

    def get_field_by_key_field(self, key_field_num, key_field_value, value_field_num):
        records = []
        with open(self.metadata_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                key_field_val = row[key_field_num]
                value_field_val = row[value_field_num]
                records.append((key_field_val, value_field_val))
        for record in records:
            if record[0] == key_field_value:
                return record[1]
        return None


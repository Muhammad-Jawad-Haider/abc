import csv
import pandas as pd


class Patient:
    def __init__(self, id, practice):
        self.data = ''
        self.__id = id
        self.practice = practice

    @property
    def id(self):
        return self.__id

    @property
    def practice(self):
        return self.__practice

    @practice.setter
    def practice(self, practice):
        self.__practice = practice

    def load_data(self, filepath):
        row = get_row_by_column_value(filepath, 'Practice', self.practice)
        self.data = row[row['PatientID'] == self.id]


def get_row_by_column_value(file_path, column_name, target_value):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Find the index of the column
        column_index = reader.fieldnames.index(column_name) if column_name in reader.fieldnames else None

        if column_index is None:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        # Iterate through the rows
        for row in reader:
            # Check if the target value matches the value in the specified column
            if row[column_name] == target_value:
                return pd.DataFrame(row)

    # If the specific row is not found
    return None

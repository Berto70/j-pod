"""Saving the results"""

def save_table_csv(table, filename, columns):
    """
    Save the selected columns, in the given order, as a CSV file.
    """
    out = table[columns]
    for name in out.colnames:
        unit = out[name].unit
        if unit is not None and str(unit) != "":
            out.rename_column(name, f"{name} [{unit}]")
    out.write(filename, format="ascii.csv", overwrite=True)
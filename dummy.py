
def write_to_sql():
    # This function will write data to an SQL database
    query = "INSERT INTO table_name (column1, column2) VALUES (%s, %s)"
    data = ("value1", "value2")
    # Execute the query using a database connection (not shown here)
    return "Data written to SQL database successfully."
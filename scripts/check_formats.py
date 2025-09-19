import sqlite3
import pandas as pd

# Connect to the database
db_path = '../cricket_data.db'
conn = sqlite3.connect(db_path)

# An improved SQL query to group by match_type and count the unique matches in each group
query = """
SELECT 
    match_type, 
    COUNT(DISTINCT match_id) as number_of_matches
FROM 
    batting_innings
GROUP BY 
    match_type
ORDER BY
    number_of_matches DESC
"""

# Execute the query and load the result into a pandas DataFrame
match_formats_with_counts = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

print("Found the following match formats and their counts in the database:")
print(match_formats_with_counts.to_string(index=False))

import sqlite3

conn = sqlite3.connect('database.db')
conn.execute("ALTER TABLE vehicles ADD COLUMN driver_id INTEGER REFERENCES drivers(id)")
conn.close()
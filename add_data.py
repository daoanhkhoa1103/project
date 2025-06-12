import sqlite3

conn = sqlite3.connect('database.db')

# Thêm dữ liệu cho bảng drivers
drivers_data = [
    ('Nguyễn Văn A', '1990-05-10', '0987654321', '123 Đường ABC, Quận 1', 'A1234567', '2020-01-15', '2025-01-15'),
    ('Trần Thị B', '1985-12-20', '0912345678', '456 Đường DEF, Quận 2', 'B7654321', '2018-05-10', '2023-05-10'),
    ('Lê Văn C', '1995-08-15', '0901234567', '789 Đường GHI, Quận 3', 'C1122334', '2021-10-20', '2026-10-20')
]
conn.executemany("INSERT INTO drivers (name, birthdate, phone, address, license_number, license_issued_date, license_expiry_date) VALUES (?,?,?,?,?,?,?)", drivers_data)

# Thêm dữ liệu cho bảng vehicles
vehicles_data = [
    ('51A-123.45', 'Xe tải', 'Hino', 'ABC123456789', 'DEF987654321', 1),  # Thêm driver_id = 1
    ('51B-543.21', 'Xe khách', 'Thaco', 'GHI987654321', 'JKL123456789', 2),  # Thêm driver_id = 2
    ('51C-987.65', 'Xe con', 'Toyota', 'MNO123456789', 'PQR987654321', 3)  # Thêm driver_id = 3
]
conn.executemany("INSERT INTO vehicles (license_plate, vehicle_type, brand, chassis_number, engine_number, driver_id) VALUES (?,?,?,?,?,?)", vehicles_data)

# Thêm dữ liệu cho bảng routes
routes_data = [
    (1, 1, 'TP. Hồ Chí Minh', 'Hà Nội', '2023-02-20 08:00:00', '2023-02-21 17:00:00', 1700, 200),
    (2, 2, 'Đà Nẵng', 'Nha Trang', '2023-02-21 10:00:00', '2023-02-22 14:00:00', 600, 80),
    (3, 3, 'Biên Hòa, Đồng Nai', 'Vũng Tàu', '2023-02-22 12:00:00', '2023-02-22 18:00:00', 100, 15)
]
conn.executemany("INSERT INTO routes (driver_id, vehicle_id, start_point, end_point, start_time, end_time, distance, fuel_consumption) VALUES (?,?,?,?,?,?,?,?)", routes_data)

conn.commit()
conn.close()
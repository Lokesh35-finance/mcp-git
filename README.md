# Now, between New York (EDT/EST) and India (IST)
python3 tz_diff.py --from now --src America/New_York --dst Asia/Kolkata

# A specific timestamp interpreted in the source TZ
python3 tz_diff.py --from "2025-08-17 10:00" --src America/New_York --dst Asia/Kolkata

# Just check current offset difference between two zones
python3 tz_diff.py --src Europe/London --dst Asia/Kolkata

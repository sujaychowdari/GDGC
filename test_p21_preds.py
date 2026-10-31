import pandas as pd
import numpy as np

# We can re-run a quick CatBoost or check what CatBoost predicted for Port 21
# In check_test_preds.py we had:
# service '-' (17407 total):
# pred: Availability: 1232, Benign: 8762, Reconnaissance: 7366, Spoofing: 47!
# And port 21 has service '-' and 10,396 rows!
# Let's check test port 21 rows vs their predicted classes.

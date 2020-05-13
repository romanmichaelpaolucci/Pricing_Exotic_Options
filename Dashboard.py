import tkinter as tk
from tkinter import ttk
#from OptionContract import *

# Main Frame
root = tk.Tk()
root.title('Options Trading Dashboard')

#Create Tab Control
TAB_CONTROL = ttk.Notebook(root)
#Tab1
option_pricing = ttk.Frame(TAB_CONTROL)
TAB_CONTROL.add(option_pricing, text='Option Pricing')
#Tab2
option_simulation = ttk.Frame(TAB_CONTROL)
TAB_CONTROL.add(option_simulation, text='Monte Carlo Simulation')
TAB_CONTROL.pack(expand=1, fill="both")

# Initial Asset Price
initial_asset_price_label = tk.Label(master=option_pricing, text='Initial Asset Price')
initial_asset_price_label.grid(column=0, row=0)

initial_asset_price_sv = tk.StringVar()
initial_asset_price_entry = tk.Entry(master=option_pricing, textvar=initial_asset_price_sv)
initial_asset_price_entry.grid(column=0, row=1)

# Strike Price
strike_price_label = tk.Label(master=option_pricing, text='Strike Price')
strike_price_label.grid(column=0, row=2)

strike_price_sv = tk.StringVar()
strike_price_entry = tk.Entry(master=option_pricing, textvar=strike_price_sv)
strike_price_entry.grid(column=0, row=3)

# Risk Free Rate
risk_free_rate_label = tk.Label(master=option_pricing, text='Risk Free Rate')
risk_free_rate_label.grid(column=0, row=4)

risk_free_rate_sv = tk.StringVar()
risk_free_rate_entry = tk.Entry(master=option_pricing, textvar=risk_free_rate_sv)
risk_free_rate_entry.grid(column=0, row=5)

# Drift
drift_label = tk.Label(master=option_pricing, text='Asset Drift')
drift_label.grid(column=0, row=6)

drift_sv = tk.StringVar()
drift_entry = tk.Entry(master=option_pricing, textvar=drift_sv)
drift_entry.grid(column=0, row=7)

# Volatility
volatility_label = tk.Label(master=option_pricing, text='Asset Volatility')
volatility_label.grid(column=0, row=8)

volatility_sv = tk.StringVar()
volatility_entry = tk.Entry(master=option_pricing, textvar=volatility_sv)
volatility_entry.grid(column=0, row=9)

"""# Simulations
simulation_label = tk.Label(master=root, text='Simulations')
simulation_label.grid(column=0, row=10)

simulation_sv = tk.StringVar()
simulation_entry = tk.Entry(master=root, textvar=simulation_sv)
simulation_entry.grid(column=0, row=11)

# Run Simulations
run_simulations = tk.Button(master=root, text='Run Simulations')
run_simulations.grid(column=0, row=12)"""
""
root.mainloop()

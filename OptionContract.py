import math
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm


class OptionTools:

    def __init__(self):
        pass

    # Return annualized remaining time to maturity and days to maturity for simulations
    def compute_time_to_expiration(self, Y, M, D):
        d0 = date.today()
        d1 = date(Y, M, D)
        delta = d1 - d0
        return delta.days/365, delta.days

    # For neural network learning Black-Scholes
    def generate_random_option(self, n, call=True):
        options = []
        for i in range(0, n):
            # NOTE: These parameters will determine the model's performance and capabilities...
            asset_price = random.randrange(10, 30)
            asset_volatility = random.random()
            strike_price = random.randrange(10, 30)
            time_to_expiration = random.randrange(30, 364)/365 # If we have to many observations expiring tomorrow the model may just predict zero as the option is almost worthless
            risk_free_rate = random.random()
            if call:
                options.append(EuropeanCall(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
            else:
                options.append(EuropeanPut(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        return options

    # Simulate options, returns a set of OptionSimulations
    def simulate_calls(self, n_options, strike_price, initial_asset_price, drift, delta_t, asset_volatility, risk_free_rate, time_to_expiration):
        # List of stochastic processes modeling the underlying asset
        stochastic_processes = []
        # Generate a stochastic process for each option
        for i in range(n_options):
            stochastic_processes.append(StochasticProcess(initial_asset_price, drift, delta_t, asset_volatility)) # Note delta t is annualized

        # Automatically create the required steps in time for the simulation by creating a time_step in the stochastic process if its still greator than 0
        # Make n_time_steps for each stochastic process
        for stochastic_process in stochastic_processes:
            tte = time_to_expiration
            while((tte-stochastic_process.delta_t) > 0):
                # Account for the passing of time
                tte = tte - stochastic_process.delta_t
                # Take a time step in the stochastic process
                stochastic_process.time_step()

        # List of option simulations holding realized option variables at every time step
        option_simulations = []
        # Generate n_options simulations classes to hold each observation
        for i in range(n_options):
            # Create an option simulation for every sample path to hold the option variables (prie, delta, etc...)
            option_simulations.append(OptionSimulation(initial_asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))

        # For each stochastic process realization and option simulation
        for z in range(n_options):
            # Reset the decrement for the next option simulation
            time_to_expiration_var = time_to_expiration
            # Price the option for each asset price in the stochsatic process given by z stored in the option simulation given by z
            for i in range(len(stochastic_processes[z].asset_prices)):
                # Check if we still have time in the option
                if (time_to_expiration_var - stochastic_processes[z].delta_t) > 0: # Avoid loss of percision down to 0
                    # Create a european call to record the variables at the z stochsatic processes's i asset price and other static variables with the z stochastic process
                    e = EuropeanCall(stochastic_processes[z].asset_prices[i], stochastic_processes[z].asset_volatility, strike_price, time_to_expiration_var, risk_free_rate)
                    # Append all variables for the i asset price in this z stochastic process
                    option_simulations[z].option_prices.append(e.price)
                    option_simulations[z].option_deltas.append(e.delta)
                    option_simulations[z].option_gammas.append(e.gamma)
                    option_simulations[z].option_vegas.append(e.vega)
                    option_simulations[z].asset_prices.append(stochastic_processes[z].asset_prices[i])
                # Decrement the time_to_expiration by the step in time within the stochastic process, even though z iterates through each stochasstic process the step in time is constant acorss all of them
                if (time_to_expiration_var - stochastic_processes[z].delta_t) > 0:
                    time_to_expiration_var -= stochastic_processes[z].delta_t
                # Break the loop if we are out of time steps, go to the next stochastic process and price an option simulation for it
                else:
                    break
        # Return the option simulations for further analysis
        return option_simulations

    # Takes a set of option simulations returns a vector output of average option price at end of option life, max simulated price, initial simulated price, and min simulated price
    def simulation_analysis(self, option_simulations):
        initial_option_price = 0
        max_option_price = 0
        average_option_price = 0
        min_option_price = 0
        options_in_the_money = 0
        options_out_of_the_money = 0
        ending_prices = []
        # For each option simulation
        for option_simulation in option_simulations:
            # Set initial option price
            initial_option_price = option_simulation.option_prices[0]
            # Get Max Option Price
            if option_simulation.option_prices[len(option_simulation.option_prices)-1] > max_option_price:
                max_option_price = option_simulation.option_prices[len(option_simulation.option_prices)-1]
            # Get Min option price
            if option_simulation.option_prices[len(option_simulation.option_prices)-1] < min_option_price:
                min_option_price = option_simulation.option_prices[len(option_simulation.option_prices)-1]
            # Store for average ending option price
            ending_prices.append(option_simulation.option_prices[len(option_simulation.option_prices)-1])
        return sum(ending_prices)/len(option_simulations), max_option_price, initial_option_price, min_option_price

    # Returns the probability of exerise after simulation, takes set of option simulations
    def probability_of_exercise(self, option_simulations, call=True):
        exercised = 0
        for option_simulation in option_simulations:
            exercised = exercised + option_simulation.exercise_on_expiration(call)
        return exercised/len(option_simulations)

    # Retrns the max, min, and avg exercise value
    def exercise_value_analysis(self, option_simulations):
        max = 0
        avg = 0
        min = 0
        for o in option_simulations:
            if o.exercise_value() >= max:
                max = o.exercise_value()
            if o.exercise_value() <= min:
                min = o.exercise_value()
            avg += o.exercise_value()
        avg = avg/len(option_simulations)
        return max, avg, min

    # Takes an option simulation set, chart each sample path and the respective variable
    def aggregate_chart_option_simulation(self, option_simulations, asset_prices, option_prices, option_deltas, option_gammas, option_vegas):
        # Sum the amount of variables we are plotting
        subplots = asset_prices + option_prices + option_deltas + option_gammas + option_vegas
        # Create subplots for each variable we are plotting
        fig, axs = plt.subplots(subplots)
        fig.suptitle('Option Simulation Outcome')
        # If the variables is to be charted chart it on an independent axis
        if asset_prices:
            axs[0].set_title('Simulated Asset Prices')
            for o in option_simulations:
                axs[0].plot(o.asset_prices)
                # pick any option simulation and fetch the strike price (same for all simulations)
            axs[0].axhline(y=option_simulations[0].strike_price, color='r', linestyle='-', label='Strike Price')
            # To show strike price label
            axs[0].legend()
        if option_prices:
            axs[1].set_title('Option Prices Consequence of Asset Price Change')
            for o in option_simulations:
                axs[1].plot(o.option_prices)
        if option_deltas:
            axs[2].set_title('Option Deltas Consequence of Asset Price Change')
            for o in option_simulations:
                axs[2].plot(o.option_deltas)
        if option_gammas:
            axs[3].set_title('Option Gammas Consequence of Asset Price Change')
            for o in option_simulations:
                axs[3].plot(o.option_gammas)
        if option_deltas:
            axs[4].set_title('Option Deltas Consequence of Asset Price Change')
            for o in option_simulations:
                axs[4].plot(o.option_vegas)

        fig.subplots_adjust(hspace=.5)
        plt.show()


    # Optimal Re-hedging
    # Returns hedging error from t=1 to expiration
    def simulation_rehedging_analysis(self, option_simulations):
        hedging_errors = []
        for o in option_simulations:
            hedging_errors.append(o.option_deltas[0]*(o.asset_prices[len(o.asset_prices)-1]-o.asset_prices[0]))
        return hedging_errors


# Models the underling asset assuming geometetric brownian motion
class StochasticProcess:

    # Probability of motion in a certain direction
    def motion_probability(self, motion_to):
        if motion_to > self.current_asset_price:
            pass
        elif motion_to <= self.current_asset_price:
            pass

    def time_step(self):
        # Brownian motion is ~N(0, delta_t), np.random.normal takes mean and standard deviation
        dW = np.random.normal(0, math.sqrt(self.delta_t))
        dS = self.drift*self.current_asset_price*self.delta_t + self.asset_volatility*self.current_asset_price*dW
        self.asset_prices.append(self.current_asset_price + dS)
        # Reassign the new current asset price for next time step
        self.current_asset_price = self.current_asset_price + dS

    def __init__(self, asset_price, drift, delta_t, asset_volatility):
        self.current_asset_price = asset_price
        self.asset_prices = []
        self.asset_prices.append(asset_price)
        self.drift = drift
        self.delta_t = delta_t
        self.asset_volatility = asset_volatility


class EuropeanCall:

    def call_delta(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(b*strike_price)) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        return z1

    def call_gamma(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(b*strike_price)) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z2 = z1/(asset_price*asset_volatility*math.sqrt(time_to_expiration))
        return z2

    def call_vega(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(b*strike_price)) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z2 = asset_price*z1*math.sqrt(time_to_expiration)
        return z2

    def call_price(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(b*strike_price)) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z1 = z1*asset_price
        x2 = math.log(asset_price/(b*strike_price)) - .5*(asset_volatility*asset_volatility)*time_to_expiration
        x2 = x2/(asset_volatility*(time_to_expiration**.5))
        z2 = norm.cdf(x2)
        z2 = b*strike_price*z2
        return z1 - z2

    def __init__(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        self.asset_price = asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.price = self.call_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)
        self.delta = self.call_delta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)
        self.gamma = self.call_gamma(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)
        self.vega = self.call_vega(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)


class EuropeanPut:

    def put_delta(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(b*strike_price)) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        return z1 - 1

    def put_gamma(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(b*strike_price)) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z2 = z1/(asset_price*asset_volatility*math.sqrt(time_to_expiration))
        return z2

    def put_vega(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(b*strike_price)) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z2 = asset_price*z1*math.sqrt(time_to_expiration)
        return z2

    def put_price(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log((b*strike_price)/asset_price) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z1 = b*strike_price*z1
        x2 = math.log((b*strike_price)/asset_price) - .5*(asset_volatility*asset_volatility)*time_to_expiration
        x2 = x2/(asset_volatility*(time_to_expiration**.5))
        z2 = norm.cdf(x2)
        z2 = asset_price*z2
        return z1 - z2

    def __init__(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        self.asset_price = asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.price = self.put_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)
        self.delta = self.put_delta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)
        self.gamma = self.put_gamma(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)
        self.vega = self.put_vega(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)


class OptionSimulation:

    def exercise_on_expiration(self, call=True):
        # Call
        if call:
            if self.asset_prices[len(self.asset_prices)-1] > self.strike_price:
                return True
            else:
                return False
        # Put
        else:
            if self.asset_prices[len(self.asset_prices)-1] < self.strike_price:
                return True
            else:
                return False

    def exercise_value(self, call=True):
        # Call
        if call:
            profit = self.asset_prices[len(self.asset_prices)-1] - self.strike_price
            return profit if profit >= 0 else 0

        # Put
        else:
            profit =  self.strike_price - self.asset_prices[len(self.asset_prices)-1]
            return profit if profit >= 0 else 0

    def __init__(
        self, initial_asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        self.initial_asset_price = initial_asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.asset_prices = []
        self.option_prices = []
        self.option_deltas = []
        self.option_gammas = []
        self.option_vegas = []

# Simulate the option
result_set = OptionTools().simulate_calls(10, 80, 50, .2, 1/365, .3, .08, 2)
# If your option is ever worth more than the average of all the simulations sell it immediately...

#OptionTools().aggregate_chart_option_simulation(result_set, True, True, True, True, True)

#s = OptionTools().simulation_analysis(result_set)
k = OptionTools().probability_of_exercise(result_set)
#t = OptionTools().simulation_rehedging_analysis(result_set)
#z = OptionTools().exercise_value_analysis(result_set)
#print(s)
print(k)
#print(np.average(t))
#print(z)

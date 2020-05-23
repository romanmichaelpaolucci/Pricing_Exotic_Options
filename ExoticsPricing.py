import math
import numpy as np
from scipy.stats import norm


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


class BarrierOption:

    def __init__(self, strike, barrier):
        self.strike = strike
        self.knock_out_price = barrier
        self.knock_in_price = barrier
        self.knocked_out = False
        self.knock_in = False


class EuroCall:

    def __init__(self, strike):
        self.strike = strike


class EuroCallSimulation:

    def __init__(self, EuroCall, n_options, initial_asset_price, drift, delta_t, asset_volatility, time_to_expiration, risk_free_rate):
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

        payoffs = []
        # for each stochastic process determine the payoff (if there is one)
        for stochastic_process in stochastic_processes:
            payoff = stochastic_process.asset_prices[len(stochastic_process.asset_prices)-1] - EuroCall.strike
            z = payoff if payoff > 0 else 0
            payoffs.append(z)
        self.price = np.average(payoffs)*math.exp(-time_to_expiration*risk_free_rate)


class CallBarrierOptionUpandOutSimulation:

    def __init__(self, CallBarrierOptionUpandOut, n_options, initial_asset_price, drift, delta_t, asset_volatility, time_to_expiration, risk_free_rate):
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

        payoffs = []
        # for each stochastic process determine the payoff (if there is one)
        for stochastic_process in stochastic_processes:
            for i in range(0, len(stochastic_process.asset_prices)):
                if stochastic_process.asset_prices[i] >= CallBarrierOptionUpandOut.knock_out_price:
                    CallBarrierOptionUpandOut.knocked_out = True
            # After running through the asset prices determine if it got kickedout
            if CallBarrierOptionUpandOut.knocked_out != True:
                payoff = stochastic_process.asset_prices[len(stochastic_process.asset_prices)-1] - CallBarrierOptionUpandOut.strike
                z = payoff if payoff >= 0 else 0
                payoffs.append(z)
            if CallBarrierOptionUpandOut.knocked_out == True:
                payoffs.append(0)
            #Reset the flag
            CallBarrierOptionUpandOut.knocked_out = False
        self.price = np.average(payoffs)*math.exp(-time_to_expiration*risk_free_rate)


class CallBarrierOptionDownandOutSimulation:

    def __init__(self, CallBarrierOptionUpandOut, n_options, initial_asset_price, drift, delta_t, asset_volatility, time_to_expiration, risk_free_rate):
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

        payoffs = []
        # for each stochastic process determine the payoff (if there is one)
        for stochastic_process in stochastic_processes:
            for i in range(0, len(stochastic_process.asset_prices)):
                if stochastic_process.asset_prices[i] <= CallBarrierOptionUpandOut.knock_out_price:
                    CallBarrierOptionUpandOut.knocked_out = True
            # After running through the asset prices determine if it got kickedout
            if CallBarrierOptionUpandOut.knocked_out != True:
                payoff = stochastic_process.asset_prices[len(stochastic_process.asset_prices)-1] - CallBarrierOptionUpandOut.strike
                z = payoff if payoff >= 0 else 0
                payoffs.append(z)
            if CallBarrierOptionUpandOut.knocked_out == True:
                payoffs.append(0)
            #Reset the flag
            CallBarrierOptionUpandOut.knocked_out = False
        self.price = np.average(payoffs)*math.exp(-time_to_expiration*risk_free_rate)


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


print('Black-Scholes European Call price: ', EuropeanCall(296, .234, 295, 1/12, .08).price)
print('Monte Carlo Euro Call price: ', EuroCallSimulation(EuroCall(296), 10000, 295, 0, 1/365, .2435, 39/365, .0017).price)
# Comparing this to the black-scholes for a similar option we get an appropriately priced option!
print('Monte Carlo Up and Out Call: ', CallBarrierOptionUpandOutSimulation(BarrierOption(296, 330), 10000, 295, 0, 1/365, .2435, 39/365, .0017).price)
print('Monte Carlo Down and Out Call: ', CallBarrierOptionDownandOutSimulation(BarrierOption(296, 29), 10000, 295, 0, 1/365, .2435, 39/365, .0017).price)
#print(CallBarrierOptionUpandOutSimulation(CallBarrierOptionUpandOut(30, 40), 10000, 20, .2, 1/365, .2, 1, .08).price)

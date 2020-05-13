# Optimal Delta Re-hedging
Roman Paolucci

Is there a way to construct the portfolio so cumulative attempts to rehedge the portfolio are zero?  If not how can we minimize the cost function to maintain a risk-neutral condition?  Compute a risk neutral expected value for the hedging error?

### Cost of Re-hedging Function


#####$ L(\epsilon ,n)=\epsilon +nT $
- $n =$ Number of Transactions
- $\epsilon = \Sigma_{i=1}^{n}{\epsilon_i} = $ Cumulative Hedging Error
- $\epsilon_k = \delta_t(S_{t+z} - S_{t}) = $ Hedging Error for Transaction $k$

### Hedge Portfolio Value from PDE
$H = S*P + Q*D + (\epsilon +nT)$
- $S = $ Underlying asset Price
- $P = $ Net shares in underlying asset
- $D = $ Option price
- $Q = $ Net quantity in option


### Computing change in hedging error

Use Ito's lemma to find the differential of a time dependent function of a stochastic variable, after some stochastic calculus rules for simplification...

### $d\epsilon_k =$  $\epsilon_{k_t}dt + \epsilon_{k_w} dW_t + \frac{1}{2}\epsilon_{k_{ww}}dt$

Modeling the underlying asset as following GBM
### $d \epsilon_{k_{t}} =$  $\delta_t * (S_{t+z} \mu dt + S_{t+z}\sigma dW_{t} - S_{t} \mu dt - S_{t} \sigma dW_{t})$

The problem is our hedge has these two different prices and stochastic terms we can't hedge away that risk, but can we minimize it?


Model the underlying asset as Geometric Brownian motion

### $d\epsilon_k =$  $\delta_t*(dS_{t+n} - dS_{t})$

### $d\epsilon_k =$ $\delta_t * (S_{t+z} \mu dt + S_{t+z}\sigma dW_{t} - S_{t} \mu dt - S_{t} \sigma dW_{t})$
### $d\epsilon_k = \delta_t[\mu dt(S_{t+z} - S_{t}) + \sigma dW_t(S_{t+z} - S_{t})]$
Factor by grouping
### $d \epsilon_k = \delta_t(\mu dt + \sigma dW_t)(S_{t+z}-S_t)$

### European Option Classes

### Simulating The Underlying Asset

The assumption is the underlying asset the option is being priced for follows Geometric Brownian motion.  The StochasticProcess class models this law of motion.

```Python
def __init__(self, asset_price, drift, delta_t, asset_volatility):
      self.current_asset_price = asset_price
      self.asset_prices = []
      self.asset_prices.append(asset_price)
      self.drift = drift
      self.delta_t = delta_t
      self.asset_volatility = asset_volatility
```
To initialize this class an initial asset price is required along with a forecasted drift, time step length, and implied volatility.  Though drift and implied volatility are set once during the initialization of this class it is possible to make them time varying by altering the simulation functions.
```Python
def time_step(self):
      # Brownian motion is ~N(0, delta_t), np.random.normal takes mean and standard deviation
      dW = np.random.normal(0, math.sqrt(self.delta_t))
      dS = self.drift*self.current_asset_price*self.delta_t + self.asset_volatility*self.current_asset_price*dW
      self.asset_prices.append(self.current_asset_price + dS)
      # Reassign the new current asset price for next time step
      self.current_asset_price = self.current_asset_price + dS
```
The simulation uses the StochasticProcess' internal function time_step which allows for multiple instances of the StochasticProcess to be made and simulated synchronously with EuropeanCalls and EuropeanPuts to be stored in an OptionSimulation instance.

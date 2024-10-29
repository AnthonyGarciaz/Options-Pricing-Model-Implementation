import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta


class OptionsPricer:
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate

    def get_stock_data(self, ticker, lookback_days=90):
        """Fetch and analyze stock data"""
        print(f"\n📊 Fetching data for {ticker}...")

        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        hist = stock.history(start=start_date, end=end_date)

        current_price = hist['Close'].iloc[-1]
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        print(f"✓ Current Stock Price: ${current_price:.2f}")
        print(f"✓ Historical Volatility: {volatility:.1%}")

        return current_price, volatility, hist

    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option price using Black-Scholes formula"""
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def monte_carlo_price(self, S, K, T, r, sigma, option_type='call', simulations=10000):
        """Price option using Monte Carlo simulation"""
        print(f"\n🎲 Running Monte Carlo simulation with {simulations:,} iterations...")

        # Generate random price paths
        Z = np.random.standard_normal(simulations)
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        # Discount payoffs
        price = np.exp(-r * T) * np.mean(payoffs)

        return price, ST

    def plot_analysis(self, S, K, T, hist_prices, ST, option_type, bs_price, mc_price):
        """Create visual analysis of the option pricing"""
        # Use a built-in style
        plt.style.use('bmh')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Historical prices and strike
        ax1.plot(hist_prices.index, hist_prices['Close'], 'b-', label='Historical Stock Price', linewidth=2)
        ax1.axhline(y=K, color='r', linestyle='--', label=f'Strike Price (${K:.2f})', linewidth=2)
        ax1.plot(hist_prices.index[-1], S, 'go', markersize=10, label=f'Current Price (${S:.2f})')

        # Add profit zone with lighter colors
        if option_type == 'call':
            ax1.fill_between(hist_prices.index, K, ax1.get_ylim()[1], alpha=0.2, color='g', label='Profit Zone')
        else:
            ax1.fill_between(hist_prices.index, ax1.get_ylim()[0], K, alpha=0.2, color='g', label='Profit Zone')

        ax1.set_title('Stock Price History and Strike Price', fontsize=14, pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)

        # Plot 2: Monte Carlo price distribution
        n, bins, _ = ax2.hist(ST, bins=50, density=True, alpha=0.7, color='skyblue',
                              label='Simulated Prices', edgecolor='black')
        ax2.axvline(x=K, color='r', linestyle='--', label=f'Strike Price (${K:.2f})', linewidth=2)
        ax2.axvline(x=S, color='g', linestyle='-', label=f'Current Price (${S:.2f})', linewidth=2)

        # Add price labels
        ax2.annotate(f'Strike\n${K:.2f}', xy=(K, max(n)), xytext=(10, 10),
                     textcoords='offset points', ha='left', va='bottom')
        ax2.annotate(f'Current\n${S:.2f}', xy=(S, max(n)), xytext=(10, 10),
                     textcoords='offset points', ha='left', va='bottom')

        ax2.set_title('Monte Carlo Price Distribution at Expiration', fontsize=14, pad=20)
        ax2.set_xlabel('Stock Price ($)', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)

        plt.tight_layout()
        plt.show()

    def analyze_option(self, ticker, strike_price, days_to_expiry, option_type='call'):
        """Complete option analysis with both pricing methods"""
        # Get market data
        S, sigma, hist_prices = self.get_stock_data(ticker)

        # Setup parameters
        K = strike_price
        T = days_to_expiry / 365
        r = self.risk_free_rate

        # Calculate prices using both methods
        bs_price = self.black_scholes(S, K, T, r, sigma, option_type)
        mc_price, ST = self.monte_carlo_price(S, K, T, r, sigma, option_type)

        # Create visualizations
        self.plot_analysis(S, K, T, hist_prices, ST, option_type, bs_price, mc_price)

        # Calculate probability of profit
        prob_profit = np.mean(ST > K) if option_type == 'call' else np.mean(ST < K)

        # Print friendly analysis
        self.print_analysis(S, K, T, sigma, bs_price, mc_price, prob_profit, option_type)

    def print_analysis(self, S, K, T, sigma, bs_price, mc_price, prob_profit, option_type):
        """Print user-friendly analysis"""
        print("\n" + "=" * 50)
        print("📈 OPTIONS PRICING ANALYSIS")
        print("=" * 50)

        print("\n🔑 KEY INFORMATION:")
        print(f"Current Stock Price: ${S:.2f}")
        print(f"Strike Price: ${K:.2f}")
        print(f"Days to Expiration: {T * 365:.0f}")
        print(f"Volatility: {sigma:.1%}")
        print(f"Option Type: {option_type.upper()}")

        print("\n💰 PRICING RESULTS:")
        print(f"Black-Scholes Price: ${bs_price:.2f}")
        print(f"Monte Carlo Price: ${mc_price:.2f}")
        print(f"Average Price: ${(bs_price + mc_price) / 2:.2f}")

        print("\n📊 PROBABILITY ANALYSIS:")
        print(f"Probability of Profit: {prob_profit:.1%}")

        print("\n💡 INTERPRETATION:")
        if option_type == 'call':
            if S > K:
                print("✅ Option is IN THE MONEY")
                print(f"   Currently ${S - K:.2f} in the money")
            else:
                print("⚠️ Option is OUT OF THE MONEY")
                print(f"   Needs ${K - S:.2f} rise to break even")
        else:
            if S < K:
                print("✅ Option is IN THE MONEY")
                print(f"   Currently ${K - S:.2f} in the money")
            else:
                print("⚠️ Option is OUT OF THE MONEY")
                print(f"   Needs ${S - K:.2f} fall to break even")


def main():
    # Initialize pricer
    pricer = OptionsPricer()

    # Analyze an option
    pricer.analyze_option(
        ticker='AAPL',
        strike_price=170,
        days_to_expiry=30,
        option_type='call'
    )


if __name__ == "__main__":
    main()
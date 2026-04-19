#!/usr/bin/env python3
"""
Debug script to understand why expected returns are similar
"""

def simulate_calculation(current_price):
    """Simulate the calculation logic from analyzer.py"""
    
    print(f"\n=== DEBUGGING FOR CURRENT_PRICE = ₹{current_price} ===")
    
    # Initial values (from HOLD recommendation path)
    buy_price = current_price * 0.95  # Wait for 5% dip
    sell_price = current_price * 1.15  # Target 15% gain
    stop_loss = current_price * 0.90   # 10% stop loss
    
    print(f"Initial values:")
    print(f"  buy_price = {buy_price:.2f} (current * 0.95)")
    print(f"  sell_price = {sell_price:.2f} (current * 1.15)")
    print(f"  stop_loss = {stop_loss:.2f} (current * 0.90)")
    
    # Calculate initial risk-reward ratio
    risk = abs(buy_price - stop_loss)
    reward = abs(sell_price - buy_price)
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    print(f"\nInitial risk-reward calculation:")
    print(f"  risk = |{buy_price:.2f} - {stop_loss:.2f}| = {risk:.2f}")
    print(f"  reward = |{sell_price:.2f} - {buy_price:.2f}| = {reward:.2f}")
    print(f"  risk_reward_ratio = {risk_reward_ratio:.2f}")
    
    # Check if risk-reward enforcement kicks in
    if risk_reward_ratio < 2.0:
        print(f"\nRisk-reward ratio {risk_reward_ratio:.2f} < 2.0, enforcing 2.5:1 ratio...")
        
        # Adjust sell price to achieve minimum 2:1 ratio
        risk = abs(buy_price - stop_loss)
        new_sell_price = buy_price + (risk * 2.5)  # 2.5:1 ratio for good trades
        
        print(f"  risk = {risk:.2f}")
        print(f"  new_sell_price = {buy_price:.2f} + ({risk:.2f} * 2.5) = {new_sell_price:.2f}")
        
        sell_price = new_sell_price
        risk_reward_ratio = 2.5
    
    # Calculate expected return percentage
    expected_return_percent = ((sell_price - buy_price) / buy_price) * 100
    
    print(f"\nFinal values:")
    print(f"  buy_price = ₹{buy_price:.2f}")
    print(f"  sell_price = ₹{sell_price:.2f}")
    print(f"  expected_return = {expected_return_percent:.1f}%")
    print(f"  risk_reward_ratio = {risk_reward_ratio:.2f}")
    
    return expected_return_percent

# Test with different stock prices
test_prices = [100, 500, 1000, 5000, 10000]

for price in test_prices:
    expected_return = simulate_calculation(price)

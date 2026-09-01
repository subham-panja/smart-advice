#!/usr/bin/env python3
"""
Position & Pyramid Price Adjustment Tool
File: backend/adjust_bought_price.py

Interactive CLI tool to adjust actual Dhan fill prices for:
1. Initial Position Entries
2. Subsequent Pyramid Additions

Automatically recalculates:
- Weighted Average Entry Price
- Total Position Investment (with brokerage)
- Recalibrated Stop Loss & Target prices
- Initial Risk & Risk % of Capital
- Remaining Account Capital
"""

from datetime import timezone

import config
from database import get_open_positions, update_position
from utils.trading_clock import trading_now


def print_banner():
    print("\n" + "=" * 80)
    print(" " * 22 + "POSITION & PYRAMID PRICE ADJUSTMENT TOOL")
    print("=" * 80)


def fetch_and_display_positions():
    positions = get_open_positions()
    if not positions:
        print("\n❌ No active open positions found in database.")
        return None

    print("\nActive Open Positions:")
    print("-" * 80)
    print(
        f"{'No.':<5} | {'Symbol':<12} | {'Qty':<5} | {'Entry Price':<12} | {'Total Inv':<12} | {'Stop Loss':<12} | {'Pyramids':<8}"
    )
    print("-" * 80)

    for idx, pos in enumerate(positions, 1):
        symbol = pos.get("symbol", "N/A")
        qty = pos.get("quantity", 0)
        entry_price = pos.get("entry_price", 0.0)
        total_inv = pos.get("total_investment", 0.0)
        sl = pos.get("current_stop_loss", pos.get("stop_loss", 0.0))
        pyramids = pos.get("adds_count", 0)

        print(
            f"[{idx}]   | {symbol:<12} | {qty:<5} | ₹{entry_price:<11.2f} | ₹{total_inv:<11.2f} | ₹{sl:<11.2f} | {pyramids:<8}"
        )

    print("-" * 80)
    return positions


def adjust_single_entry(pos):
    symbol = pos["symbol"]
    old_entry = pos.get("entry_price", 0.0)
    old_qty = pos.get("quantity", 0)
    old_total_inv = pos.get("total_investment", 0.0)
    old_sl = pos.get("current_stop_loss", pos.get("stop_loss", old_entry))
    old_target = pos.get("current_target", pos.get("target", old_entry))

    initial_cap = config.TRADING_OPTIONS.get("initial_capital", 10000.0)
    brokerage_pct = config.TRADING_OPTIONS.get("brokerage_charges", 0.0020)

    print(f"\n--- Adjusting Initial Entry for {symbol} ---")
    print(f"Current Entry Price: ₹{old_entry:.2f}")
    print(f"Current Quantity:    {old_qty}")
    print(f"Current Total Inv:   ₹{old_total_inv:.2f}")

    # Prompt user for new fill price
    price_input = input(f"\nEnter actual Dhan fill price (or press Enter to keep ₹{old_entry:.2f}): ").strip()
    if price_input:
        try:
            new_entry = float(price_input)
        except ValueError:
            print("❌ Invalid price entered. Operation cancelled.")
            return
    else:
        new_entry = old_entry

    # Prompt user for new quantity
    qty_input = input(f"Enter actual Quantity (or press Enter to keep {old_qty}): ").strip()
    if qty_input:
        try:
            new_qty = int(qty_input)
        except ValueError:
            print("❌ Invalid quantity entered. Operation cancelled.")
            return
    else:
        new_qty = old_qty

    if new_entry == old_entry and new_qty == old_qty:
        print("No changes specified.")
        return

    # Recalculate values
    new_total_inv = round(new_qty * new_entry * (1 + brokerage_pct), 2)
    sl_dist = old_entry - old_sl
    target_dist = old_target - old_entry
    new_sl = round(new_entry - sl_dist, 2)
    new_target = round(new_entry + target_dist, 2)
    new_initial_risk = round((new_entry - new_sl) * new_qty, 2)
    new_risk_pct = round((new_initial_risk / initial_cap) * 100, 2)

    # Preview changes
    print("\n" + "=" * 60)
    print(f"PREVIEW CONFIRMATION FOR {symbol}:")
    print("=" * 60)
    print(f"Entry Price:     ₹{old_entry:.2f}  -->  ₹{new_entry:.2f}")
    print(f"Quantity:        {old_qty}  -->  {new_qty}")
    print(f"Total Investment:₹{old_total_inv:.2f}  -->  ₹{new_total_inv:.2f}")
    print(f"Stop Loss:       ₹{old_sl:.2f}  -->  ₹{new_sl:.2f}")
    print(f"Target Price:    ₹{old_target:.2f}  -->  ₹{new_target:.2f}")
    print(f"Risk % of Cap:   {pos.get('risk_pct_of_cap', 0)}%  -->  {new_risk_pct}%")
    print("=" * 60)

    confirm = input("Save these changes to MongoDB database? (y/n): ").strip().lower()
    if confirm == "y":
        update_dict = {
            "entry_price": new_entry,
            "quantity": new_qty,
            "initial_quantity": new_qty,
            "total_investment": new_total_inv,
            "stop_loss": new_sl,
            "current_stop_loss": new_sl,
            "target": new_target,
            "current_target": new_target,
            "initial_risk": new_initial_risk,
            "risk_pct_of_cap": new_risk_pct,
            "entry_adjusted_manually": True,
            "updated_at": trading_now(timezone.utc).replace(tzinfo=None),
        }
        update_position(symbol, update_dict)
        print(f"\n✅ Position {symbol} updated successfully!")
    else:
        print("\nOperation cancelled. No changes saved.")


def adjust_pyramid_position(pos):
    symbol = pos["symbol"]
    updates = pos.get("updates", [])

    brokerage_pct = config.TRADING_OPTIONS.get("brokerage_charges", 0.0020)
    initial_cap = config.TRADING_OPTIONS.get("initial_capital", 10000.0)

    legs = []
    # Leg 0: Initial buy
    initial_leg_price = pos.get("initial_entry_price", pos.get("entry_price", 0.0))
    initial_leg_qty = pos.get("initial_quantity", pos.get("quantity", 1))
    legs.append({"leg_name": "Initial Buy (Leg 0)", "price": initial_leg_price, "qty": initial_leg_qty})

    # Leg 1..N: Pyramid adds from updates array
    pyramid_updates = [u for u in updates if u.get("type") in ("PYRAMID_ADD", "ADD")]
    for idx, py_up in enumerate(pyramid_updates, 1):
        legs.append(
            {
                "leg_name": f"Pyramid Add #{idx} (Leg {idx})",
                "price": py_up.get("add_price", py_up.get("current_sl", 0.0)),
                "qty": py_up.get("add_qty", 1),
                "update_ref": py_up,
            }
        )

    print(f"\n--- Execution Legs for {symbol} ---")
    for idx, leg in enumerate(legs):
        print(f"[{idx}] {leg['leg_name']:<25} | Qty: {leg['qty']} | Fill Price: ₹{leg['price']:.2f}")

    leg_idx_input = input(f"\nSelect leg to adjust (0-{len(legs)-1}) or press Enter to cancel: ").strip()
    if not leg_idx_input:
        return
    try:
        leg_idx = int(leg_idx_input)
        if leg_idx < 0 or leg_idx >= len(legs):
            print("❌ Invalid leg index.")
            return
    except ValueError:
        print("❌ Invalid input.")
        return

    target_leg = legs[leg_idx]
    print(f"\nAdjusting {target_leg['leg_name']}:")
    price_input = input(f"Enter actual Dhan fill price (press Enter to keep ₹{target_leg['price']:.2f}): ").strip()
    qty_input = input(f"Enter actual Quantity for this leg (press Enter to keep {target_leg['qty']}): ").strip()

    new_leg_price = float(price_input) if price_input else target_leg["price"]
    new_leg_qty = int(qty_input) if qty_input else target_leg["qty"]

    # Update leg in memory
    legs[leg_idx]["price"] = new_leg_price
    legs[leg_idx]["qty"] = new_leg_qty

    # Recalculate Combined Position Metrics
    total_qty = sum(leg["qty"] for leg in legs)
    total_raw_cost = sum(leg["price"] * leg["qty"] for leg in legs)
    new_weighted_avg_entry = round(total_raw_cost / total_qty, 2)
    new_total_inv = round(total_raw_cost * (1 + brokerage_pct), 2)

    old_avg_entry = pos.get("entry_price", 0.0)
    old_sl = pos.get("current_stop_loss", pos.get("stop_loss", old_avg_entry))
    old_target = pos.get("current_target", pos.get("target", old_avg_entry))

    sl_dist = old_avg_entry - old_sl
    target_dist = old_target - old_avg_entry
    new_sl = round(new_weighted_avg_entry - sl_dist, 2)
    new_target = round(new_weighted_avg_entry + target_dist, 2)
    new_initial_risk = round((new_weighted_avg_entry - new_sl) * total_qty, 2)
    new_risk_pct = round((new_initial_risk / initial_cap) * 100, 2)

    # Preview
    print("\n" + "=" * 60)
    print(f"RECALCULATED COMBINED POSITION PREVIEW FOR {symbol}:")
    print("=" * 60)
    print(f"Weighted Avg Entry Price: ₹{old_avg_entry:.2f}  -->  ₹{new_weighted_avg_entry:.2f}")
    print(f"Total Quantity:           {pos.get('quantity')}  -->  {total_qty}")
    print(f"Total Investment:         ₹{pos.get('total_investment'):.2f}  -->  ₹{new_total_inv:.2f}")
    print(f"Stop Loss:                ₹{old_sl:.2f}  -->  ₹{new_sl:.2f}")
    print(f"Target Price:             ₹{old_target:.2f}  -->  ₹{new_target:.2f}")
    print("=" * 60)

    confirm = input("Save changes to MongoDB database? (y/n): ").strip().lower()
    if confirm == "y":
        update_dict = {
            "entry_price": new_weighted_avg_entry,
            "quantity": total_qty,
            "total_investment": new_total_inv,
            "stop_loss": new_sl,
            "current_stop_loss": new_sl,
            "target": new_target,
            "current_target": new_target,
            "initial_risk": new_initial_risk,
            "risk_pct_of_cap": new_risk_pct,
            "entry_adjusted_manually": True,
            "updated_at": trading_now(timezone.utc).replace(tzinfo=None),
        }
        update_position(symbol, update_dict)
        print(f"\n✅ Position {symbol} updated successfully!")
    else:
        print("\nOperation cancelled. No changes saved.")


def main():
    print_banner()
    while True:
        positions = fetch_and_display_positions()
        if not positions:
            break

        selection = input("\nSelect stock number to adjust (1-%d) or '0' to exit: " % len(positions)).strip()
        if selection == "0" or not selection:
            print("\nExiting adjustment tool. Good luck trading!")
            break

        try:
            pos_idx = int(selection) - 1
            if pos_idx < 0 or pos_idx >= len(positions):
                print("❌ Invalid selection number. Try again.")
                continue
        except ValueError:
            print("❌ Invalid input. Enter a number.")
            continue

        selected_pos = positions[pos_idx]
        adds_count = selected_pos.get("adds_count", 0)

        if adds_count > 0:
            adjust_pyramid_position(selected_pos)
        else:
            adjust_single_entry(selected_pos)


if __name__ == "__main__":
    main()

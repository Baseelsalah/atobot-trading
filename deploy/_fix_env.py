"""Fix .env on VPS: remove ORB, add momentum, add momentum params."""
import re

env_path = "/opt/atobot/.env"

with open(env_path) as f:
    content = f.read()

# Fix STRATEGIES line
content = re.sub(
    r'^STRATEGIES=.*$',
    'STRATEGIES=["vwap_scalp","momentum"]',
    content,
    flags=re.MULTILINE,
)

# Add Momentum params if not present
if "MOMENTUM_RSI_OVERSOLD" not in content:
    momentum_block = """
# -- Momentum Strategy (ULTRA-tuned from backtest: +$681/3mo, 61.9% WR) -------
MOMENTUM_RSI_OVERSOLD=32.0
MOMENTUM_TAKE_PROFIT_PERCENT=2.0
MOMENTUM_STOP_LOSS_PERCENT=1.0
MOMENTUM_VOLUME_MULTIPLIER=1.5
MOMENTUM_ORDER_SIZE_USD=17000.0
"""
    # Insert after the ORB section or VWAP section
    content += momentum_block

with open(env_path, "w") as f:
    f.write(content)

print("Updated .env:")
for line in content.splitlines():
    if "STRATEG" in line or "MOMENTUM" in line:
        print(f"  {line}")
print("Done!")

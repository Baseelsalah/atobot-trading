#!/bin/bash
# AtoBot Validation - With Orders
# Tests Activity Ledger + Trade Pairing + Restart Safety

export VALIDATION_MODE=true
export VALIDATION_ALLOW_ORDERS=true

npx tsx scripts/run_validation.ts

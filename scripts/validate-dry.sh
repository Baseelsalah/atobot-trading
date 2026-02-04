#!/bin/bash
# AtoBot Validation - Dry Run (no orders)
# Tests Activity Ledger + Restart Safety

export VALIDATION_MODE=true
export VALIDATION_ALLOW_ORDERS=false

npx tsx scripts/run_validation.ts

# scripts/train.ps1
# CryptoTrader training launcher for PowerShell.
#
# HOW TO RUN (from the repo root):
#     .\scripts\train.ps1
# If PowerShell blocks the script, run it once like this instead:
#     powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1
#
# Edit the values below, save, run. Each maps to a config field via the
# CT_<SECTION>__<FIELD> environment-variable convention (these override
# config/config.yaml for this run only).

# Always run from the repository root so relative paths resolve.
Set-Location (Join-Path $PSScriptRoot "..")

# --- Data to train on ---------------------------------------------------
# Defaults below reflect the best config found by scripts/sweep.py (profitable
# on the 1h hold-out). Tweak and re-run to keep experimenting.
$Days      = 730          # how many days of history to fetch
$Exchange  = "binance"    # binance | bybit | kraken | coinbase | binanceus
$Symbol    = "BTC/USDT"
$Timeframe = "1h"         # 5m | 15m | 1h ...

# --- Barriers (used for BOTH labels and trade exits) --------------------
$env:CT_BARRIERS__TP_MULT = "1.5"   # take-profit in ATR
$env:CT_BARRIERS__SL_MULT = "1.0"   # stop-loss in ATR
$env:CT_BARRIERS__HORIZON = "15"    # time-exit after N bars

# --- Strategy entry thresholds (permissive; the EV gate is the real filter)
$env:CT_STRATEGY__LONG_THRESHOLD  = "0.50"
$env:CT_STRATEGY__SHORT_THRESHOLD = "0.50"

# --- Risk / cost control ------------------------------------------------
$env:CT_RISK__USE_EV_FILTER       = "true"  # EV gate (needs meta for real P(win))
$env:CT_RISK__MIN_EXPECTED_VALUE  = "0.0"   # raise above 0 for a safety margin
$env:CT_RISK__MAX_LEVERAGE        = "1.0"   # cap notional at 1x equity
$env:CT_RISK__COOLDOWN_BARS       = "3"     # wait N bars after a trade
$env:CT_RISK__RISK_PER_TRADE      = "0.005" # 0.5% of equity risked per trade

# --- Execution costs ----------------------------------------------------
# 0.0004 = taker (market orders, honest default). Uncomment for maker/limit:
# $env:CT_EXECUTION__TAKER_FEE = "0.0002"

# --- Model (LightGBM) ---------------------------------------------------
$env:CT_MODEL__N_ESTIMATORS    = "800"
$env:CT_MODEL__LEARNING_RATE   = "0.02"
$env:CT_MODEL__RANDOM_STATE    = "42"        # fixed seed => reproducible, comparable runs
$env:CT_MODEL__USE_META_LABELING = "true"    # secondary win/lose model; EV gate needs it
# $env:CT_MODEL__CLASS_WEIGHT = "balanced"  # uncomment to A/B test balanced weighting

# --- Run ----------------------------------------------------------------
Write-Host "Training on $Exchange $Symbol $Timeframe, $Days days..." -ForegroundColor Cyan
python scripts/train_model.py --days $Days --exchange $Exchange --symbol $Symbol --timeframe $Timeframe

# --- Clean up the temporary overrides so they don't leak into the session
Get-ChildItem Env: | Where-Object Name -like "CT_*" | ForEach-Object { Remove-Item "Env:$($_.Name)" }

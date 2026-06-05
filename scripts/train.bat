@echo off
REM scripts/train.bat
REM CryptoTrader training launcher for cmd.exe (double-click or: scripts\train.bat)
REM Edit the values below, save, run. Each SET maps to a config field via the
REM CT_<SECTION>__<FIELD> convention (overrides config\config.yaml for this run).

REM Run from the repo root (the folder this script's parent lives in).
cd /d "%~dp0\.."

REM --- Data to train on (defaults = best config from scripts/sweep.py) ---
set DAYS=730
set EXCHANGE=binance
set SYMBOL=BTC/USDT
set TIMEFRAME=1h

REM --- Barriers (labels + exits) ---
set CT_BARRIERS__TP_MULT=1.5
set CT_BARRIERS__SL_MULT=1.0
set CT_BARRIERS__HORIZON=15

REM --- Strategy thresholds (permissive; EV gate is the real filter) ---
set CT_STRATEGY__LONG_THRESHOLD=0.50
set CT_STRATEGY__SHORT_THRESHOLD=0.50

REM --- Risk / cost control ---
set CT_RISK__USE_EV_FILTER=true
set CT_RISK__MIN_EXPECTED_VALUE=0.0
set CT_RISK__MAX_LEVERAGE=1.0
set CT_RISK__COOLDOWN_BARS=3
set CT_RISK__RISK_PER_TRADE=0.005

REM --- Execution: 0.0004 taker (honest); uncomment next line for maker ---
REM set CT_EXECUTION__TAKER_FEE=0.0002

REM --- Model (LightGBM) ---
set CT_MODEL__N_ESTIMATORS=800
set CT_MODEL__LEARNING_RATE=0.02
set CT_MODEL__RANDOM_STATE=42
set CT_MODEL__USE_META_LABELING=true
REM set CT_MODEL__CLASS_WEIGHT=balanced   (uncomment to A/B test balanced weighting)

echo Training on %EXCHANGE% %SYMBOL% %TIMEFRAME%, %DAYS% days...
python scripts\train_model.py --days %DAYS% --exchange %EXCHANGE% --symbol %SYMBOL% --timeframe %TIMEFRAME%

pause
